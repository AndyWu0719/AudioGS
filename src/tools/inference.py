"""
AGS Inference Engine.

Provides a unified interface for:
1. Text-to-Speech generation (Text → Atoms → Audio)
2. Audio inversion (Audio → Atoms)
3. Atom editing (pitch shift, time stretch)

Usage:
    from src.tools.inference import AGSInferenceEngine
    
    engine = AGSInferenceEngine("path/to/checkpoint.pt", device="cuda")
    
    # Generate TTS
    audio = engine.generate("Hello world", speaker_id=0)
    
    # Invert audio
    atoms = engine.invert_audio("path/to/audio.wav")
    
    # Edit atoms
    edited = engine.edit_atoms(atoms, pitch_shift=2, speed_rate=0.8)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
import yaml

# Setup paths
TOOLS_DIR = Path(__file__).parent.absolute()
SRC_DIR = TOOLS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

from models.atom import AudioGSModel
from models.flow_dit import get_flow_model
from models.flow_matching import FlowODESolver
from data.text_encoder import CharacterTokenizer, TextEncoder
from data.dataset import AtomNormalizer
from losses.spectral_loss import CombinedAudioLoss
from utils.density_control import AdaptiveDensityController, rebuild_optimizer_from_model

# Optional: CUDA renderer
try:
    from cuda_gabor import get_cuda_gabor_renderer
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False


class AGSInferenceEngine:
    """
    Audio Gaussian Splatting Inference Engine.
    
    Bridges Flow DiT generation (Stage 2) with Gabor rendering (Stage 1).
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize the inference engine.
        
        Args:
            checkpoint_path: Path to Flow DiT checkpoint
            config_path: Path to config YAML (optional, will try to load from checkpoint dir)
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load config
        if config_path:
            self.config = self._load_config(config_path)
        else:
            # Try to find config in checkpoint directory
            config_candidates = [
                self.checkpoint_path.parent.parent / "config.yaml",
                self.checkpoint_path.parent / "config.yaml",
                PROJECT_ROOT / "configs" / "flow_config.yaml",
            ]
            for candidate in config_candidates:
                if candidate.exists():
                    self.config = self._load_config(str(candidate))
                    break
            else:
                raise FileNotFoundError("Config file not found. Please provide config_path.")
        
        self.sample_rate = self.config.get('output', {}).get('sample_rate', 24000)
        
        # Load checkpoint first to get speaker mapping
        self._preload_checkpoint()
        
        # Initialize components with correct speaker count
        self._init_models()
        self._load_checkpoint()
        
        print(f"[AGSInferenceEngine] Initialized on {device}")
        print(f"[AGSInferenceEngine] Sample rate: {self.sample_rate}")
        print(f"[AGSInferenceEngine] Speakers: {len(self.speaker_to_id)}")
    
    def _preload_checkpoint(self):
        """Pre-load checkpoint to extract metadata."""
        ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        self.speaker_to_id = ckpt.get('speaker_to_id', {})
        self._num_speakers = len(self.speaker_to_id) + 10  # Add buffer
    
    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML."""
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _init_models(self):
        """Initialize all models."""
        model_cfg = self.config['model']
        text_cfg = self.config['text_encoder']
        
        # Tokenizer
        self.tokenizer = CharacterTokenizer()
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=text_cfg['embed_dim'],
            hidden_dim=text_cfg['hidden_dim'],
            num_layers=text_cfg['num_layers'],
            num_heads=text_cfg['num_heads'],
        ).to(self.device)
        
        # Flow DiT model - use speaker count from checkpoint
        num_anchors = model_cfg.get('num_anchors', 2048)
        split_factor = model_cfg.get('split_factor', 8)
        
        self.model = get_flow_model(
            size=model_cfg['size'],
            num_speakers=self._num_speakers,  # From preloaded checkpoint
            text_dim=text_cfg['hidden_dim'],
            num_anchors=num_anchors,
            split_factor=split_factor,
        ).to(self.device)
        
        # ODE solver
        flow_cfg = self.config.get('flow', {})
        self.solver = FlowODESolver(
            self.model, 
            sigma_min=flow_cfg.get('sigma_min', 1e-4)
        )
        
        # Atom normalizer
        self.normalizer = AtomNormalizer(sample_rate=self.sample_rate)
        
        # CUDA renderer
        if RENDERER_AVAILABLE:
            self.renderer = get_cuda_gabor_renderer(sample_rate=self.sample_rate)
        else:
            self.renderer = None
            print("[Warning] CUDA renderer not available, audio generation disabled")
        
        # Set to eval mode
        self.model.eval()
        self.text_encoder.eval()
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(ckpt['model'])
        self.text_encoder.load_state_dict(ckpt['text_encoder'])
        
        # Load speaker mapping if available
        self.speaker_to_id = ckpt.get('speaker_to_id', {})
        
        print(f"[AGSInferenceEngine] Loaded checkpoint: {self.checkpoint_path.name}")
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        speaker_id: int = 0,
        steps: int = 25,
        method: str = 'rk4',
        temperature: float = 1.0,
        duration: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate audio from text (TTS).
        
        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID for multi-speaker TTS
            steps: Number of ODE solver steps
            method: Solver method ('euler', 'midpoint', 'rk4')
            temperature: Sampling temperature (1.0 = no scaling)
            duration: Target duration in seconds (optional)
            
        Returns:
            audio: Generated audio waveform [T]
        """
        if self.renderer is None:
            raise RuntimeError("CUDA renderer not available")
        
        # Encode text - get duration prediction
        text_batch = self.tokenizer.batch_encode([text], max_length=256, return_tensors=True)
        input_ids = text_batch['input_ids'].to(self.device)
        attention_mask = text_batch['attention_mask'].to(self.device)
        
        text_emb, _, log_duration_pred = self.text_encoder(input_ids, attention_mask)
        
        # Use predicted duration if not provided
        if duration is None:
            # Convert from log scale
            duration = torch.exp(log_duration_pred).item()
            duration = max(duration, 1.0)
            duration = min(duration, 10.0)  # Cap at 10 seconds
        
        # Speaker ID
        speaker_ids = torch.tensor([speaker_id], device=self.device)
        
        # Generate atoms via Flow
        total_atoms = self.model.total_atoms
        
        atoms_norm = self.solver.sample(
            shape=(1, total_atoms, 6),
            num_steps=steps,
            method=method,
            device=self.device,
            speaker_ids=speaker_ids,
            text_embeddings=text_emb,
            text_mask=attention_mask.bool(),
        )
        
        # Temperature scaling (before denormalization)
        if temperature != 1.0:
            atoms_norm = atoms_norm * temperature
        
        # Denormalize
        atoms = self.normalizer.denormalize(atoms_norm)[0]  # [N, 6]
        
        # Scale tau by predicted duration (tau is normalized to [0, 1])
        # atoms[:, 0] is already in [0, 1], will be scaled in _render_atoms
        
        # Render audio
        audio = self._render_atoms(atoms, duration)
        
        return audio
    
    def invert_audio(
        self,
        audio_path: str,
        steps: int = 2000,
        config: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Invert audio to atom representation (Analysis-by-Synthesis).
        
        Args:
            audio_path: Path to input audio file
            steps: Number of optimization steps
            config: Optional config override
            
        Returns:
            atoms: Optimized atom parameters [N, 6]
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        waveform = waveform.to(self.device)
        audio_duration = len(waveform) / self.sample_rate
        num_samples = len(waveform)
        
        # Use config from checkpoint or provided
        cfg = config or self.config
        
        # Initialize AudioGS model
        initial_atoms = cfg.get('model', {}).get('initial_num_atoms', 4000)
        max_atoms = cfg.get('model', {}).get('max_num_atoms', 20000)
        
        model = AudioGSModel(
            num_atoms=initial_atoms,
            sample_rate=self.sample_rate,
            audio_duration=audio_duration,
            device=self.device,
        )
        model.initialize_from_audio(waveform)
        
        # Loss function
        loss_fn = CombinedAudioLoss(
            sample_rate=self.sample_rate,
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_lengths=[512, 1024, 2048],
        ).to(self.device)
        
        # Optimizer
        lr_config = {
            'lr_amplitude': 0.01,
            'lr_position': 0.0001,
            'lr_frequency': 0.01,
            'lr_sigma': 0.01,
            'lr_phase': 0.02,
            'lr_chirp': 0.002,
        }
        optimizer = rebuild_optimizer_from_model(model, torch.optim.Adam, lr_config)
        
        # Density controller
        density_controller = AdaptiveDensityController(
            grad_threshold=0.0002,
            sigma_split_threshold=0.01,
            prune_amplitude_threshold=0.001,
            max_num_atoms=max_atoms,
        )
        
        # Optimization loop
        for iteration in range(steps):
            amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
            
            pred_waveform = self.renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
            
            loss, _ = loss_fn(pred_waveform, waveform, model_amplitude=amplitude, model_sigma=sigma)
            
            optimizer.zero_grad()
            loss.backward()
            model.accumulate_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Density control
            density_controller.update_thresholds(loss.item())
            if 500 <= iteration < 4000 and iteration % 100 == 0:
                density_controller.densify_and_prune(model, optimizer)
        
        # Extract final atoms
        with torch.no_grad():
            amplitude, tau, omega, sigma, phi, gamma = model.get_all_params()
            tau_norm = tau / audio_duration
            tau_norm = tau_norm.clamp(0, 1)
            
            atoms = torch.stack([tau_norm, omega, sigma, amplitude, phi, gamma], dim=1)
            sorted_indices = torch.argsort(atoms[:, 0])
            atoms = atoms[sorted_indices]
        
        return atoms.cpu()
    
    def edit_atoms(
        self,
        atoms: torch.Tensor,
        pitch_shift: float = 0,
        speed_rate: float = 1.0,
    ) -> torch.Tensor:
        """
        Edit atom parameters for pitch/speed modification.
        
        Args:
            atoms: Input atoms [N, 6] (tau_norm, omega, sigma, amplitude, phi, gamma)
            pitch_shift: Semitones to shift pitch (positive = higher)
            speed_rate: Speed multiplier (>1 = faster, <1 = slower)
            
        Returns:
            edited_atoms: Modified atoms [N, 6]
        """
        atoms = atoms.clone()
        
        # Pitch shift: modify omega (frequency)
        if pitch_shift != 0:
            # semitones to frequency ratio: 2^(semitones/12)
            ratio = 2 ** (pitch_shift / 12)
            atoms[:, 1] = atoms[:, 1] * ratio  # omega *= ratio
        
        # Speed change: modify tau and sigma
        if speed_rate != 1.0:
            # tau (position) scales inversely
            atoms[:, 0] = atoms[:, 0] / speed_rate
            # sigma (width) scales inversely 
            atoms[:, 2] = atoms[:, 2] / speed_rate
        
        return atoms
    
    def _render_atoms(
        self,
        atoms: torch.Tensor,
        duration: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Render atoms to audio.
        
        Args:
            atoms: Atom parameters [N, 6]
            duration: Target duration in seconds
            
        Returns:
            audio: Rendered waveform [T]
        """
        if self.renderer is None:
            raise RuntimeError("CUDA renderer not available")
        
        atoms = atoms.to(self.device)
        
        # Estimate duration from tau if not provided
        if duration is None:
            tau_max = atoms[:, 0].max().item()
            duration = max(tau_max + 0.5, 1.0)  # Add margin
        
        num_samples = int(duration * self.sample_rate)
        
        # Denormalize tau
        tau = atoms[:, 0] * duration
        omega = atoms[:, 1]
        sigma = atoms[:, 2]
        amplitude = atoms[:, 3]
        phi = atoms[:, 4]
        gamma = atoms[:, 5]
        
        # Render
        audio = self.renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
        
        return audio.cpu()
    
    def atoms_to_audio(
        self,
        atoms: torch.Tensor,
        duration: float,
    ) -> torch.Tensor:
        """
        Public method to render atoms to audio.
        
        Args:
            atoms: Atom parameters [N, 6]
            duration: Audio duration in seconds
            
        Returns:
            audio: Rendered waveform [T]
        """
        return self._render_atoms(atoms, duration)
    
    def save_audio(
        self,
        audio: torch.Tensor,
        path: str,
        sample_rate: Optional[int] = None,
    ):
        """Save audio to file."""
        sr = sample_rate or self.sample_rate
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save(path, audio.cpu(), sr)
        print(f"[AGSInferenceEngine] Saved audio to {path}")
