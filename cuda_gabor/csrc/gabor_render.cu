/*
 * Gabor Atom Rendering CUDA Kernels
 * 
 * High-performance forward and backward kernels for Gabor atom rendering.
 * Similar architecture to 3DGS's diff-gaussian-rasterization.
 * 
 * Gabor atom equation:
 *   g(t) = A * exp(-(t-τ)²/(2σ²)) * cos(2π(ω(t-τ) + ½γ(t-τ)²) + φ)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define PI 3.14159265358979323846f
#define TWO_PI 6.28318530717958647692f

// =============================================================================
// FORWARD KERNEL (WINDOWED - OPTIMIZED)
// =============================================================================
// Key optimization: Each thread handles ONE ATOM, writes only to its window.
// This is O(window_size) per atom instead of O(num_atoms) per sample.
// Uses atomicAdd since multiple atoms may contribute to the same sample.

__global__ void gabor_forward_kernel(
    const float* __restrict__ amplitude,  // [N]
    const float* __restrict__ tau,         // [N]
    const float* __restrict__ omega,       // [N]
    const float* __restrict__ sigma,       // [N]
    const float* __restrict__ phi,         // [N]
    const float* __restrict__ gamma,       // [N]
    float* __restrict__ output,            // [T]
    const int num_atoms,
    const int num_samples,
    const float sample_rate,
    const float sigma_mult
) {
    const int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= num_atoms) return;
    
    // Load atom parameters
    const float A = amplitude[atom_idx];
    const float tau_i = tau[atom_idx];
    const float omega_i = omega[atom_idx];
    const float sigma_i = sigma[atom_idx];
    const float phi_i = phi[atom_idx];
    const float gamma_i = gamma[atom_idx];
    
    // Skip atoms with negligible amplitude
    if (fabsf(A) < 1e-8f) return;
    
    const float sigma_sq = sigma_i * sigma_i + 1e-8f;
    const float window_bound = sigma_i * sigma_mult;
    
    // Compute window bounds in sample indices
    const int window_start = max(0, (int)((tau_i - window_bound) * sample_rate));
    const int window_end = min(num_samples - 1, (int)((tau_i + window_bound) * sample_rate));
    
    // Loop only over samples in this atom's window
    for (int tid = window_start; tid <= window_end; tid++) {
        const float t = (float)tid / sample_rate;
        const float t_centered = t - tau_i;
        const float t_sq = t_centered * t_centered;
        
        // Envelope
        const float envelope = expf(-t_sq / (2.0f * sigma_sq));
        
        // Phase and carrier
        const float phase = TWO_PI * (omega_i * t_centered + 0.5f * gamma_i * t_sq) + phi_i;
        const float carrier = cosf(phase);
        
        // Accumulate contribution using atomicAdd (thread-safe)
        atomicAdd(&output[tid], A * envelope * carrier);
    }
}


// =============================================================================
// BACKWARD KERNEL
// =============================================================================

__global__ void gabor_backward_kernel(
    const float* __restrict__ amplitude,
    const float* __restrict__ tau,
    const float* __restrict__ omega,
    const float* __restrict__ sigma,
    const float* __restrict__ phi,
    const float* __restrict__ gamma,
    const float* __restrict__ grad_output,  // [T]
    float* __restrict__ grad_amplitude,      // [N]
    float* __restrict__ grad_tau,            // [N]
    float* __restrict__ grad_omega,          // [N]
    float* __restrict__ grad_sigma,          // [N]
    float* __restrict__ grad_phi,            // [N]
    float* __restrict__ grad_gamma,          // [N]
    const int num_atoms,
    const int num_samples,
    const float sample_rate,
    const float sigma_mult
) {
    const int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= num_atoms) return;
    
    const float A = amplitude[atom_idx];
    const float tau_i = tau[atom_idx];
    const float omega_i = omega[atom_idx];
    const float sigma_i = sigma[atom_idx];
    const float phi_i = phi[atom_idx];
    const float gamma_i = gamma[atom_idx];
    
    const float sigma_sq = sigma_i * sigma_i + 1e-8f;
    const float window_bound = sigma_i * sigma_mult;
    
    // Compute window bounds in sample indices
    const int window_start = max(0, (int)((tau_i - window_bound) * sample_rate));
    const int window_end = min(num_samples - 1, (int)((tau_i + window_bound) * sample_rate));
    
    // Gradient accumulators
    float grad_A_acc = 0.0f;
    float grad_tau_acc = 0.0f;
    float grad_omega_acc = 0.0f;
    float grad_sigma_acc = 0.0f;
    float grad_phi_acc = 0.0f;
    float grad_gamma_acc = 0.0f;
    
    // Loop only over samples in window (optimization!)
    for (int tid = window_start; tid <= window_end; tid++) {
        const float t = (float)tid / sample_rate;
        const float t_centered = t - tau_i;
        const float t_sq = t_centered * t_centered;
        
        const float grad_out = grad_output[tid];
        
        // Envelope
        const float envelope = expf(-t_sq / (2.0f * sigma_sq));
        
        // Phase and carrier
        const float phase = TWO_PI * (omega_i * t_centered + 0.5f * gamma_i * t_sq) + phi_i;
        const float carrier = cosf(phase);
        const float sin_phase = sinf(phase);
        
        // Envelope derivatives
        // d(envelope)/d(tau) = envelope * (t-tau)/sigma^2 * d(t-tau)/d(tau)
        //                    = envelope * (t-tau)/sigma^2 * (-1)  [chain rule!]
        const float d_envelope_d_tau = -envelope * t_centered / sigma_sq;  // NOTE: negative!
        const float d_envelope_d_sigma = envelope * t_sq / (sigma_i * sigma_sq);

        
        // Carrier derivatives
        const float d_carrier_d_tau = sin_phase * TWO_PI * (omega_i + gamma_i * t_centered);
        const float d_carrier_d_omega = -sin_phase * TWO_PI * t_centered;
        const float d_carrier_d_phi = -sin_phase;
        const float d_carrier_d_gamma = -sin_phase * PI * t_sq;
        
        // Accumulate gradients
        grad_A_acc += grad_out * envelope * carrier;
        grad_tau_acc += grad_out * A * (d_envelope_d_tau * carrier + envelope * d_carrier_d_tau);
        grad_omega_acc += grad_out * A * envelope * d_carrier_d_omega;
        grad_sigma_acc += grad_out * A * d_envelope_d_sigma * carrier;
        grad_phi_acc += grad_out * A * envelope * d_carrier_d_phi;
        grad_gamma_acc += grad_out * A * envelope * d_carrier_d_gamma;
    }
    
    // Write gradients
    grad_amplitude[atom_idx] = grad_A_acc;
    grad_tau[atom_idx] = grad_tau_acc;
    grad_omega[atom_idx] = grad_omega_acc;
    grad_sigma[atom_idx] = grad_sigma_acc;
    grad_phi[atom_idx] = grad_phi_acc;
    grad_gamma[atom_idx] = grad_gamma_acc;
}


// =============================================================================
// C++ WRAPPER FUNCTIONS
// =============================================================================

void gabor_render_forward_cuda(
    const float* amplitude,
    const float* tau,
    const float* omega,
    const float* sigma,
    const float* phi,
    const float* gamma,
    float* output,
    int num_atoms,
    int num_samples,
    float sample_rate,
    float sigma_mult
) {
    // Zero output buffer before accumulation (required for atomicAdd)
    cudaMemset(output, 0, num_samples * sizeof(float));
    
    // Launch kernel with one thread per ATOM (not per sample)
    const int num_blocks = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    gabor_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(
        amplitude, tau, omega, sigma, phi, gamma,
        output,
        num_atoms, num_samples, sample_rate, sigma_mult
    );
}

void gabor_render_backward_cuda(
    const float* amplitude,
    const float* tau,
    const float* omega,
    const float* sigma,
    const float* phi,
    const float* gamma,
    const float* grad_output,
    float* grad_amplitude,
    float* grad_tau,
    float* grad_omega,
    float* grad_sigma,
    float* grad_phi,
    float* grad_gamma,
    int num_atoms,
    int num_samples,
    float sample_rate,
    float sigma_mult
) {
    const int num_blocks = (num_atoms + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    gabor_backward_kernel<<<num_blocks, BLOCK_SIZE>>>(
        amplitude, tau, omega, sigma, phi, gamma,
        grad_output,
        grad_amplitude, grad_tau, grad_omega,
        grad_sigma, grad_phi, grad_gamma,
        num_atoms, num_samples, sample_rate, sigma_mult
    );
}
