/*
 * PyTorch Extension Bindings for Gabor Atom Rendering
 */

#include <torch/extension.h>
#include "gabor_render.h"

// Forward declaration
torch::Tensor gabor_render_forward(
    torch::Tensor amplitude,
    torch::Tensor tau,
    torch::Tensor omega,
    torch::Tensor sigma,
    torch::Tensor phi,
    torch::Tensor gamma,
    int num_samples,
    float sample_rate,
    float sigma_mult
) {
    // Validate inputs
    TORCH_CHECK(amplitude.is_cuda(), "amplitude must be CUDA tensor");
    TORCH_CHECK(tau.is_cuda(), "tau must be CUDA tensor");
    TORCH_CHECK(omega.is_cuda(), "omega must be CUDA tensor");
    TORCH_CHECK(sigma.is_cuda(), "sigma must be CUDA tensor");
    TORCH_CHECK(phi.is_cuda(), "phi must be CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA tensor");
    
    TORCH_CHECK(amplitude.is_contiguous(), "amplitude must be contiguous");
    TORCH_CHECK(tau.is_contiguous(), "tau must be contiguous");
    TORCH_CHECK(omega.is_contiguous(), "omega must be contiguous");
    TORCH_CHECK(sigma.is_contiguous(), "sigma must be contiguous");
    TORCH_CHECK(phi.is_contiguous(), "phi must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    
    const int num_atoms = amplitude.size(0);
    
    // Create output tensor
    auto output = torch::zeros({num_samples}, amplitude.options());
    
    if (num_atoms > 0) {
        gabor_render_forward_cuda(
            amplitude.data_ptr<float>(),
            tau.data_ptr<float>(),
            omega.data_ptr<float>(),
            sigma.data_ptr<float>(),
            phi.data_ptr<float>(),
            gamma.data_ptr<float>(),
            output.data_ptr<float>(),
            num_atoms,
            num_samples,
            sample_rate,
            sigma_mult
        );
    }
    
    return output;
}

std::vector<torch::Tensor> gabor_render_backward(
    torch::Tensor amplitude,
    torch::Tensor tau,
    torch::Tensor omega,
    torch::Tensor sigma,
    torch::Tensor phi,
    torch::Tensor gamma,
    torch::Tensor grad_output,
    float sample_rate,
    float sigma_mult
) {
    // Validate inputs
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    
    const int num_atoms = amplitude.size(0);
    const int num_samples = grad_output.size(0);
    
    // Create gradient tensors
    auto grad_amplitude = torch::zeros_like(amplitude);
    auto grad_tau = torch::zeros_like(tau);
    auto grad_omega = torch::zeros_like(omega);
    auto grad_sigma = torch::zeros_like(sigma);
    auto grad_phi = torch::zeros_like(phi);
    auto grad_gamma = torch::zeros_like(gamma);
    
    if (num_atoms > 0) {
        gabor_render_backward_cuda(
            amplitude.data_ptr<float>(),
            tau.data_ptr<float>(),
            omega.data_ptr<float>(),
            sigma.data_ptr<float>(),
            phi.data_ptr<float>(),
            gamma.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_amplitude.data_ptr<float>(),
            grad_tau.data_ptr<float>(),
            grad_omega.data_ptr<float>(),
            grad_sigma.data_ptr<float>(),
            grad_phi.data_ptr<float>(),
            grad_gamma.data_ptr<float>(),
            num_atoms,
            num_samples,
            sample_rate,
            sigma_mult
        );
    }
    
    return {grad_amplitude, grad_tau, grad_omega, grad_sigma, grad_phi, grad_gamma};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gabor_render_forward, "Gabor render forward (CUDA)");
    m.def("backward", &gabor_render_backward, "Gabor render backward (CUDA)");
}
