/*
 * Gabor Atom Rendering - C++ Header
 */

#ifndef GABOR_RENDER_H
#define GABOR_RENDER_H

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
);

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
);

#endif // GABOR_RENDER_H
