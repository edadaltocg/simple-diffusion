import torch

from simple_diffusion.diffusion import GaussianDiffusionProcess
from . import utils


@torch.no_grad()
def p_sample(model, x, t, diffusion_process: GaussianDiffusionProcess):
    r"""
    Reverse denoising process.

    $$
        \mu_\theta\left(\mathbf{x}_t, t\right)=\frac{1}{\sqrt{\alpha_t}}\left(\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\left(\mathbf{x}_t, t\right)\right)\right.
    $$

        Args:
            model (_type_): _description_
            x (_type_): _description_
            t (_type_): _description_
            diffusion_process (GaussianDiffusionProcess): _description_

        Returns:
            _type_: _description_
    """
    t = torch.tensor([t])
    # Factor to the model output
    eps_factor = (1 - utils.extract(diffusion_process.alphas, t, x)) / utils.extract(
        diffusion_process.sqrt_one_minus_alphas_cumprod, t, x
    )
    # Model output
    eps_theta = model(x, t)
    # Final values
    mean = (1 / utils.extract(diffusion_process.alphas, t, x).sqrt()) * (
        x - (eps_factor * eps_theta)
    )
    # Generate z
    z = torch.randn_like(x)
    # Fixed sigma
    sigma_t = utils.extract(diffusion_process.betas, t, x).sqrt()
    sample = mean + sigma_t * z
    return sample


@torch.no_grad()
def p_sample_loop(model, shape, diffusion_process: GaussianDiffusionProcess):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(diffusion_process.timesteps)):
        cur_x = p_sample(model, cur_x, i, diffusion_process)
        x_seq.append(cur_x)
    return x_seq
