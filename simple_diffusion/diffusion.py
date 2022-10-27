import functools

import torch
import torch.nn.functional as F
from torch import Tensor

from . import utils


# Noise schedules
def cosine_beta_schedule(timesteps):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    s = 0.008
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=1e-6, beta_end=1e-3):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def make_beta_schedule(schedule, **kwargs):
    if schedule == "cosine":
        return functools.partial(cosine_beta_schedule, **kwargs)
    elif schedule == "linear":
        return functools.partial(linear_beta_schedule, **kwargs)
    elif schedule == "quadratic":
        return functools.partial(quadratic_beta_schedule, **kwargs)
    elif schedule == "sigmoid":
        return functools.partial(sigmoid_beta_schedule, **kwargs)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


class GaussianDiffusionProcess:
    """Diffusion process"""

    def __init__(self, timesteps: int, beta_schedule: str) -> None:
        r"""

        Args:
            timesteps (int): number of timesteps T in the forward diffusion process
            beta_schedule (Callable): function that takes the number of timesteps and returns a tensor of shape (T,)

        Attributes:

            $$
            \alpha_t=1-\beta_t
            $$
        """
        self.timesteps = timesteps
        # define betas
        self.betas = make_beta_schedule(beta_schedule)(timesteps)

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat(
                [self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]
            )
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start: Tensor, t: Tensor, noise=None):
        r"""
        Sample from the diffusion process at time t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = utils.extract(self.sqrt_alphas_cumprod, t, x_start)
        sqrt_one_minus_alphas_cumprod_t = utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start: Tensor, x_t: Tensor, t: Tensor):
        """
        Compute the mean and variance of the diffusion posterior:

        $$
            q(x_{t-1} | x_t, x_0)
        $$
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            utils.extract(self.posterior_mean_coef1, t, x_t) * x_start
            + utils.extract(self.posterior_mean_coef2, t, x_t) * x_t
        )
        posterior_variance = utils.extract(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = utils.extract(
            self.posterior_log_variance_clipped, t, x_t
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_mean_variance(self, x_start: Tensor, t: Tensor):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps minus 1. Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = utils.extract(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = utils.extract(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = utils.extract(self.log_one_minus_alphas_cumprod, t, x_start)
        return mean, variance, log_variance

    def predict_xstart_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor):
        assert x_t.shape == eps.shape
        return (
            utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t
            - utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t) * eps
        )

    def predict_eps_from_xstart(self, x_t: Tensor, t: Tensor, x_start: Tensor):
        return (
            utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t - x_start
        ) / utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t)


def test():
    process = GaussianDiffusionProcess(100, "cosine")
    print(process.betas.shape)
    print(process.alphas.shape)
    print(process.alphas_cumprod.shape)
    print(process.alphas_cumprod_prev.shape)
    print(process.sqrt_recip_alphas.shape)
    print(process.sqrt_alphas_cumprod.shape)
    print(process.sqrt_one_minus_alphas_cumprod.shape)
    print(process.posterior_variance.shape)

    x_start = torch.randn(1, 3, 32, 32)
    t = torch.randint(0, 100, (1,))
    print(process.q_sample(x_start, t).shape)


if __name__ == "__main__":
    test()
