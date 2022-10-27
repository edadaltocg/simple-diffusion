r"""
Likelihood computation.

$$
    \begin{aligned}
        \mathcal{L} & = \mathbb{E}\left[\mathcal{L}_T + \sum_{t>1}\mathcal{L}_{t-1} + \mathcal{L}_0 \right] \\
        \mathcal{L}_T &=D_{K L}\left(q\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| p\left(\mathbf{x}_T\right)\right) \\
        \mathcal{L}_{t-1} &=D_{K L}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right) \\
        \mathcal{L}_0 &=-\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)
    \end{aligned}
$$
    """
import numpy as np
import torch

from simple_diffusion.diffusion import GaussianDiffusionProcess
from torch import Tensor


def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
    return kl


def approx_standard_normal_cdf(x):
    return 0.5 * (
        1.0
        + torch.tanh(
            torch.tensor(torch.sqrt(torch.tensor([2.0]) / torch.pi))
            * (x + 0.044715 * torch.pow(x, 3))
        )
    )


def discretized_gaussian_log_likelihood(x, means, log_scales):
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1 - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(torch.clamp(cdf_delta, min=1e-12)),
        ),
    )
    return log_probs


def p_mean_variance(
    model, diffusion_process: GaussianDiffusionProcess, x: Tensor, t: Tensor
):
    """
    Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
    the initial x, x_0.

    :param model: the model, which takes a signal and a batch of timesteps
                    as input.
    :param x: the [N x C x ...] tensor at time t.
    :param t: a 1-D Tensor of timesteps.
    :return: a dict with the following keys:
                - 'mean': the model mean output.
                - 'variance': the model variance output.
                - 'log_variance': the log of 'variance'.
                - 'pred_xstart': the prediction for x_0.
    """
    B, C = x.shape[:2]
    assert t.shape == (B,)
    model_output = model(x, t)
    pred_xstart = diffusion_process.predict_xstart_from_eps(
        x_t=x, t=t, eps=model_output
    )
    # clip the denoised image to be in the range [-1, 1]
    pred_xstart = torch.clamp(pred_xstart, -1.0, 1.0)
    (
        model_mean,
        model_variance,
        model_log_variance,
    ) = diffusion_process.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

    assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
    return {
        "mean": model_mean,
        "variance": model_variance,
        "log_variance": model_log_variance,
        "pred_xstart": pred_xstart,
    }


def prior_bpd(x_start: Tensor, diffusion_process: GaussianDiffusionProcess):
    """
    Get the prior KL term for the variational lower-bound, measured in
    bits-per-dim.

    This term can't be optimized, as it only depends on the encoder.

    :param x_start: the [N x C x ...] tensor of inputs.
    :return: a batch of [N] KL values (in bits), one per batch element.
    """
    batch_size = x_start.shape[0]
    t = torch.tensor(
        [diffusion_process.timesteps - 1] * batch_size, device=x_start.device
    )
    qt_mean, _, qt_log_variance = diffusion_process.q_mean_variance(x_start, t)
    kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
    return torch.mean(torch.flatten(kl_prior, 1), -1) / np.log(2.0)


def vb_terms_bpd(model, diffusion_process: GaussianDiffusionProcess, x_start, x_t, t):
    """
    Get a term for the variational lower-bound.

    The resulting units are bits (rather than nats, as one might expect).
    This allows for comparison to other papers.

    :return: a dict with the following keys:
             - 'output': a shape [N] tensor of NLLs or KLs.
             - 'pred_xstart': the x_0 predictions.
    """
    (
        true_mean,
        _,
        true_log_variance_clipped,
    ) = diffusion_process.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
    out = p_mean_variance(model, diffusion_process, x_t, t)
    kl = normal_kl(
        true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
    )
    kl = torch.mean(torch.flatten(kl, 1), -1) / np.log(2.0)

    decoder_nll = -discretized_gaussian_log_likelihood(
        x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
    )
    assert decoder_nll.shape == x_start.shape
    # preserve batch size
    decoder_nll = torch.mean(torch.flatten(decoder_nll, 1), -1) / np.log(2.0)

    # At the first timestep return the decoder NLL,
    # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    output = torch.where((t == 0), decoder_nll, kl)
    return {"output": output, "pred_xstart": out["pred_xstart"]}


def calc_vlb_bpd(model, diffusion_process: GaussianDiffusionProcess, x_start: Tensor):
    """
    Compute the entire variational lower-bound, measured in bits-per-dim.

    :param model: the model to evaluate loss on.
    :param x_start: the [N x C x ...] tensor of inputs.

    :return:
             - total_bpd: the total variational lower-bound, per batch element.
    """
    device = x_start.device
    batch_size = x_start.shape[0]

    vb = []
    for t in list(range(diffusion_process.timesteps))[::-1]:
        t_batch = torch.tensor([t] * batch_size, device=device)
        noise = torch.randn_like(x_start)
        x_t = diffusion_process.q_sample(x_start=x_start, t=t_batch, noise=noise)
        # Calculate VLB term at the current timestep
        with torch.no_grad():
            out = vb_terms_bpd(
                model,
                diffusion_process,
                x_start=x_start,
                x_t=x_t,
                t=t_batch,
            )
        vb.append(out["output"])

    vb = torch.stack(vb, dim=1)
    prior_bpd_ = prior_bpd(x_start, diffusion_process)
    total_bpd = vb.sum(dim=1) + prior_bpd_
    return total_bpd
