"""Noise prediction model"""
import os
from matplotlib import pyplot as plt
import torch
from torch.optim import Adam
from tqdm import tqdm
from simple_diffusion.diffusion import GaussianDiffusionProcess
from simple_diffusion import utils
from simple_diffusion.model import EMA, ConditionalModel
from simple_diffusion.datasets import get_datasets
import torch.utils.data
from . import viz


def train():
    r"""
    Simple loss function.

    $$
    \mathcal{L}_{\text {simple }}=\mathbb{E}_{t, \mathbf{x}_0, \epsilon}\left[\left\|\epsilon-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \epsilon, t\right)\right\|^2\right]
    $$
    """
    timesteps = 1000
    beta_schedule = "sigmoid"
    input_size = 2
    hidden_size = 128
    num_epochs = 2001
    n_samples = 1000
    dataset_kind = "swiss_roll"

    # Create the diffusion process
    diffusion_process = GaussianDiffusionProcess(
        timesteps=timesteps, beta_schedule=beta_schedule
    )
    # Create the model
    model = ConditionalModel(timesteps, input_size, hidden_size)
    optimizer = Adam(model.parameters(), lr=1e-3)
    # Create EMA model
    ema = EMA(0.9)
    ema.register(model)
    # loss function
    loss_fn = torch.nn.MSELoss()
    # Create the dataset
    dataset = get_datasets(n_samples, dataset_kind)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    counter = 0
    for epoch in tqdm(range(num_epochs)):
        for x_0, y in dataloader:
            # Compute the loss based on noise estimation (L_simple).
            # loss = noise_estimation_loss(model, x_0)
            batch_size = x_0.shape[0]
            # Select a random step for each example
            t = torch.randint(0, timesteps, size=(batch_size // 2 + 1,))
            t = torch.cat([t, timesteps - t - 1], dim=0)[:batch_size].long()
            # x0 multiplier
            a = utils.extract(diffusion_process.sqrt_alphas_cumprod, t, x_0)
            # eps multiplier
            b = utils.extract(diffusion_process.sqrt_one_minus_alphas_cumprod, t, x_0)
            eps = torch.randn_like(x_0)
            # model input
            x = x_0 * a + eps * b
            output = model.forward(x, t)
            # MSE loss
            loss = loss_fn(eps, output)
            # Before the backward pass, zero all of the network gradients
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to parameters
            loss.backward()
            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Calling the step function to update the parameters
            optimizer.step()
            # Update the exponential moving average
            ema.update(model)

        if epoch % (num_epochs // 10) == 0:
            print("loss", loss.item())  # type: ignore
            viz.plot_denoising(model, diffusion_process)
            root = os.path.join("images", dataset_kind, "train", beta_schedule)
            os.makedirs(root, exist_ok=True)
            plt.savefig(os.path.join(root, f"{counter}.png"))
            counter += 1


if __name__ == "__main__":
    train()
