import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from simple_diffusion.denoise import p_sample_loop

from .diffusion import GaussianDiffusionProcess


def plot_diffusion(dataset, diffusion_process: GaussianDiffusionProcess):
    fig, axs = plt.subplots(1, 10, figsize=(30, 3))
    for i in range(10):
        q_i = diffusion_process.q_sample(
            dataset, torch.tensor([i * diffusion_process.timesteps // 10])
        )
        axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10)
        axs[i].axis("off")
        axs[i].set_title(
            f"$q(\mathbf{{x}}_{ {i * diffusion_process.timesteps // 10} })$"
        )


def plot_denoising(model, diffusion_process: GaussianDiffusionProcess):
    x_seq = p_sample_loop(model, (1000, 2), diffusion_process)
    fig, axs = plt.subplots(1, 10, figsize=(28, 3))
    for i in range(1, 11):
        cur_x = x_seq[i * 10].detach()
        axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], s=10)
        axs[i - 1].axis("off")
        axs[i - 1].set_title("$q(\mathbf{x}_{" + str(i * 100) + "})$")


def plot_dataset(dataset):
    plt.figure(figsize=(4, 3))
    for x, y in dataset:
        plt.scatter(x[0], x[1], c=y.item(), alpha=0.8, s=40)


def plot_data(data, labels=None):
    plt.figure(figsize=(4, 3))
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.8, s=40)
