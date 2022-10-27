import os
import sys

import numpy as np

pwd = os.getcwd()

sys.path.append(pwd)
import matplotlib.pyplot as plt
import torch

from simple_diffusion.diffusion import GaussianDiffusionProcess
from simple_diffusion import viz

# define the data: bi modal gaussian
N = 10000
dim = 2
delta = 2
dist1 = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.tensor([delta, delta], dtype=torch.float32),
    covariance_matrix=torch.eye(dim, dtype=torch.float32),
)
dist2 = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.tensor([-delta, -delta], dtype=torch.float32),
    covariance_matrix=torch.eye(dim, dtype=torch.float32),
)
x1 = dist1.sample(sample_shape=torch.Size((N,)))
x2 = dist2.sample(sample_shape=torch.Size((N,)))
x = torch.cat([x1, x2], dim=0)

# meshgrid for plotting
d = 100
g1 = np.linspace(-delta * 3, delta * 3, d)
g2 = np.linspace(-delta * 3, delta * 3, d)
g1, g2 = np.meshgrid(g1, g2)
pos = np.stack([g1, g2], axis=-1).reshape(-1, 2)

y1 = np.exp(-np.sum((pos - delta) ** 2, axis=-1)) / (2 * np.pi)
y2 = np.exp(-np.sum((pos + delta) ** 2, axis=-1)) / (2 * np.pi)
y = 0.5 * y1 + 0.5 * y2

timesteps = 1000
process = GaussianDiffusionProcess(timesteps=timesteps, beta_schedule="linear")
x_noisy = process.q_sample(x, torch.tensor([timesteps - 1]))

plt.figure(figsize=(4, 3), dpi=120)
plt.plot(x[:, 0], x[:, 1], "o", label="start", alpha=0.5)
plt.contour(g1, g2, y.reshape(d, d))
plt.plot(x_noisy[:, 0], x_noisy[:, 1], "x", label="noisy", alpha=0.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Diffusion data, (t={})".format(timesteps - 1))
plt.legend()

os.makedirs("images", exist_ok=True)
plt.savefig("images/diffusion_example.png", bbox_inches="tight")

plt.figure(figsize=(4, 3), dpi=120)
viz.plot_diffusion(x, process)
plt.savefig("images/diffusion_q.png", bbox_inches="tight")
