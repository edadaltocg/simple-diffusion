import os
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
    make_s_curve,
    make_swiss_roll,
)

from simple_diffusion import viz
from .diffusion import GaussianDiffusionProcess


def sample_batch(n_samples, kind="swiss_roll"):
    x, y = None, None
    if kind == "swiss_roll":
        x, y = make_swiss_roll(n_samples, noise=0.3)
        x = x[:, [0, 2]]  # type: ignore
    elif kind == "s_curve":
        x, y = make_s_curve(n_samples, noise=0.2)
        x = x[:, [0, 2]]  # type: ignore
    elif kind == "moons":
        x, y = make_moons(n_samples, noise=0.05)
    elif kind == "circles":
        x, y = make_circles(n_samples, noise=0.2)
    elif kind == "blobs":
        x, y = make_blobs(n_samples, centers=2, cluster_std=0.5, return_centers=False)  # type: ignore
    else:
        raise ValueError(f"Unknown kind {kind}")

    # normalize between -1 and 1
    x = (x - x.min()) / (x.max() - x.min())  # type: ignore
    return x, y


def get_datasets(n_samples, kind="swiss_roll"):
    x, y = sample_batch(n_samples, kind)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return dataset


def test():
    n = 1000
    for kind in ["swiss_roll", "s_curve", "moons", "circles", "blobs"]:
        dataset = get_datasets(n, kind=kind)
        data = dataset.tensors[0]

        viz.plot_data(data)
        os.makedirs("images/datasets", exist_ok=True)
        plt.savefig(f"images/datasets/{kind}.png")
        plt.close()

        for beta_schedule in ["linear", "cosine", "sigmoid", "quadratic"]:
            diffusion_process = GaussianDiffusionProcess(
                timesteps=1000, beta_schedule=beta_schedule
            )
            viz.plot_diffusion(data, diffusion_process)
            plt.tight_layout()
            os.makedirs(f"images/{kind}", exist_ok=True)
            plt.savefig(f"images/{kind}/diffusion_{beta_schedule}.png")
            plt.close()


if __name__ == "__main__":
    test()
