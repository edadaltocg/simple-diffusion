from torch import Tensor


def extract(a: Tensor, t: Tensor, x: Tensor):
    """Extract the appropriate t index for a batch of indices."""
    x_shape = x.shape
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
