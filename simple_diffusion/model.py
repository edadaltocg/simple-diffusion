import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConditionalLinear(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, timesteps: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lin = nn.Linear(input_size, hidden_size)
        self.time_embedding = nn.Embedding(timesteps, hidden_size)
        self.time_embedding.weight.data.uniform_()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        out = self.lin(x)
        t_embed = self.time_embedding(t)
        out = t_embed.view(-1, self.hidden_size) * out  # product or sum?
        return out


class ConditionalModel(nn.Module):
    def __init__(self, timesteps: int, input_size: int, hidden_size=128):
        super().__init__()
        self.lin1 = ConditionalLinear(input_size, hidden_size, timesteps)
        self.lin2 = ConditionalLinear(hidden_size, hidden_size, timesteps)
        self.lin3 = nn.Linear(hidden_size, input_size)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)


class EMA:
    """This idea is found in most of the implementations, which allows to implement a form of model momentum.
    Instead of directly updating the weights of the model, we keep a copy of the previous values of the weights,
    and then update a weighted mean between the previous and new version of the weights.
    """

    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
