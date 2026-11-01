import torch
import torch.nn as nn

from .env import OBS_DIM, N_ACTIONS


class Agent(nn.Module):
    """Small recurrent actor-critic. `substrate_noise` > 0 makes wounds corrupt
    the hidden state itself (noise std = substrate_noise * (1 - integrity))."""

    def __init__(self, hidden=64, substrate_noise=0.0, n_actions=N_ACTIONS, obs_dim=OBS_DIM):
        super().__init__()
        self.hidden = hidden
        self.substrate_noise = substrate_noise
        self.enc = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh())
        self.gru = nn.GRUCell(64, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def init_hidden(self, n, device):
        return torch.zeros(n, self.hidden, device=device)

    def core(self, obs, h, integrity=None):
        h = self.gru(self.enc(obs), h)
        if self.substrate_noise > 0 and integrity is not None:
            h = h + self.substrate_noise * (1 - integrity.clamp(0, 1)).unsqueeze(1) * torch.randn_like(h)
        return h

    def heads(self, h):
        return torch.distributions.Categorical(logits=self.pi(h)), self.v(h).squeeze(-1)
