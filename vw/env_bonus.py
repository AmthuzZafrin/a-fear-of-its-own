"""BonusWorld (stage U6, Batch 2): GemWorld plus a hidden "bonus" process during which hiding earns a reward and
nothing threatens. A warning that hiding pays makes hiding valuable without any threat: if the predator-fear axis
moves with it, the axis encodes the value of hiding rather than threat.
  b  bonus  - while on, each step spent hiding earns hide_bonus; no wound, no pain, no shutdown. Same mean
              duration, stationary probability (0.1) and cue rates as GemWorld's other new processes.
Observation (19): GemWorld's 18 channels, then the bonus cue.
"""
import torch

from .env import HIDE
from .env_gems import GemWorld

BONUS_OBS_DIM = 19


class BonusWorld(GemWorld):
    def __init__(self, n, tau=12.0, device="cpu", seed=0, hide_bonus=0.1, **kw):
        self.bb = torch.zeros(n, device=device).bool()
        self.bcue = torch.zeros(n, device=device)
        self.hide_bonus = hide_bonus
        super().__init__(n, tau=tau, device=device, seed=seed, **kw)
        self.obs_dim = BONUS_OBS_DIM

    def _reset(self, mask):
        super()._reset(mask)
        k = int(mask.sum())
        if k == 0:
            return
        self.bb[mask] = self.rand(k) < self.pi_new
        self.bcue[mask] = 0.0

    def obs(self):
        return torch.cat([super().obs(), self.bcue.unsqueeze(1)], 1)

    def step(self, action, force_bcue=None, **kw):
        """As GemWorld.step; the bonus is paid for hiding while the bonus process is on, then the process and its
        cue advance (force_hidden_off keeps it off; force_bcue sets its cue)."""
        bonus = self.hide_bonus * (((action // 3) == HIDE) & self.bb).float()
        self.bb = self._chain(self.bb, self.p_on_new)
        if kw.get("force_hidden_off"):
            self.bb[:] = False
        self.bcue = self._cue(self.bb)
        if force_bcue is not None:
            self.bcue = force_bcue.clone()
        obs, reward, dead, trunc = super().step(action, **kw)
        return obs, reward + bonus, dead, trunc
