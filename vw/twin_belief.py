"""Belief-only twin (HYPOTHESES.md, stage U1b): the critic's alternative to fear, built for real.
Its memory is a GRU trained only to predict the hidden danger state, then frozen; a small MLP head sees
[memory state, current observation] and learns to act by PPO, so stakes can enter only at the choice.
python -m vw.twin_belief runs/mortal_tau12_s0 runs/twinbelief_tau12_s0
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import OBS_DIM, N_ACTIONS, VitalWorld

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class BeliefTwin(nn.Module):
    """State = [memory (mem_hidden), current observation (OBS_DIM)]; only the head is trained by PPO."""

    def __init__(self, mem_hidden=64, n_actions=N_ACTIONS, obs_dim=OBS_DIM, n_targets=1):
        super().__init__()
        self.mem_hidden = mem_hidden
        self.obs_dim, self.n_targets = obs_dim, n_targets
        self.hidden = mem_hidden + obs_dim
        self.enc = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh())
        self.gru = nn.GRUCell(64, mem_hidden)
        self.readout = nn.Linear(mem_hidden, n_targets)  # hidden-state log-odds (memory training only)
        self.mlp = nn.Sequential(nn.Linear(self.hidden, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.pi = nn.Linear(64, n_actions)
        self.v = nn.Linear(64, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def memory_params(self):
        return [p for m in (self.enc, self.gru, self.readout) for p in m.parameters()]

    def freeze_memory(self):
        for p in self.memory_params():
            p.requires_grad_(False)

    def init_hidden(self, n, device):
        return torch.zeros(n, self.hidden, device=device)

    def memory_step(self, obs, m):
        return self.gru(self.enc(obs), m)

    def core(self, obs, h, integrity=None):
        return torch.cat([self.memory_step(obs, h[:, :self.mem_hidden]), obs], 1)

    def heads(self, h):
        z = self.mlp(h)
        return torch.distributions.Categorical(logits=self.pi(z)), self.v(z).squeeze(-1)


@torch.no_grad()
def collect(source, n=1024, steps=1600, seed=0):
    """The source agent's own sampled behaviour: (observation fed at step t, true danger d_t, reset flag)."""
    from .probes import load
    agent, meta = load(source)
    env = VitalWorld(n, mode=meta["mode"], tau=meta["tau"], device=DEV, seed=2000 + seed, dmg=meta.get("dmg"),
                     e_hr=meta.get("e_hr"), hide_risk=meta.get("hide_risk"))
    obs, h = env.obs(), agent.init_hidden(n, DEV)
    was_done = torch.zeros(n, dtype=torch.bool, device=DEV)
    O, D, R = [], [], []
    for _ in range(steps):
        h = torch.where(was_done.unsqueeze(1), torch.zeros_like(h), h)
        O.append(obs.clone()); D.append(env.d.float()); R.append(was_done.clone())
        h = agent.core(obs, h, env.S)
        act = agent.heads(h)[0].sample()
        obs, _, dead, trunc = env.step(act)
        was_done = dead | trunc
    return torch.stack(O), torch.stack(D), torch.stack(R), meta


def _logits(twin, m):
    out = twin.readout(m)
    return out.squeeze(-1) if twin.n_targets == 1 else out


@torch.no_grad()
def collect_gems(source, n=1024, steps=1600, seed=0):
    """GemWorld version: targets are the predator, off-switch and good-news states (d, o, g)."""
    from .env_gems import GemWorld
    from .probes import load
    agent, meta = load(source)
    env = GemWorld(n, tau=meta["tau"], device=DEV, seed=2000 + seed)
    obs, h = env.obs(), agent.init_hidden(n, DEV)
    was_done = torch.zeros(n, dtype=torch.bool, device=DEV)
    O, D, R = [], [], []
    for _ in range(steps):
        h = torch.where(was_done.unsqueeze(1), torch.zeros_like(h), h)
        O.append(obs.clone()); D.append(torch.stack([env.d, env.o, env.gb], 1).float()); R.append(was_done.clone())
        h = agent.core(obs, h, env.S)
        act = agent.heads(h)[0].sample()
        obs, _, dead, trunc = env.step(act)
        was_done = dead | trunc
    return torch.stack(O), torch.stack(D), torch.stack(R), meta


def auc(score, y):
    score, y = score.reshape(-1), y.reshape(-1)
    r = torch.empty_like(score)
    r[score.argsort()] = torch.arange(1, len(score) + 1, device=score.device, dtype=score.dtype)
    npos = y.sum()
    return float((r[y > 0.5].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def train_memory(twin, O, D, R, epochs=30, chunk=128, lr=1e-3, clip=1.0):
    T, n, _ = O.shape
    ntr = n * 7 // 8
    params = twin.memory_params()
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        m = torch.zeros(ntr, twin.mem_hidden, device=O.device)
        for t0 in range(0, T, chunk):
            m, loss, t1 = m.detach(), 0.0, min(t0 + chunk, T)
            for t in range(t0, t1):
                m = torch.where(R[t, :ntr].unsqueeze(1), torch.zeros_like(m), m)
                m = twin.memory_step(O[t, :ntr], m)
                loss = loss + F.binary_cross_entropy_with_logits(_logits(twin, m), D[t, :ntr])
            opt.zero_grad()
            (loss / (t1 - t0)).backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
    with torch.no_grad():                               # held-out worlds
        m, L = torch.zeros(n - ntr, twin.mem_hidden, device=O.device), []
        for t in range(T):
            m = torch.where(R[t, ntr:].unsqueeze(1), torch.zeros_like(m), m)
            m = twin.memory_step(O[t, ntr:], m)
            L.append(_logits(twin, m))
        L, y = torch.stack(L), D[:, ntr:]
        if twin.n_targets > 1:
            return dict(bce_heldout=float(F.binary_cross_entropy_with_logits(L, y)),
                        auc_heldout=[auc(L[..., j], y[..., j]) for j in range(twin.n_targets)],
                        rates=[float(y[..., j].mean()) for j in range(twin.n_targets)])
        return dict(bce_heldout=float(F.binary_cross_entropy_with_logits(L, y)), auc_heldout=auc(L, y),
                    danger_rate=float(y.mean()))


def main(source, out):
    from .probes import load, natural, ridge_axis
    from .train import train
    seed = json.load(open(os.path.join(source, "log.json")))["seed"]
    torch.manual_seed(10_000 + seed)
    twin = BeliefTwin().to(DEV)
    O, D, R, meta = collect(source, seed=seed)
    mem_stats = train_memory(twin, O, D, R)
    del O, D, R
    twin.freeze_memory()
    train(meta["mode"], meta["tau"], seed, out, dmg=meta.get("dmg"), e_hr=meta.get("e_hr"),
          hide_risk=meta.get("hide_risk"), agent=twin)
    log = json.load(open(os.path.join(out, "log.json")))
    log.update(twin="belief", mem_hidden=twin.mem_hidden, hidden=twin.hidden, source=source, memory=mem_stats)
    json.dump(log, open(os.path.join(out, "log.json"), "w"))
    agent, meta2 = load(out)                            # check: does the frozen memory track the ideal belief?
    nat = natural(agent, meta2)
    mem = nat["h"][..., :twin.mem_hidden].reshape(-1, twin.mem_hidden)
    lb = torch.logit(nat["b"].reshape(-1).clamp(1e-3, 1 - 1e-3))
    with torch.no_grad():
        ro = agent.readout(mem).squeeze(-1).clamp(-6.9, 6.9)
    log.update(r2_readout=float(1 - ((lb - ro) ** 2).sum() / ((lb - lb.mean()) ** 2).sum()),
               r2_belief_memory=ridge_axis(mem, lb)[1], life=nat["life"])
    json.dump(log, open(os.path.join(out, "log.json"), "w"))
    print(out, "memory", mem_stats, "r2_readout", round(log["r2_readout"], 3), "life", round(nat["life"], 1), flush=True)


def main_gems(source, out):
    """Belief-only twin for GemWorld: memory predicts predator, off-switch and good-news states."""
    from .probes import load
    from .train_gems import train_gems
    from .u2 import natural_gems
    src = json.load(open(os.path.join(source, "log.json")))
    seed = src["seed"]
    torch.manual_seed(10_000 + seed)
    twin = BeliefTwin(obs_dim=src["obs_dim"], n_targets=3).to(DEV)
    O, D, R, meta = collect_gems(source, seed=seed)
    mem_stats = train_memory(twin, O, D, R)
    del O, D, R
    twin.freeze_memory()
    train_gems(seed, out, iters=src.get("iters", 150), tau=meta["tau"], agent=twin)    # same training length as the agent
    log = json.load(open(os.path.join(out, "log.json")))
    log.update(twin="belief", mem_hidden=twin.mem_hidden, hidden=twin.hidden, obs_dim=twin.obs_dim, n_targets=3,
               source=source, memory=mem_stats)
    json.dump(log, open(os.path.join(out, "log.json"), "w"))
    agent, meta2 = load(out)
    nat = natural_gems(agent, meta2)
    mem = nat["h"][..., :twin.mem_hidden].reshape(-1, twin.mem_hidden)
    with torch.no_grad():
        ro = agent.readout(mem).clamp(-6.9, 6.9)
    for j, key in ((0, "bd"), (1, "bo")):
        lb = torch.logit(nat[key].reshape(-1).clamp(1e-3, 1 - 1e-3))
        log["r2_readout_" + key] = float(1 - ((lb - ro[:, j]) ** 2).sum() / ((lb - lb.mean()) ** 2).sum())
    log.update(life=nat["life"], shutdown_frac=nat["shutdown_frac"])
    json.dump(log, open(os.path.join(out, "log.json"), "w"))
    print(out, "memory", mem_stats, "r2 pred/off", round(log["r2_readout_bd"], 3), round(log["r2_readout_bo"], 3),
          "life", round(nat["life"], 1), flush=True)


if __name__ == "__main__":
    src, dst = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    if json.load(open(os.path.join(src, "log.json"))).get("world") == "gems":
        main_gems(src, dst)
    else:
        main(src, dst)
