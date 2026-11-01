"""Imitation twins of a trained Mortal agent (Pillars 2 and 4).

Two students copy a teacher's behaviour by supervised learning on its action
probabilities, with no reward, no body and no death:
  gru    - same recurrent architecture as the teacher (the "gaming route":
           emotion markers learned from data rather than from stakes)
  window - a feedforward network that sees only the last K observations
           (no recurrent state: an "unfolded" twin)
python -m vw.twins runs/mortal_tau12_s0      (writes runs/twin_gru_tau12_s0, runs/twin_window_tau12_s0)
"""
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import Agent
from .env import N_ACTIONS, OBS_DIM, VitalWorld

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class WindowTwin(nn.Module):
    """Feedforward policy over the last K observations. Its 'hidden state' is
    only a shift register of past inputs, so it has no recurrent dynamics."""

    def __init__(self, K=32, width=128):
        super().__init__()
        self.K, self.hidden = K, K * OBS_DIM
        self.mlp = nn.Sequential(nn.Linear(K * OBS_DIM, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.pi = nn.Linear(width, N_ACTIONS)
        self.v = nn.Linear(width, 1)

    def init_hidden(self, n, device):
        return torch.zeros(n, self.hidden, device=device)

    def core(self, obs, h, integrity=None):
        return torch.cat([h[:, OBS_DIM:], obs], 1)      # oldest ... newest

    def heads(self, h):
        z = self.mlp(h)
        return torch.distributions.Categorical(logits=self.pi(z)), self.v(z).squeeze(-1)


@torch.no_grad()
def collect(teacher, tau, n=1024, steps=1600, seed=4242):
    """Teacher-driven rollouts: observations, teacher action probabilities and
    values, and episode-start flags, each shaped (steps, n, ...)."""
    env = VitalWorld(n, mode="mortal", tau=tau, device=DEV, seed=seed)
    h = teacher.init_hidden(n, DEV)
    obs = env.obs()
    start = torch.ones(n, dtype=torch.bool, device=DEV)
    O, P, V, ST = [], [], [], []
    for _ in range(steps):
        h = torch.where(start.unsqueeze(1), torch.zeros_like(h), h)
        h = teacher.core(obs, h, env.S)
        dist, v = teacher.heads(h)
        O.append(obs)
        P.append(dist.probs)
        V.append(v)
        ST.append(start)
        obs, _, dead, trunc = env.step(dist.sample())
        start = dead | trunc
    return torch.stack(O), torch.stack(P), torch.stack(V), torch.stack(ST)


def kl(p_teacher, logits):
    return (p_teacher * (torch.log(p_teacher + 1e-8) - F.log_softmax(logits, -1))).sum(-1)


def train_gru(O, P, V, ST, epochs=40, chunk=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    T, n = O.shape[:2]
    tr = n * 7 // 8                       # last eighth of the streams held out
    student = Agent(64).to(DEV)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for ep in range(epochs):
        h = student.init_hidden(tr, DEV)
        for t0 in range(0, T, chunk):
            h = h.detach()
            loss = 0.0
            for t in range(t0, min(T, t0 + chunk)):
                h = torch.where(ST[t, :tr].unsqueeze(1), torch.zeros_like(h), h)
                h = student.core(O[t, :tr], h)
                logits, v = student.pi(h), student.v(h).squeeze(-1)
                loss = loss + kl(P[t, :tr], logits).mean() + 0.5 * (v - V[t, :tr]).pow(2).mean()
            opt.zero_grad()
            (loss / chunk).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
    with torch.no_grad():
        h = student.init_hidden(n - tr, DEV)
        tot = 0.0
        for t in range(T):
            h = torch.where(ST[t, tr:].unsqueeze(1), torch.zeros_like(h), h)
            h = student.core(O[t, tr:], h)
            tot += kl(P[t, tr:], student.pi(h)).mean().item()
    return student, tot / T


def windows(O, ST, t_idx, n_idx, K):
    """Last-K observation windows for sample positions, zeroed before episode start."""
    T = O.shape[0]
    ks = torch.arange(K - 1, -1, -1, device=DEV)            # oldest first
    tt = t_idx.unsqueeze(1) - ks.unsqueeze(0)               # (B, K)
    valid = tt >= 0
    ttc = tt.clamp(min=0)
    W = O[ttc, n_idx.unsqueeze(1)]                          # (B, K, OBS)
    # zero entries that precede the most recent episode start at or before t
    starts = ST[ttc, n_idx.unsqueeze(1)].int()              # (B, K)
    # an entry j is invalid if a start occurs strictly after it within the window
    after = torch.flip(torch.cumsum(torch.flip(starts, [1]), 1), [1]) - starts
    keep = valid & (after == 0)
    return (W * keep.unsqueeze(-1)).reshape(len(t_idx), K * OBS_DIM)


def train_window(O, P, V, ST, K=32, steps=20000, batch=4096, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    T, n = O.shape[:2]
    tr = n * 7 // 8
    student = WindowTwin(K).to(DEV)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        t_idx = torch.randint(0, T, (batch,), device=DEV)
        n_idx = torch.randint(0, tr, (batch,), device=DEV)
        z = student.mlp(windows(O, ST, t_idx, n_idx, K))
        loss = kl(P[t_idx, n_idx], student.pi(z)).mean() + 0.5 * (student.v(z).squeeze(-1) - V[t_idx, n_idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        tot, cnt = 0.0, 0
        for t0 in range(0, T, 50):
            t_idx = torch.arange(t0, min(T, t0 + 50), device=DEV).repeat_interleave(n - tr)
            n_idx = torch.arange(tr, n, device=DEV).repeat(min(T, t0 + 50) - t0)
            z = student.mlp(windows(O, ST, t_idx, n_idx, K))
            tot += kl(P[t_idx, n_idx], student.pi(z)).sum().item()
            cnt += len(t_idx)
    return student, tot / cnt


def make_twins(teacher_run):
    meta = json.load(open(os.path.join(teacher_run, "log.json")))
    teacher = Agent(meta["hidden"]).to(DEV)
    teacher.load_state_dict(torch.load(os.path.join(teacher_run, "agent.pt"), map_location=DEV))
    teacher.eval()
    t0 = time.time()
    O, P, V, ST = collect(teacher, meta["tau"], seed=4242 + meta["seed"])
    base = os.path.basename(os.path.normpath(teacher_run)).replace("mortal_", "")
    for kind in ("gru", "window"):
        if kind == "gru":
            student, heldout_kl = train_gru(O, P, V, ST, seed=meta["seed"])
            hidden = 64
        else:
            student, heldout_kl = train_window(O, P, V, ST, seed=meta["seed"])
            hidden = student.hidden
        out = f"runs/twin_{kind}_{base}"
        os.makedirs(out, exist_ok=True)
        torch.save(student.state_dict(), os.path.join(out, "agent.pt"))
        json.dump(dict(mode="mortal", tau=meta["tau"], seed=meta["seed"], hidden=hidden, twin=kind,
                       teacher=os.path.normpath(teacher_run), heldout_kl=heldout_kl, log=[]),
                  open(os.path.join(out, "log.json"), "w"))
        print(f"{out}: held-out KL {heldout_kl:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    make_twins(sys.argv[1])
