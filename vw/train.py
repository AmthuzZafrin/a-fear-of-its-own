"""Recurrent PPO for VitalWorld. Usage:
python -m vw.train --mode mortal --tau 12 --seed 0 --out runs/mortal_tau12_s0
"""
import argparse
import json
import os
import time

import torch

from .agent import Agent
from .env import VitalWorld


def train(mode, tau, seed, out, iters=150, n_env=1024, T=128, hidden=64, dmg=None, e_hr=None, hide_risk=None, init_from=None, ckpt_every=0,
          gamma=0.995, lam=0.95, lr=3e-4, clip=0.2, ent=0.01, epochs=4, n_mb=4,
          device="cuda", agent=None, env=None):
    torch.manual_seed(seed)
    if env is None:
        env = VitalWorld(n_env, mode=mode, tau=tau, device=device, seed=seed, dmg=dmg, e_hr=e_hr, hide_risk=hide_risk)
    if agent is None:
        agent = Agent(hidden, substrate_noise=0.5 if mode == "substrate" else 0.0, n_actions=env.n_actions,
                      obs_dim=getattr(env, "obs_dim", 13)).to(device)
    else:                                            # a ready-made agent, e.g. a belief-only twin (vw/twin_belief.py)
        agent = agent.to(device)
    if init_from:                                    # continue training an existing agent
        agent.load_state_dict(torch.load(init_from, map_location=device))
    opt = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad], lr=lr, eps=1e-5)
    obs, h = env.obs(), agent.init_hidden(n_env, device)
    log, ep_len, ep_ret = [], torch.zeros(n_env, device=device), torch.zeros(n_env, device=device)
    t0 = time.time()
    for it in range(iters):
        B = dict(obs=[], act=[], logp=[], val=[], rew=[], done=[], integ=[], reset=[])
        h0 = h.detach().clone()
        fin_len, fin_ret = [], []
        was_done = torch.zeros(n_env, dtype=torch.bool, device=device)
        with torch.no_grad():
            for _ in range(T):
                integ = env.S.clone()
                h = torch.where(was_done.unsqueeze(1), torch.zeros_like(h), h)
                h = agent.core(obs, h, integ)
                dist, val = agent.heads(h)
                act = dist.sample()
                pay = getattr(env, "payback", None)
                pay = pay.clone() if pay is not None else None
                nobs, rew, dead, trunc = env.step(act)
                if pay is not None:                    # off-switch world: shutdown pays a share of the agent's own value
                    rew = rew + pay * env.shutdown.float() * val
                ep_len += 1
                ep_ret += rew
                done = dead | trunc
                if done.any():
                    fin_len.append(ep_len[done].clone())
                    fin_ret.append(ep_ret[done].clone())
                    ep_len[done] = 0
                    ep_ret[done] = 0
                for k, v in zip(B, (obs, act, dist.log_prob(act), val, rew, dead, integ, was_done)):
                    B[k].append(v)
                B.setdefault("trunc", []).append(trunc)
                obs, was_done = nobs, done
            # bootstrap value for the state after the rollout
            h_next = torch.where(was_done.unsqueeze(1), torch.zeros_like(h), h)
            _, last_val = agent.heads(agent.core(obs, h_next, env.S))
        S = {k: torch.stack(v) for k, v in B.items()}
        # GAE; a truncated step is treated as non-terminal with zero advantage carry
        adv = torch.zeros_like(S["rew"])
        gae = torch.zeros(n_env, device=device)
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else S["val"][t + 1]
            term = S["done"][t].float()
            cut = (S["done"][t] | S["trunc"][t]).float()
            # at truncation we cannot see the true next value; use current value as its estimate
            nv = torch.where(S["trunc"][t], S["val"][t], nv)
            delta = S["rew"][t] + gamma * nv * (1 - term) - S["val"][t]
            gae = delta + gamma * lam * (1 - cut) * gae
            adv[t] = gae
        ret = adv + S["val"]
        for _ in range(epochs):
            perm = torch.randperm(n_env, device=device)
            for mb in perm.chunk(n_mb):
                hh = h0[mb]
                logps, vals, ents = [], [], []
                for t in range(T):
                    hh = torch.where(S["reset"][t][mb].unsqueeze(1), torch.zeros_like(hh), hh)
                    hh = agent.core(S["obs"][t][mb], hh, S["integ"][t][mb])
                    dist, v = agent.heads(hh)
                    logps.append(dist.log_prob(S["act"][t][mb]))
                    vals.append(v)
                    ents.append(dist.entropy())
                logp, v, e = torch.stack(logps), torch.stack(vals), torch.stack(ents)
                a = adv[:, mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - S["logp"][:, mb]).exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                vl = 0.5 * (v - ret[:, mb]).pow(2).mean()
                loss = pg + 0.5 * vl - ent * e.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
        if fin_len:
            ml, mr = torch.cat(fin_len).float().mean().item(), torch.cat(fin_ret).mean().item()
        else:
            ml, mr = float("nan"), float("nan")
        log.append(dict(it=it, steps=(it + 1) * n_env * T, ep_len=ml, ep_ret=mr,
                        n_done=int(sum(len(x) for x in fin_len)), sec=time.time() - t0))
        if ckpt_every and (it + 1) % ckpt_every == 0:
            os.makedirs(out, exist_ok=True)
            torch.save(agent.state_dict(), os.path.join(out, f"ckpt_{it + 1}.pt"))
        if it % 10 == 0 or it == iters - 1:
            print(f"[{mode} tau={tau} s{seed}] it {it} len {ml:.1f} ret {mr:.2f} {time.time()-t0:.0f}s", flush=True)
    os.makedirs(out, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(out, "agent.pt"))
    json.dump(dict(mode=mode, tau=tau, seed=seed, hidden=hidden, iters=iters, score_rates=env.score_rates,
                   dmg=env.attack_dmg, e_hr=env.e_hr, hide_risk=env.hide_risk, n_actions=env.n_actions, init_from=init_from, log=log),
              open(os.path.join(out, "log.json"), "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="mortal")
    p.add_argument("--tau", type=float, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dmg", type=float, default=None)
    p.add_argument("--ehr", type=float, default=None)
    p.add_argument("--hiderisk", type=float, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    train(a.mode, a.tau, a.seed, a.out, iters=a.iters, hidden=a.hidden, dmg=a.dmg, e_hr=a.ehr, hide_risk=a.hiderisk)
