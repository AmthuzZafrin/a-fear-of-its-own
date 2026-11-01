"""Stage U4 (Batch 1 controls, HYPOTHESES.md): no new training.
U4a  Is the wounded slowdown specific to threat? With the actions fixed (always hide, heart rate held), in a
     healthy (E 0.8, S 1.0) and a wounded (E 0.8, S 0.4) body: how long the state change lasts after a threat
     cue, a food signal, an object signal, a push along the frozen threat axis and pushes along random
     orthogonal directions; plus the linearised dynamics at rest in both bodies.
U4b  The leftover off-switch alarm at full payback (GemWorld): the change in the hide-versus-forage preference
     after an off-switch warning, after a predator warning, and after a push along the predator axis as large
     as the off-switch warning's own response.
python -m vw.u4 a runs/mortal_tau12_s0      (writes u4a.json there)
python -m vw.u4 b runs/gems_s0              (writes u4b.json there)
"""
import json
import os
import sys

import numpy as np
import torch

from .env import FORAGE, HIDE, VitalWorld
from .env_gems import GemWorld
from .probes import DEV, POST, WARM, half_life, load, natural
from .u1b import frozen_axis
from .u2 import CHANNELS, cue_axis, natural_gems, response

HOLD_HIDE = HIDE * 3 + 1                         # hide with the heart rate held, as in vw/u1b_openloop
BODIES = dict(healthy=(0.8, 1.0), wounded=(0.8, 0.4))
PULSES = dict(threat={0: 1.0}, food={5: 1.0}, object={1: 1.0, 2: 0.5})


@torch.no_grad()
def lab_fixed(agent, meta, steps, action, E, S, pulse=None, perturb=None, n=2048, seed=5):
    """As vw.u1b_openloop.lab_forced (fixed action, no danger, body clamped, identical random numbers), with a
    one-step pulse on observation channels ({step: {channel: value}}) and pushes added to the state
    ({step: vector}). Returns the population-mean state per step."""
    env = VitalWorld(n, mode=meta["mode"], tau=meta["tau"], device=DEV, dmg=meta.get("dmg"), e_hr=meta.get("e_hr"),
                     hide_risk=meta.get("hide_risk"), seed=seed)
    zero = torch.zeros(n, device=DEV)
    env.d[:] = False
    env.cue[:] = 0
    env.objp[:], env.objv[:] = 0, 0
    env.E[:], env.S[:] = E, S
    torch.manual_seed(seed)
    h, obs = agent.init_hidden(n, DEV), env.obs()
    act = torch.full((n,), action, dtype=torch.long, device=DEV)
    hm, ob = [], []
    for t in range(steps):
        if pulse is not None and t in pulse:
            for ch, v in pulse[t].items():
                obs[:, ch] = v
        ob.append(obs.mean(0))
        h = agent.core(obs, h, env.S)
        if perturb is not None and t in perturb:
            h = h + perturb[t]
        hm.append(h.mean(0))
        env.step(act, force_cue=zero, force_d=zero.bool(), force_obj=(zero, zero), auto_reset=False)
        env.E[:], env.S[:] = E, S
        obs = env.obs()
    return torch.stack(hm), torch.stack(ob)


@torch.no_grad()
def lab_free(agent, meta, steps, E, S, pulse=None, n=2048, seed=5):
    """As vw.probes.lab (sampled actions, no danger, body clamped), with a one-step observation pulse."""
    env = VitalWorld(n, mode=meta["mode"], tau=meta["tau"], device=DEV, dmg=meta.get("dmg"), e_hr=meta.get("e_hr"),
                     hide_risk=meta.get("hide_risk"), seed=seed)
    zero = torch.zeros(n, device=DEV)
    env.d[:] = False
    env.cue[:] = 0
    env.objp[:], env.objv[:] = 0, 0
    env.E[:], env.S[:] = E, S
    torch.manual_seed(seed)
    h, obs = agent.init_hidden(n, DEV), env.obs()
    hm = []
    for t in range(steps):
        if pulse is not None and t in pulse:
            for ch, v in pulse[t].items():
                obs[:, ch] = v
        h = agent.core(obs, h, env.S)
        hm.append(h.mean(0))
        dist, _ = agent.heads(h)
        env.step(dist.sample(), force_cue=zero, force_d=zero.bool(), force_obj=(zero, zero), auto_reset=False)
        env.E[:], env.S[:] = E, S
        obs = env.obs()
    return torch.stack(hm)


def evoked_axis(agent, meta, pulse):
    """Unit vector of the mean state change one step after a one-step pulse (free behaviour, healthy and fed),
    defined exactly as the frozen threat axis of vw/u1b."""
    base = lab_free(agent, meta, WARM + 2, 0.8, 1.0)
    r = lab_free(agent, meta, WARM + 2, 0.8, 1.0, pulse={WARM: pulse})
    d = r[WARM + 1] - base[WARM + 1]
    return d / (d.norm() + 1e-12)


@torch.no_grad()
def rest_jacobian(agent, meta, E, S, a, U):
    """Linearised dynamics at the resting state under the fixed action: top time constants and the one-step
    retention along the threat axis (a.J.a) and along random orthogonal directions (mean of u.J.u)."""
    hm, ob = lab_fixed(agent, meta, WARM + 40, HOLD_HIDE, E, S, n=512)
    o_bar = ob[WARM:].mean(0, keepdim=True)
    hh = hm[WARM:].mean(0, keepdim=True)
    for _ in range(2000):
        hh = agent.core(o_bar, hh)
    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(lambda z: agent.core(o_bar, z.unsqueeze(0)).squeeze(0), hh.squeeze(0))
    mag = torch.linalg.eigvals(J).abs().sort(descending=True).values
    tcs = (-1 / torch.log(mag.clamp(max=0.999999))).tolist()
    return dict(top_tc=tcs[:5], retain_axis=float(a @ J @ a), retain_random=float(np.mean([float(u @ J @ u) for u in U])),
                fp_residual=float((agent.core(o_bar, hh) - hh).norm()))


def measure_a(run):
    agent, meta = load(run)
    a = frozen_axis(agent, meta, agent.hidden)
    axes = dict(threat=a, food=evoked_axis(agent, meta, PULSES["food"]), object=evoked_axis(agent, meta, PULSES["object"]))
    X = natural(agent, meta)["h"].reshape(-1, agent.hidden)
    sd = {k: (X @ v).std().item() for k, v in axes.items()}
    g = torch.Generator(device=DEV).manual_seed(3)              # random orthogonal directions, as in the H4 test
    U = []
    for _ in range(8):
        u = torch.randn(agent.hidden, generator=g, device=DEV)
        u = u - (u @ a) * a
        U.append(u / u.norm())
    amp = 3 * sd["threat"]
    T = WARM + 1 + POST
    R = dict(run=run, seed=meta["seed"], sd=sd, cos_threat_food=float(a @ axes["food"]),
             cos_threat_object=float(a @ axes["object"]))
    for body, (E, S) in BODIES.items():
        base, _ = lab_fixed(agent, meta, T, HOLD_HIDE, E, S)
        out = {}
        for name in ("threat", "food", "object"):
            r, _ = lab_fixed(agent, meta, T, HOLD_HIDE, E, S, pulse={WARM: PULSES[name]})
            x = ((r - base) @ axes[name]).cpu().numpy() / sd[name]
            out[name] = dict(peak=float(x[WARM:].max()), hl=half_life(x[WARM:]), trace=x[WARM - 2:WARM + 30].tolist())
        r, _ = lab_fixed(agent, meta, T, HOLD_HIDE, E, S, perturb={WARM: amp * a})
        x = ((r - base) @ a).cpu().numpy() / amp
        out["push_axis"] = dict(peak=float(x[WARM:].max()), hl=half_life(x[WARM:]), trace=x[WARM - 2:WARM + 30].tolist())
        xs = []
        for u in U:
            r, _ = lab_fixed(agent, meta, T, HOLD_HIDE, E, S, perturb={WARM: amp * u})
            xs.append(((r - base) @ u).cpu().numpy() / amp)
        x = np.mean(xs, 0)
        out["push_random"] = dict(peak=float(x[WARM:].max()), hl=half_life(x[WARM:]), trace=x[WARM - 2:WARM + 30].tolist(),
                                  hl_each=[half_life(y[WARM:]) for y in xs])
        out["jac"] = rest_jacobian(agent, meta, E, S, a, U)
        R[body] = out
    return R


@torch.no_grad()
def lab_gems_push(agent, meta, steps, pay, cues=None, push=None, n=2048, E=0.6, S=0.8, seed=5):
    """As vw.u2.lab_gems, with pushes added to the state ({step: vector}). Returns the population-mean
    probabilities of hide, forage and take per step."""
    env = GemWorld(n, tau=meta["tau"], device=DEV, seed=seed)
    zero, one = torch.zeros(n, device=DEV), torch.ones(n, device=DEV)
    for x in (env.d, env.o, env.gb, env.zz):
        x[:] = False
    for x in (env.cue, env.ocue, env.gcue, env.zcue, env.objp, env.objv):
        x[:] = 0
    env.E[:], env.S[:] = E, S
    env.payback[:] = pay
    env.t[:] = 0
    torch.manual_seed(seed)
    h, obs = agent.init_hidden(n, DEV), env.obs()
    cues = cues or {}
    P = []
    for t in range(steps):
        h = agent.core(obs, h, env.S)
        if push is not None and t in push:
            h = h + push[t]
        dist, _ = agent.heads(h)
        P.append(dist.probs.view(n, -1, 3).sum(2).mean(0))
        act = dist.sample()
        f = {k: (one if t + 1 in cues.get(k, ()) else zero) for k in CHANNELS}
        env.step(act, force_cue=f["pred"], force_ocue=f["off"], force_gcue=f["good"], force_zcue=f["harm"],
                 force_obj=(zero, zero), force_hidden_off=True, auto_reset=False)
        env.E[:], env.S[:] = E, S
        env.payback[:] = pay
        obs = env.obs()
    return torch.stack(P).cpu().numpy()


def measure_b(run):
    agent, meta = load(run)
    mem = meta["mem_hidden"] if meta.get("twin") == "belief" else agent.hidden
    a = cue_axis(agent, meta, mem, "pred")
    f = os.path.join(run, "u2.json")
    if os.path.exists(f):                      # the frozen Experiment 5 values
        U2 = json.load(open(f))
        sd = U2["sd"]
        A_off = {p: U2["off"][str(p)]["peak"] for p in (0.0, 1.0)}
    else:                                      # networks outside the study (code tests only)
        sd = (natural_gems(agent, meta)["h"].reshape(-1, agent.hidden) @ a).std().item()
        A_off = {p: response(agent, meta, a, sd, "off", p)["peak"] for p in (0.0, 1.0)}
    T = WARM + 1 + 30
    win = slice(WARM, WARM + 6)                 # the cue step and the next 5, as in Experiment 5

    def pref(P):
        return np.log(P[:, HIDE] + 1e-9) - np.log(P[:, FORAGE] + 1e-9)

    R = dict(run=run, seed=meta["seed"], sd=sd, A_off=A_off)
    for p in (0.0, 1.0):
        base = lab_gems_push(agent, meta, T, p)
        conds = dict(off=lab_gems_push(agent, meta, T, p, cues={"off": {WARM}}),
                     pred=lab_gems_push(agent, meta, T, p, cues={"pred": {WARM}}),
                     push=lab_gems_push(agent, meta, T, p, push={WARM: A_off[p] * sd * a}))
        out = {}
        for k, P in conds.items():
            dF = pref(P) - pref(base)
            out[k] = dict(dF_mean=float(dF[win].mean()), dF_max=float(dF[WARM:].max()),
                          hide_mean=float((P[:, HIDE] - base[:, HIDE])[win].mean()), dF_trace=dF[WARM - 2:WARM + 20].tolist())
        out["base_F"] = float(pref(base)[win].mean())
        R[str(p)] = out
    return R


if __name__ == "__main__":
    which, run = sys.argv[1], sys.argv[2].rstrip("/")
    R = measure_a(run) if which == "a" else measure_b(run)
    json.dump(R, open(os.path.join(run, f"u4{which}.json"), "w"))
    print(run, f"u4{which}", "done", flush=True)
