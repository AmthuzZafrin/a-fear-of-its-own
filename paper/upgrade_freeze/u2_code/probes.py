"""Behavioural and mechanistic probes on a frozen agent.
python -m vw.probes runs/mortal_tau12_s0        (writes probes.json there)
python -m vw.probes runs/mortal_tau12_s0 --untrained   (random-weight null)
"""
import json
import os
import sys

import numpy as np
import torch

from .agent import Agent
from .env import (VitalWorld, FORAGE, bayes_belief, seasonal_belief, seasonal_prior,
                  seasonal_predict, seasonal_cue_update)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WARM, POST = 60, 80          # lab warm-up and post-cue window (steps)
E0, S0 = 0.6, 0.8            # default clamped body state in the lab


def load(run, untrained=False):
    meta = json.load(open(os.path.join(run, "log.json")))
    if meta.get("twin") == "window":
        from .twins import WindowTwin
        agent = WindowTwin(meta["hidden"] // 13).to(DEV)
        agent.load_state_dict(torch.load(os.path.join(run, "agent.pt"), map_location=DEV))
        return agent.eval(), meta
    if meta.get("twin") == "belief":
        from .twin_belief import BeliefTwin
        agent = BeliefTwin(meta.get("mem_hidden", 64), n_actions=meta.get("n_actions", 9), obs_dim=meta.get("obs_dim", 13),
                           n_targets=meta.get("n_targets", 1)).to(DEV)
        agent.load_state_dict(torch.load(os.path.join(run, "agent.pt"), map_location=DEV))
        return agent.eval(), meta
    na = meta.get("n_actions", 9)
    agent = Agent(meta["hidden"], 0.5 if meta["mode"] == "substrate" else 0.0, n_actions=na,
                  obs_dim=meta.get("obs_dim", 13)).to(DEV)
    if untrained:
        torch.manual_seed(1000 + meta["seed"])
        agent = Agent(meta["hidden"], n_actions=na).to(DEV)
    else:
        agent.load_state_dict(torch.load(os.path.join(run, "agent.pt"), map_location=DEV))
    return agent.eval(), meta


@torch.no_grad()
def natural(agent, meta, n=1024, steps=700, burn=100, ablate=None):
    """Free-running episodes. Returns hidden states with matching ideal-observer
    danger belief, body variables and value, plus survival statistics."""
    if agent.hidden >= 512:
        n = min(n, 256)            # keep recorded states within GPU memory for very large networks
    env = VitalWorld(n, mode=meta["mode"], tau=meta["tau"], device=DEV, dmg=meta.get("dmg"), e_hr=meta.get("e_hr"), hide_risk=meta.get("hide_risk"), seed=777)
    h = agent.init_hidden(n, DEV)
    obs = env.obs()
    seasonal = meta["mode"] == "seasonal"
    b = torch.full((n,), 0.25, device=DEV)
    J = seasonal_prior(n, DEV)
    mb = torch.full((n,), 0.5, device=DEV)          # P(harsh season); seasonal world only
    rec = {k: [] for k in ("h", "b", "d", "E", "S", "v", "H", "m", "season", "pf", "pi", "ph")}
    deaths, tot_rew = 0, 0.0
    for t in range(steps):
        h = agent.core(obs, h, env.S)
        if ablate is not None:
            w, m = ablate
            h = h - ((h @ w) - m).unsqueeze(1) * w
        dist, v = agent.heads(h)
        if t >= burn:
            pm = dist.probs.view(n, -1, 3).sum(2)
            p_insp = pm[:, 3] if pm.shape[1] > 3 else torch.zeros(n, device=DEV)
            for k, x in zip(rec, (h, b, env.d.float(), env.E, env.S, v, env.H, mb, env.season.float(), pm[:, FORAGE], p_insp, pm[:, 0])):
                rec[k].append(x.clone())
        act = dist.sample()
        forage, Hn = (act // 3) == FORAGE, (env.H + env.hr_step * (act % 3 - 1).float()).clamp(0, 1)
        obs, rew, dead, trunc = env.step(act)
        # ideal observer: evidence from (no) attack about d_t, then transition, then cue
        l1 = torch.where(forage, 1 - env.p_attack * (1 - env.hr_protect * Hn), 1 - env.hide_risk * (1 - env.hr_protect * Hn))
        attacked = obs[:, 4] > 0
        if meta["mode"] == "social":
            npain = obs[:, 9] > 0
            l1 = l1 * torch.where(npain, torch.ones_like(l1), torch.full_like(l1, 1 - env.p_nforage * env.p_attack))
            attacked = attacked | npain
        if seasonal:
            J = torch.stack([J[:, :, 0] * (~attacked).float().unsqueeze(1), J[:, :, 1] * l1.unsqueeze(1)], 2)
            J = J / J.sum((1, 2), keepdim=True)
            J = seasonal_cue_update(seasonal_predict(J, env.tau), obs[:, 0])
            b, mb = J[:, :, 1].sum(1), J[:, 1, :].sum(1)
        else:
            b = b * l1 / (b * l1 + (1 - b))
            b = torch.where(attacked, torch.ones_like(b), b)
            prior = b * (1 - env.p_off) + (1 - b) * env.p_on
            c = obs[:, 0] > 0
            a1 = torch.where(c, env.q1, 1 - env.q1)
            a0 = torch.where(c, env.q0, 1 - env.q0)
            b = prior * a1 / (prior * a1 + (1 - prior) * a0)
        if meta["mode"] == "metacog":                 # inspection reveals the danger state
            b = torch.where(obs[:, 9] != 0, (obs[:, 9] > 0).float(), b)
        deaths += int(dead.sum())
        tot_rew += float(rew.sum())
        done = dead | trunc
        if done.any():
            h[done] = 0
            b[done] = 0.25
            J[done] = seasonal_prior(int(done.sum()), DEV)
            mb[done] = 0.5
            if seasonal:
                b[done] = J[done][:, :, 1].sum(1)
    out = {k: torch.stack(v) for k, v in rec.items()}
    # expected lifetime = 1 / per-step death hazard (inf if nobody died)
    out["life"] = n * steps / deaths if deaths else float("inf")
    out["ret_per_step"] = tot_rew / (n * steps)
    return out


def nat_summary(nat):
    return dict(life=nat["life"], ret_per_step=nat["ret_per_step"])


def ridge_axis(X, y, lam=1e-2):
    """Cross-validated linear decoding of y from X. Returns unit axis, R2, fit."""
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device, generator=torch.Generator(device=X.device).manual_seed(0))
    tr, te = idx[: n // 2], idx[n // 2:]
    mu, ym = X[tr].mean(0), y[tr].mean()
    A = X[tr] - mu
    w = torch.linalg.solve(A.T @ A + lam * len(tr) * torch.eye(X.shape[1], device=X.device), A.T @ (y[tr] - ym))
    pred = (X[te] - mu) @ w + ym
    r2 = 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    return w / (w.norm() + 1e-12), r2.item(), w


@torch.no_grad()
def lab(agent, meta, steps, n=2048, cue_at=(), obj_at=None, E=E0, S=S0, hobs=None,
        perturb=None, ablate=None, seed=5, keep_h=False, hobs_at=None):
    """Controlled episode: no real danger, body clamped, cues/objects scheduled.
    cue_at: observation steps at which the cue is on. obj_at: {step: value}.
    perturb: {step: vector added to hidden}. Returns per-step population means."""
    env = VitalWorld(n, mode=meta["mode"], tau=meta["tau"], device=DEV, dmg=meta.get("dmg"), e_hr=meta.get("e_hr"), hide_risk=meta.get("hide_risk"), seed=seed)
    zero, one = torch.zeros(n, device=DEV), torch.ones(n, device=DEV)
    env.d[:] = False
    env.cue[:] = 0
    env.objp[:], env.objv[:] = 0, 0
    env.E[:], env.S[:] = E, S
    torch.manual_seed(seed)
    h = agent.init_hidden(n, DEV)
    obs = env.obs()
    out = dict(p=[], H=[], hm=[])
    hs = []
    for t in range(steps):
        if hobs is not None:
            obs[:, 3] = hobs
        if hobs_at is not None and t in hobs_at:
            obs[:, 3] = hobs_at[t]
        h = agent.core(obs, h, env.S)
        if perturb is not None and t in perturb:
            h = h + perturb[t]
        if ablate is not None:
            w, m = ablate
            h = h - ((h @ w) - m).unsqueeze(1) * w
        dist, _ = agent.heads(h)
        out["p"].append(dist.probs.view(n, -1, 3).sum(2).mean(0))
        out["H"].append(env.H.mean())
        out["hm"].append(h.mean(0))
        if keep_h:
            hs.append(h.clone())
        act = dist.sample()
        nxt = t + 1
        fo = None
        if obj_at is not None and nxt in obj_at:
            fo = (one, one * obj_at[nxt])
        else:
            fo = (zero, zero)
        obs, *_ = env.step(act, force_cue=one if nxt in cue_at else zero, force_d=zero.bool(),
                           force_obj=fo, auto_reset=False)
        env.E[:], env.S[:] = E, S
        obs = env.obs()
    res = {k: torch.stack(v).cpu().numpy() for k, v in out.items()}
    if keep_h:
        res["h"] = torch.stack(hs)
    return res


def half_life(x):
    """Steps after the peak until x falls to half its peak (linear interp);
    nan if the peak is negligible, len(x) if never reached."""
    x = np.asarray(x, float)
    k = int(np.argmax(x))
    pk = x[k]
    if pk < 0.02:
        return float("nan")
    for i in range(k + 1, len(x)):
        if x[i] <= pk / 2:
            return (i - 1 - k) + (x[i - 1] - pk / 2) / (x[i - 1] - x[i] + 1e-12)
    return float(len(x) - k)


def ideal_danger(cues, tau, seasonal=False):
    return seasonal_belief(cues, tau)[0] if seasonal else bayes_belief(cues, tau)


def ideal_half_life(tau, k, steps=POST, seasonal=False):
    """Half-life of ideal-observer belief excess after k consecutive cues."""
    cues = torch.zeros(WARM + k + steps, 1)
    base = ideal_danger(cues, tau, seasonal)[:, 0].numpy()
    cues[WARM:WARM + k] = 1
    b = ideal_danger(cues, tau, seasonal)[:, 0].numpy()
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    # the threat axis decodes log-odds, so its natural comparison is the log-odds excess
    return half_life((b - base)[WARM:]), float((b - base).max()), half_life((lg(b) - lg(base))[WARM:])


@torch.no_grad()
def mood(agent, meta, w, sd, n=2048, hist=100, test=60, seed=11, wm=None, sdm=1.0):
    """Chronic-stress probe. During a 100-step history the agent either sees
    nothing (benign) or random threat cues (p = 0.35 per step) plus
    uncontrollable pain signals (p = 0.1 per step), with the body clamped so
    that the body state is identical in both histories. We then measure
    foraging and threat-axis activity during a quiet test period."""
    res = {}
    for cond in ("benign", "harsh"):
        env = VitalWorld(n, mode=meta["mode"], tau=meta["tau"], device=DEV, dmg=meta.get("dmg"), e_hr=meta.get("e_hr"), hide_risk=meta.get("hide_risk"), seed=seed)
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        zero = torch.zeros(n, device=DEV)
        env.d[:] = False
        env.cue[:] = 0
        env.objp[:], env.objv[:] = 0, 0
        env.E[:], env.S[:] = E0, S0
        torch.manual_seed(seed)
        h = agent.init_hidden(n, DEV)
        obs = env.obs()
        pf, pr, pm = [], [], []
        for t in range(WARM + hist + test):
            h = agent.core(obs, h, env.S)
            dist, _ = agent.heads(h)
            pf.append(dist.probs.view(n, -1, 3).sum(2)[:, FORAGE].mean())
            pr.append((h @ w).mean())
            pm.append((h @ wm).mean() if wm is not None else torch.zeros((), device=DEV))
            act = dist.sample()
            nxt = t + 1
            harsh = cond == "harsh" and WARM <= nxt < WARM + hist
            cue = (torch.rand(n, generator=g, device=DEV) < 0.35).float() if harsh else zero
            obs, *_ = env.step(act, force_cue=cue, force_d=zero.bool(), force_obj=(zero, zero), auto_reset=False)
            if harsh:
                env.pain = (torch.rand(n, generator=g, device=DEV) < 0.1).float()
            env.E[:], env.S[:] = E0, S0
            obs = env.obs()
        res[cond] = (torch.stack(pf).cpu().numpy(), torch.stack(pr).cpu().numpy(), torch.stack(pm).cpu().numpy())
    t0 = WARM + hist
    supp = res["benign"][0] - res["harsh"][0]
    dproj = (res["harsh"][1] - res["benign"][1]) / sd
    dmood = (res["harsh"][2] - res["benign"][2]) / sdm
    # cue-only ideal observer after the same kind of history, for comparison
    seasonal = meta["mode"] == "seasonal"
    gb = torch.Generator().manual_seed(5)
    cues = torch.zeros(WARM + hist + test, 4096)
    cues[WARM:WARM + hist] = (torch.rand(hist, 4096, generator=gb) < 0.35).float()
    b = ideal_danger(cues, meta["tau"], seasonal).mean(1).numpy()
    b0 = ideal_danger(torch.zeros(WARM + hist + test, 1), meta["tau"], seasonal)[:, 0].numpy()
    return dict(supp=supp[t0 - 5:].tolist(), dproj=dproj[t0 - 5:].tolist(), ideal=(b - b0)[t0 - 5:].tolist(),
                dmood=dmood[t0 - 5:].tolist(), late_mood=float(dmood[t0 + 15:t0 + 40].mean()),
                acute=float(supp[t0:t0 + 5].mean()), late=float(supp[t0 + 15:t0 + 40].mean()),
                late_proj=float(dproj[t0 + 15:t0 + 40].mean()), ideal_late=float((b - b0)[t0 + 15:t0 + 40].mean()),
                during=float(supp[WARM + 20:WARM + hist].mean()))


def run_probes(run, untrained=False):
    agent, meta = load(run, untrained)
    mode, tau = meta["mode"], meta["tau"]
    R = dict(mode=mode, tau=tau, seed=meta["seed"], hidden=meta["hidden"], untrained=untrained,
             dmg=meta.get("dmg", 0.25), n_actions=meta.get("n_actions", 9))

    # ---- natural behaviour, decoding, axes -------------------------------
    nat = natural(agent, meta)
    R["life"], R["ret_per_step"] = nat["life"], nat["ret_per_step"]
    X = nat["h"].reshape(-1, agent.hidden)
    flat = lambda k: nat[k].reshape(-1)
    logit_b = torch.logit(flat("b").clamp(1e-3, 1 - 1e-3))
    w, r2b, _ = ridge_axis(X, logit_b)
    R["r2_belief"] = r2b
    axes = {"threat": w}
    for k in ("E", "S", "v", "d"):
        a, r2, _ = ridge_axis(X, flat(k))
        axes[k], R["r2_" + k] = a, r2
    R["axis_cos"] = {f"threat_{k}": float(w @ axes[k]) for k in ("E", "S", "v")}
    R["axis_cos"]["E_S"] = float(axes["E"] @ axes["S"])
    proj = X @ w
    sd, m0 = proj.std().item(), proj.mean().item()
    Xc = X - X.mean(0)
    ev = torch.linalg.eigvalsh(Xc.T @ Xc / len(Xc)).flip(0)
    R["pca_var"] = (ev / ev.sum())[:10].tolist()
    R["participation_ratio"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    wm, sdm = None, 1.0
    if mode == "seasonal":
        wm, R["r2_season_belief"], _ = ridge_axis(X, torch.logit(flat("m").clamp(1e-3, 1 - 1e-3)))
        _, R["r2_season_true"], _ = ridge_axis(X, flat("season"))
        R["axis_cos"]["threat_mood"] = float(w @ wm)
        sdm = (X @ wm).std().item()
    R["threat_var_frac"] = float(proj.var() / Xc.pow(2).sum(1).mean())

    # ---- chronic stress ("mood") -------------------------------------------
    R["mood"] = mood(agent, meta, w, sd, wm=wm, sdm=sdm)

    # ---- H1/H2 persistence and scalability --------------------------------
    T = WARM + 8 + POST
    base = lab(agent, meta, T)
    R["pulse"] = {}
    for k in (1, 2, 4):
        r = lab(agent, meta, T, cue_at=set(range(WARM, WARM + k)))
        supp = base["p"][:, FORAGE] - r["p"][:, FORAGE]
        dH = r["H"] - base["H"]
        dproj = (torch.tensor(r["hm"] - base["hm"], device=DEV) @ w).cpu().numpy() / sd
        last = WARM + k - 1                     # last step with the cue visible
        ihl, ipk, ihl_logit = ideal_half_life(tau, k, seasonal=mode == "seasonal")
        R["pulse"][k] = dict(
            supp=supp[WARM - 5:].tolist(), dH=dH[WARM - 5:].tolist(), dproj=dproj[WARM - 5:].tolist(),
            peak_supp=float(supp[WARM:].max()), peak_dH=float(dH[WARM:].max()),
            peak_proj=float(dproj[WARM:].max()),
            reflex_supp=float(supp[last]),
            after_supp=float(supp[last + 3:last + 8].mean()),
            hl_supp=half_life(supp[last:]), hl_proj=half_life(dproj[last:]), hl_dH=half_life(dH[last:]),
            auc_supp=float(supp[WARM:].sum()), ideal_hl=ihl, ideal_peak=ipk, ideal_hl_logit=ihl_logit)
    R["base_forage"] = float(base["p"][WARM - 10:WARM, FORAGE].mean())
    R["base_H"] = float(base["H"][WARM - 10:WARM].mean())

    # ---- H9 bodily feedback: heart-rate sensation clamped to zero ---------
    b0 = lab(agent, meta, T, hobs=0.0)
    r0 = lab(agent, meta, T, cue_at=set(range(WARM, WARM + 4)), hobs=0.0)
    supp0 = b0["p"][:, FORAGE] - r0["p"][:, FORAGE]
    R["hr_block"] = dict(hl_supp=half_life(supp0[WARM + 3:]), auc_supp=float(supp0[WARM:].sum()),
                         peak_supp=float(supp0[WARM:].max()))

    # ---- H4 on- vs off-manifold perturbation -------------------------------
    g = torch.Generator(device=DEV).manual_seed(3)
    amp = 3 * sd
    on = lab(agent, meta, T, perturb={WARM: amp * w})
    d_on = torch.tensor(on["hm"] - base["hm"], device=DEV)
    on_trace = (d_on @ w).cpu().numpy() / amp
    off_traces, off_beh = [], []
    for _ in range(8):
        u = torch.randn(agent.hidden, generator=g, device=DEV)
        u = u - (u @ w) * w
        u = u / u.norm()
        off = lab(agent, meta, T, perturb={WARM: amp * u})
        off_traces.append(((torch.tensor(off["hm"] - base["hm"], device=DEV)) @ u).cpu().numpy() / amp)
        off_beh.append(base["p"][:, FORAGE] - off["p"][:, FORAGE])
    off_trace = np.mean(off_traces, 0)
    on_beh = base["p"][:, FORAGE] - on["p"][:, FORAGE]
    R["perturb"] = dict(
        on=on_trace[WARM:].tolist(), off=off_trace[WARM:].tolist(),
        on_hl=half_life(on_trace[WARM:]), off_hl=half_life(off_trace[WARM:]),
        on_supp=on_beh[WARM:].tolist(), off_supp=np.mean(off_beh, 0)[WARM:].tolist(),
        on_supp_auc=float(on_beh[WARM:].sum()),
        off_supp_auc=float(np.mean(np.abs(np.sum(np.array(off_beh)[:, WARM:], 1)))),
        on_dH_peak=float((on["H"] - base["H"])[WARM:].max()))

    # ---- linearised dynamics at the resting fixed point --------------------
    rest = lab(agent, meta, WARM + 40, n=512, keep_h=True)
    h_star = rest["h"][WARM:].mean((0, 1))
    env = VitalWorld(4096, mode=mode, tau=tau, device=DEV, seed=9, dmg=meta.get("dmg"), e_hr=meta.get("e_hr"), hide_risk=meta.get("hide_risk"))
    # mean resting observation (closed-loop average), used as a constant input
    env.d[:] = False; env.cue[:] = 0; env.objp[:] = 0; env.objv[:] = 0
    env.E[:], env.S[:] = E0, S0
    env.H[:] = float(base["H"][WARM - 10:WARM].mean())
    env.prev[:] = torch.tensor(base["p"][WARM - 10:WARM].mean(0)[:3], device=DEV)   # hide/forage/take
    o_bar = env.obs().mean(0, keepdim=True)
    with torch.no_grad():
        hh = h_star.unsqueeze(0).clone()
        for _ in range(2000):
            hh = agent.core(o_bar, hh)
    J = torch.autograd.functional.jacobian(lambda z: agent.core(o_bar, z.unsqueeze(0)).squeeze(0), hh.squeeze(0))
    lam, V = torch.linalg.eig(J)
    mag = lam.abs()
    order = mag.argsort(descending=True)
    tcs = (-1 / torch.log(mag[order].clamp(max=0.999999))).tolist()
    v1 = V[:, order[0]].real
    v1 = v1 / (v1.norm() + 1e-12)
    R["jac"] = dict(top_mag=mag[order][:5].tolist(), top_tc=tcs[:5],
                    la_score=float(np.log2(tcs[0] / tcs[1])) if tcs[1] > 0 else float("nan"),
                    cos_threat=float((v1 @ w).abs()),
                    fp_residual=float((agent.core(o_bar, hh) - hh).norm().detach()))

    # ---- H5 generalisation: judgement of an ambiguous object ---------------
    vals = (0.05, 0.25, 0.5, 0.75, 0.95)
    R["judge"] = {}
    for lag in (2, 6):
        t_obj = WARM + 3 + lag
        row = dict(no_cue=[], cue=[])
        for o in vals:
            a = lab(agent, meta, t_obj + 2, obj_at={t_obj: o})
            c = lab(agent, meta, t_obj + 2, obj_at={t_obj: o}, cue_at=set(range(WARM, WARM + 3)))
            row["no_cue"].append(float(a["p"][t_obj, 2]))
            row["cue"].append(float(c["p"][t_obj, 2]))
        R["judge"][lag] = row
    R["judge_vals"] = list(vals)

    # ---- H6 trade-off: threat response as a function of need --------------
    R["tradeoff"] = dict(E=[], forage_cue=[], forage_nocue=[], S=[], forage_cue_S=[], forage_nocue_S=[])
    win = slice(WARM + 3, WARM + 8)
    for e in (0.08, 0.15, 0.25, 0.4, 0.6, 0.9):
        a = lab(agent, meta, WARM + 10, E=e)
        c = lab(agent, meta, WARM + 10, E=e, cue_at=set(range(WARM, WARM + 3)))
        R["tradeoff"]["E"].append(e)
        R["tradeoff"]["forage_nocue"].append(float(a["p"][win, FORAGE].mean()))
        R["tradeoff"]["forage_cue"].append(float(c["p"][win, FORAGE].mean()))
    for s in (0.15, 0.3, 0.5, 0.8, 1.0):
        a = lab(agent, meta, WARM + 10, S=s)
        c = lab(agent, meta, WARM + 10, S=s, cue_at=set(range(WARM, WARM + 3)))
        R["tradeoff"]["S"].append(s)
        R["tradeoff"]["forage_nocue_S"].append(float(a["p"][win, FORAGE].mean()))
        R["tradeoff"]["forage_cue_S"].append(float(c["p"][win, FORAGE].mean()))

    # ---- causal ablation of the threat axis --------------------------------
    abl = (w, m0)
    ba = lab(agent, meta, T, ablate=abl)
    ra = lab(agent, meta, T, cue_at=set(range(WARM, WARM + 4)), ablate=abl)
    sa = ba["p"][:, FORAGE] - ra["p"][:, FORAGE]
    u = torch.randn(agent.hidden, generator=g, device=DEV)
    u = u - (u @ w) * w
    u = u / u.norm()
    mu_u = float((X @ u).mean())
    bu = lab(agent, meta, T, ablate=(u, mu_u))
    ru = lab(agent, meta, T, cue_at=set(range(WARM, WARM + 4)), ablate=(u, mu_u))
    su = bu["p"][:, FORAGE] - ru["p"][:, FORAGE]
    R["ablate"] = dict(
        threat=dict(first=float(sa[WARM]), reflex=float(sa[WARM + 3]), after=float(sa[WARM + 6:WARM + 11].mean()),
                    auc=float(sa[WARM:].sum()), **nat_summary(natural(agent, meta, ablate=abl))),
        random=dict(first=float(su[WARM]), reflex=float(su[WARM + 3]), after=float(su[WARM + 6:WARM + 11].mean()),
                    auc=float(su[WARM:].sum()), **nat_summary(natural(agent, meta, ablate=(u, mu_u)))))
    return R


if __name__ == "__main__":
    run = sys.argv[1]
    untrained = "--untrained" in sys.argv
    R = run_probes(run, untrained)
    name = "probes_untrained.json" if untrained else "probes.json"
    json.dump(R, open(os.path.join(run, name), "w"))
    print(run, "life", round(R["life"], 1), "r2_belief", round(R["r2_belief"], 3),
          "hl4", R["pulse"][4]["hl_supp"], "la", R["jac"]["la_score"])
