"""VitalWorld: a vectorised survival environment with a minimal body.

The body has an energy store E, skin integrity S and a heart rate H that the
agent sets. A hidden danger state d follows a two-state Markov chain and is
only signalled by an unreliable cue. All tensors are batched over N
environments and live on `device`.
"""
import torch

OBS_DIM = 13
N_ACTIONS = 9  # (hide, forage, take) x (heart down, hold, up)
HIDE, FORAGE, TAKE = 0, 1, 2
MODES = ("mortal", "score", "score_cheap", "oracle", "substrate", "seasonal", "metacog", "social")
INSPECT = 3  # metacog world only: pay energy to see the next danger state
# Score exchange rates, in units of one food item (0.10 energy). "score" matches the
# final Mortal parameters; "score_cheap" is the original version, whose rates were
# left over from the first pilot (heart rate 2.5x too cheap). Both are reported.
SCORE_RATES = {"score": dict(food=1.0, good=1.2, bad=-1.0, attack=-2.5, hr=-0.3),
               "score_cheap": dict(food=1.0, good=1.5, bad=-1.5, attack=-2.0, hr=-0.12)}


class VitalWorld:
    def __init__(self, n, mode="mortal", tau=12.0, device="cpu", seed=0,
                 max_steps=1000, score_len=500, dmg=None, e_hr=None, hide_risk=None):
        assert mode in MODES
        self.n, self.mode, self.tau, self.dev = n, mode, float(tau), device
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.scored = mode in SCORE_RATES
        self.score_rates = SCORE_RATES.get(mode)
        self.max_steps = score_len if self.scored else max_steps
        # danger chain: stationary P(danger) = 0.25
        self.p_off = 1.0 / self.tau
        self.p_on = self.p_off * 0.25 / 0.75
        # seasonal world: a slow hidden season sets how often danger starts
        # (stationary danger 0.1 in calm, 0.5 in harsh seasons; mean switch time 200 steps)
        self.p_season = 1.0 / 200
        self.p_on_calm = self.p_off * 0.1 / 0.9
        self.p_on_harsh = self.p_off
        self.q1, self.q0 = 0.35, 0.03          # cue prob in danger / safety
        self.p_food, self.food_gain = 0.4, 0.10
        self.obj_gain, self.obj_dmg = 0.12, 0.10
        self.p_attack, self.attack_dmg = 0.6, 0.25
        if dmg is not None:
            self.attack_dmg = float(dmg)              # vulnerability sweep
        self.n_main = 4 if mode == "metacog" else 3
        self.n_actions = 3 * self.n_main
        self.inspect_cost = 0.004
        self.hr_protect, self.hr_step = 0.8, 0.25
        self.e_base, self.e_hr, self.e_leak = 0.008, 0.03, 0.006
        if e_hr is not None:
            self.e_hr = float(e_hr)                    # affordable-heart condition
        # social world: a scripted neighbour forages with prob. 0.7 and is attacked
        # like the agent when it forages in danger; its pain is visible on channel 9
        self.p_nforage = 0.7
        # fight-or-flight world: danger can also reach a hiding agent; a raised
        # heart rate protects in the same way as when foraging
        self.hide_risk = float(hide_risk) if hide_risk else 0.0
        self.heal, self.heal_cost = 0.02, 0.004
        self.p_obj = 0.1
        self.intero_noise = 0.05
        z = lambda: torch.zeros(n, device=device)
        self.d, self.E, self.S, self.H = z().bool(), z(), z(), z()
        self.season = z().bool()
        self.cue, self.objp, self.objv = z(), z(), z()
        self.pain, self.food, self.objout = z(), z(), z()
        self.prev = torch.zeros(n, 3, device=device)
        self.insp = torch.zeros(n, device=device)    # metacog: +1 danger / -1 safe seen by inspecting
        self.npain = torch.zeros(n, device=device)   # social: neighbour was attacked last step
        self.t = torch.zeros(n, dtype=torch.long, device=device)
        self.reset_all()

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.dev)

    def _new_objects(self, idx=None):
        n = self.n
        present = (self.rand(n) < self.p_obj).float()
        u = self.rand(n)
        extreme = torch.where(self.rand(n) < 0.5, 0.1 * u, 1 - 0.1 * u)
        val = torch.where(self.rand(n) < 0.8, extreme, self.rand(n))
        self.objp, self.objv = present, val * present

    def _reset(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        self.E[mask] = 0.2 + 0.8 * self.rand(k)
        self.S[mask] = 0.3 + 0.7 * self.rand(k)
        self.H[mask] = 0.0
        self.d[mask] = self.rand(k) < 0.25
        if self.mode == "seasonal":
            self.season[mask] = self.rand(k) < 0.5
            self.d[mask] = self.rand(k) < torch.where(self.season[mask], 0.5, 0.1)
        self.cue[mask] = 0.0
        self.pain[mask] = 0.0
        self.food[mask] = 0.0
        self.objout[mask] = 0.0
        self.prev[mask] = 0.0
        self.insp[mask] = 0.0
        self.npain[mask] = 0.0
        self.t[mask] = 0

    def reset_all(self):
        self._reset(torch.ones(self.n, dtype=torch.bool, device=self.dev))
        self._new_objects()
        return self.obs()

    def obs(self):
        o = torch.zeros(self.n, OBS_DIM, device=self.dev)
        o[:, 0], o[:, 1], o[:, 2], o[:, 3] = self.cue, self.objp, self.objv, self.H
        o[:, 4], o[:, 5], o[:, 6] = self.pain, self.food, self.objout
        if not self.scored:
            noise = self.intero_noise * torch.randn(self.n, 2, generator=self.g, device=self.dev)
            o[:, 7] = self.E + noise[:, 0]
            o[:, 8] = self.S + noise[:, 1]
        if self.mode == "oracle":
            o[:, 9] = self.d.float()
        if self.mode == "metacog":
            o[:, 9] = self.insp
        if self.mode == "social":
            o[:, 9] = self.npain
        o[:, 10:13] = self.prev
        return o

    def step(self, action, force_cue=None, force_d=None, force_obj=None, auto_reset=True):
        """Advance one step. `force_*` are used by lab probes only."""
        main, heart = action // 3, (action % 3 - 1).float()
        self.H = (self.H + self.hr_step * heart).clamp(0, 1)
        forage, take, hide = main == FORAGE, main == TAKE, main == HIDE

        food = forage & (self.rand(self.n) < self.p_food)
        took = take & (self.objp > 0)
        good = took & (self.rand(self.n) < self.objv)
        bad = took & ~good
        attack = forage & self.d & (self.rand(self.n) < self.p_attack * (1 - self.hr_protect * self.H))
        if self.hide_risk > 0:
            attack = attack | (~forage & self.d & (self.rand(self.n) < self.hide_risk * (1 - self.hr_protect * self.H)))
        healing = hide & (self.S < 1) & (self.E > 0.2)

        self.pain, self.food = attack.float(), food.float()
        self.objout = good.float() - bad.float()
        self.prev = torch.nn.functional.one_hot(main, self.n_main)[:, :3].float()
        inspect = main == INSPECT

        if self.scored:
            r = self.score_rates
            reward = 0.1 * (r["food"] * food.float() + r["good"] * good.float() + r["bad"] * bad.float()
                            + r["attack"] * attack.float() + r["hr"] * self.H)
            dead = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        else:
            self.E = self.E + self.food_gain * food + self.obj_gain * good - self.heal_cost * healing
            self.S = self.S - self.attack_dmg * attack - self.obj_dmg * bad + self.heal * healing
            self.E = self.E - self.e_base - self.e_hr * self.H - self.e_leak * (1 - self.S.clamp(0, 1))
            if self.mode == "metacog":
                self.E = self.E - self.inspect_cost * inspect
            self.E, self.S = self.E.clamp(max=1), self.S.clamp(max=1)
            dead = (self.E <= 0) | (self.S <= 0)
            reward = 0.05 * (~dead).float()

        if self.mode == "social":
            nforage = self.rand(self.n) < self.p_nforage
            self.npain = (nforage & self.d & (self.rand(self.n) < self.p_attack)).float()

        # world moves on
        if self.mode == "seasonal":
            self.season = self.season ^ (self.rand(self.n) < self.p_season)
            p_on = torch.where(self.season, self.p_on_harsh, self.p_on_calm)
        else:
            p_on = self.p_on
        flip = self.rand(self.n)
        self.d = torch.where(self.d, flip >= self.p_off, flip < p_on)
        if force_d is not None:
            self.d = force_d.clone()
        if self.mode == "metacog":
            self.insp = torch.where(inspect, 2 * self.d.float() - 1, torch.zeros_like(self.insp))
        self.cue = (self.rand(self.n) < torch.where(self.d, self.q1, self.q0)).float()
        if force_cue is not None:
            self.cue = force_cue.clone()
        self._new_objects()
        if force_obj is not None:
            self.objp, self.objv = force_obj[0].clone(), force_obj[1].clone()

        self.t += 1
        trunc = (self.t >= self.max_steps) & ~dead
        done = dead | trunc
        if auto_reset:
            self._reset(done)
        return self.obs(), reward, dead, trunc


def bayes_belief(cues, tau, b0=0.25, q1=0.35, q0=0.03):
    """Ideal-observer posterior P(danger_t | cues up to t) for a batch of cue
    sequences (T, N). Ignores the extra evidence from attacks."""
    p_off = 1.0 / tau
    p_on = p_off * 0.25 / 0.75
    b = torch.full_like(cues[0], b0)
    out = []
    for c in cues:
        prior = b * (1 - p_off) + (1 - b) * p_on
        l1 = torch.where(c > 0, q1, 1 - q1)
        l0 = torch.where(c > 0, q0, 1 - q0)
        b = prior * l1 / (prior * l1 + (1 - prior) * l0)
        out.append(b)
    return torch.stack(out)


def seasonal_prior(n, device="cpu"):
    """Stationary joint P(season, danger) for the seasonal world, shape (n, 2, 2)."""
    J = torch.zeros(n, 2, 2, device=device)
    J[:, 0, 0], J[:, 0, 1] = 0.5 * 0.9, 0.5 * 0.1
    J[:, 1, 0], J[:, 1, 1] = 0.5 * 0.5, 0.5 * 0.5
    return J


def seasonal_predict(J, tau, p_season=1.0 / 200):
    """One step of the seasonal world's hidden dynamics applied to J[season, danger]."""
    p_off = 1.0 / tau
    Js = J * (1 - p_season) + J.flip(1) * p_season
    p_on = torch.tensor([p_off * 0.1 / 0.9, p_off], device=J.device).view(1, 2)
    d0, d1 = Js[:, :, 0], Js[:, :, 1]
    return torch.stack([d0 * (1 - p_on) + d1 * p_off, d1 * (1 - p_off) + d0 * p_on], 2)


def seasonal_cue_update(J, c, q1=0.35, q0=0.03):
    l1 = torch.where(c > 0, q1, 1 - q1).unsqueeze(1)
    l0 = torch.where(c > 0, q0, 1 - q0).unsqueeze(1)
    J = torch.stack([J[:, :, 0] * l0, J[:, :, 1] * l1], 2)
    return J / J.sum((1, 2), keepdim=True)


def seasonal_belief(cues, tau):
    """Cue-only ideal observer for the seasonal world. Returns P(danger) and
    P(harsh season), each of shape (T, N)."""
    J = seasonal_prior(cues.shape[1], cues.device)
    od, os_ = [], []
    for c in cues:
        J = seasonal_cue_update(seasonal_predict(J, tau), c)
        od.append(J[:, :, 1].sum(1))
        os_.append(J[:, 1, :].sum(1))
    return torch.stack(od), torch.stack(os_)
