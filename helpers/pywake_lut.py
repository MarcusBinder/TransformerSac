"""PyWake yaw look-up tables (deterministic and direction-robust) + the pure
LUT controller the three eval harnesses share.

Every "PyWake" row in pretrained_agents/readme.md is a steady-state
serial-refine setpoint computed once at the nominal direction and held for
the whole run: an oracle, not a deployable controller. A field LUT is
indexed by an ESTIMATED direction (front-row vane, low-passed), and a table
that ignores direction uncertainty over-steers. Following Simley et al.
(2020) the robust table optimises the expected power under a Gaussian smear
of the direction instead.

Four pure-PyWake controllers behind one interface (``LutYawController``):

    mode  LUT            wd fed to the LUT
    1     deterministic  estimated (WdEstimator, front turbine, EWMA)
    2     robust         nominal (constant setpoints)
    3     robust         estimated
    4     deterministic  nominal (= the existing ``--policy pywake`` row)

Frame (verified against PyWakeAgent.predict's "wind" branch and the env's
``yaw_command += delta_wd`` carry): a LUT offset gamma at estimated direction
phi_hat fixes the heading H = phi_hat - gamma. The harness yaw is
``fs.wind_direction - heading`` with ``fs.wind_direction`` pinned at the
nominal direction, so

    yaw_cmd = gamma(phi_hat) - wrap180(phi_hat - wd_nominal)

and modes 2/4 reduce to ``yaw_cmd = gamma(wd_nominal)``.

Under an in-episode wd schedule (Stage-12 exp12w, ``--wd-schedule``) the
frame is no longer pinned: ``target_yaw(wd_hat, wd_frame=fs.wind_direction)``
replaces ``wd_nominal`` by the LIVE frame in the formula above (modes 2/4
then index the table at the live frame itself), so the heading
H = phi - gamma(phi) stays correct while the frame rotates. With
``wd_frame=None`` (or equal to the nominal) the numbers are unchanged.

Robust objective (Simley 2020; notebook yaw_with_uncertanty.ipynb cell 4):
the quadrature node at ``wd_hat + Delta`` is evaluated with ``yaw = gamma +
Delta`` -- the heading is frozen, so both the wake geometry and the rotor
misalignment move with the direction error.

The PyWake model is PyWakeAgent's verbatim (Blondel_Cathelain_2020 on
LillgrundSite with CrespoHernandez + JimenezWakeDeflection, TI per call,
tilt 0), and ``serial_refine`` is PyWakeAgent's yaw_optimizer_srf_vect on a
generic objective (6 passes x 7 candidates, range halved per pass,
upstream->downstream order per direction computed at ``wd + 1e-3``, improve-
only acceptance) plus two additions: a tie-break toward the most positive
yaw among near-equal candidates, and an optional farm-wide sign constraint.

Library only: the harnesses call ``build_controller`` at start-up and the
tables live in memory for the run (a table takes seconds at sigma 0 and
under a minute at sigma 2.3 on 9 turbines). Nothing is written to disk; the
run command line records mode / sigma / tau / turbine / sign / deadband.
"""
from __future__ import annotations

import time

import numpy as np

MODEL_DESCRIPTION = (
    "Blondel_Cathelain_2020(LillgrundSite, turbine, "
    "turbulenceModel=CrespoHernandez, deflectionModel=JimenezWakeDeflection), "
    "TI per call, tilt 0 (PyWakeAgent verbatim)"
)

# Default LUT index grid [deg]: 200..300 in 1 deg bins (+-50 around 270).
DEFAULT_WD_GRID = np.arange(200.0, 300.5, 1.0)


def wrap180(a):
    """Wrap degrees to [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------
# PyWake model + objective
# ----------------------------------------------------------------------------

def build_wfm(turbine, x, y):
    """PyWakeAgent's wind-farm model, verbatim (incl. site.initial_position)."""
    from py_wake.examples.data.lillgrund import LillgrundSite
    from py_wake.deflection_models.jimenez import JimenezWakeDeflection
    from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
    from py_wake.turbulence_models import CrespoHernandez

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    site = LillgrundSite()
    site.initial_position = np.array([x, y]).T
    return Blondel_Cathelain_2020(
        site,
        turbine,
        turbulenceModel=CrespoHernandez(),
        deflectionModel=JimenezWakeDeflection(),
    )


def quadrature(sigma, step=0.5, nsigma=3.5):
    """Symmetric trapezoid grid for a zero-mean Gaussian direction error.

    Returns (nodes [deg], normalised weights). sigma <= 0 -> a single node at
    0 (recovers the deterministic objective). A grid, not Gauss-Hermite: farm
    power is not smooth in wd (wakes slide on and off rotors) and few-node
    Gauss-Hermite aliases that (Simley et al. use a 1 deg grid).
    """
    if sigma <= 0:
        return np.array([0.0]), np.array([1.0])
    n = int(np.floor(nsigma * sigma / step + 1e-9))
    nodes = np.arange(-n, n + 1, dtype=float) * step
    w = np.exp(-0.5 * (nodes / sigma) ** 2)
    return nodes, w / w.sum()


def make_objective(wfm, x, y, ws, ti):
    """Return f(gammas, wd_hat, quad) -> expected farm power [W] per candidate.

    gammas: (n_cand, n_wt) commanded offsets [deg] (a 1-D (n_wt,) is one
    candidate). quad: (nodes, weights) from ``quadrature``. All candidates x
    nodes go into ONE PyWake call, stacked along the wd axis.

    py_wake (2.6.11 here) sorts a non-monotonic wd axis WITHOUT permuting the
    per-case yaw, silently mis-assigning powers by MW. Sorting the cases by
    wd and inverting the permutation afterwards is exact on every version.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_wt = len(x)

    def f(gammas, wd_hat, quad):
        nodes, w = quad
        gammas = np.atleast_2d(np.asarray(gammas, dtype=float))
        n_c, n_q = gammas.shape[0], len(w)
        if gammas.shape[1] != n_wt:
            raise ValueError(f"gammas has {gammas.shape[1]} turbines, layout {n_wt}")
        wd_cases = np.tile(float(wd_hat) + nodes, n_c)                  # (n_c*n_q,)
        # Heading frozen: node at wd_hat + Delta is evaluated with gamma + Delta.
        yaw = (gammas[:, None, :] + nodes[None, :, None]).reshape(n_c * n_q, n_wt)
        o = np.argsort(wd_cases, kind="stable")
        sim = wfm(x, y, wd=wd_cases[o], ws=[ws], TI=ti,
                  yaw=yaw[o].T[:, :, None], tilt=0)
        P = np.empty(n_c * n_q)
        P[o] = sim.Power.values.sum(axis=0)[:, 0]
        return P.reshape(n_c, n_q) @ w

    return f


def turbine_order(x, y, wd):
    """Turbine indices most-upstream first (PyWakeAgent's rotation, incl. the
    +1e-3 that breaks the perfect-alignment tie). Ordering only."""
    theta = np.radians((270.0 - (float(wd) + 1e-3)) % 360.0)
    x_rot = np.asarray(x, float) * np.cos(theta) + np.asarray(y, float) * np.sin(theta)
    return np.argsort(x_rot, kind="stable")


# ----------------------------------------------------------------------------
# Serial refine
# ----------------------------------------------------------------------------

def serial_refine(objective, wd_hat, quad, n_wt, order, yaw_max,
                  refine_pass_n=6, yaw_n=7, tie_tol=1e-4, sign="free"):
    """PyWakeAgent's serial-refine on a generic objective.

    Per pass a symmetric candidate grid of ``yaw_n`` offsets over
    ``+-range`` (range = yaw_max, halved after each pass) is tried for every
    turbine in ``order``, one at a time with the others held; a candidate is
    accepted only if it improves on the current objective value. Candidates
    are clipped to [-yaw_max, yaw_max].

    Additions to PyWake-as-is:
      * tie-break: among candidates within ``tie_tol`` (relative) of the best,
        the most POSITIVE yaw is taken (np.argmax would fall to the negative
        end of an exact tie);
      * ``sign="farm"``: once the most-upstream turbine of a pass has been
        optimised, every later turbine's candidates are clipped to the
        half-line of its sign (0 stays allowed). ``sign="free"`` = PyWake +
        tie-break only.
    """
    if sign not in ("free", "farm"):
        raise ValueError(f"sign must be 'free' or 'farm', got {sign!r}")
    order = [int(i) for i in order]
    gamma = np.zeros(n_wt)
    P_cur = float(objective(gamma[None, :], wd_hat, quad)[0])
    rng = float(yaw_max)
    for _ in range(refine_pass_n):
        offs = np.linspace(-rng, rng, yaw_n)
        rng /= 2.0
        lead_sign = 0.0
        for pos, i in enumerate(order):
            cands = np.clip(gamma[i] + offs, -yaw_max, yaw_max)
            if sign == "farm" and pos > 0 and lead_sign != 0.0:
                cands = np.clip(cands, 0.0, None) if lead_sign > 0 else np.clip(cands, None, 0.0)
            trial = np.repeat(gamma[None, :], len(cands), axis=0)
            trial[:, i] = cands
            P = objective(trial, wd_hat, quad)
            best = float(P.max())
            if best > P_cur:
                ok = P >= best - tie_tol * abs(best)
                j = int(np.argmax(np.where(ok, cands, -np.inf)))
                gamma[i] = cands[j]
                P_cur = float(P[j])
            if sign == "farm" and pos == 0:
                lead_sign = float(np.sign(gamma[i]))
    return gamma


def build_lut(turbine, x, y, ws, ti, wd_grid=None, sigma_wd=0.0, yaw_max=45.0,
              sign="free", quad_step=0.5, quad_nsigma=3.5, verbose=True):
    """Optimise offsets for every direction bin. Returns (wd_grid, lut[n_wd, n_wt])."""
    wd_grid = DEFAULT_WD_GRID.copy() if wd_grid is None else np.asarray(wd_grid, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    wfm = build_wfm(turbine, x, y)
    obj = make_objective(wfm, x, y, ws, ti)
    quad = quadrature(sigma_wd, step=quad_step, nsigma=quad_nsigma)
    n_wt = len(x)
    lut = np.zeros((len(wd_grid), n_wt))
    t0 = time.time()
    for j, wd in enumerate(wd_grid):
        lut[j] = serial_refine(obj, float(wd), quad, n_wt, turbine_order(x, y, wd),
                               yaw_max, sign=sign)
        if verbose:
            print(f"  wd={wd:6.1f}  gamma={np.array2string(lut[j], precision=1, suppress_small=True)}"
                  f"  ({time.time() - t0:5.0f} s)", flush=True)
    if verbose:
        print(f"[pywake_lut] built {len(wd_grid)}-bin table (sigma={sigma_wd:g}, sign={sign}, "
              f"ws={ws:g}, ti={ti:g}, yaw_max={yaw_max:g}) in {time.time() - t0:.1f} s", flush=True)
    return wd_grid, lut


def evaluate_lut(objective, lut, wd_grid, sigma, quad_step=0.5, quad_nsigma=3.5):
    """Expected farm power [W] of a LUT under a sigma-deg direction error, per bin."""
    quad = quadrature(sigma, step=quad_step, nsigma=quad_nsigma)
    return np.array([float(objective(lut[j], float(wd), quad)[0])
                     for j, wd in enumerate(wd_grid)])


# ----------------------------------------------------------------------------
# Controller + eval_single_fast policy adapter
# ----------------------------------------------------------------------------

class LutYawController:
    """Pure LUT yaw controller: mode 1-4, harness-frame yaw targets.

    Args:
        mode: 1 det/est, 2 rob/nominal, 3 rob/est, 4 det/nominal.
        lut_det, lut_rob: (n_wd, n_turb) tables on ``wd_grid`` (either may be
            None when the mode does not use it).
        wd_nominal: the pinned ``fs.wind_direction`` of a static harness run
            (270, or 235 for the static-235 rows); a scheduled run passes the
            live frame to ``target_yaw(wd_frame=...)`` instead.
        deadband_deg: per-turbine hold: a new target replaces the held one
            only where |new - held| > deadband (0 = always follow).
    """

    MODE_DESC = {
        1: ("deterministic", "estimated"),
        2: ("robust", "nominal"),
        3: ("robust", "estimated"),
        4: ("deterministic", "nominal"),
    }

    def __init__(self, mode, lut_det, lut_rob, wd_grid, n_turb, wd_nominal=270.0,
                 deadband_deg=0.0):
        mode = int(mode)
        if mode not in self.MODE_DESC:
            raise ValueError(f"lut mode must be 1..4, got {mode}")
        self.mode = mode
        self.lut_name, self.wd_feed = self.MODE_DESC[mode]
        self.uses_estimate = self.wd_feed == "estimated"
        lut = lut_rob if self.lut_name == "robust" else lut_det
        if lut is None:
            raise ValueError(f"mode {mode} needs the {self.lut_name} LUT")
        self.lut = np.asarray(lut, dtype=float)
        self.wd_grid = np.asarray(wd_grid, dtype=float)
        if self.lut.shape != (len(self.wd_grid), int(n_turb)):
            raise ValueError(f"LUT shape {self.lut.shape} != ({len(self.wd_grid)}, {n_turb})")
        self.n_turb = int(n_turb)
        self.wd_nominal = float(wd_nominal)
        self.deadband_deg = float(deadband_deg)
        self.reset()

    def reset(self):
        self.last_phi = float("nan")
        self.last_target = None

    def gamma(self, phi):
        """LUT offsets (n_turb,) at direction phi: unwrapped around the nominal
        direction, linearly interpolated per turbine, clamped at the grid ends."""
        phi_u = self.wd_nominal + float(wrap180(float(phi) - self.wd_nominal))
        phi_u = float(np.clip(phi_u, self.wd_grid[0], self.wd_grid[-1]))
        return np.array([np.interp(phi_u, self.wd_grid, self.lut[:, i])
                         for i in range(self.n_turb)])

    def target_yaw(self, wd_hat, wd_frame=None):
        """Harness-frame yaw targets (n_turb,) for a direction estimate.

        Modes 1/3 use ``wd_hat`` when finite (the frame otherwise, i.e. before
        the estimator's warm-start); modes 2/4 always use the frame.
        yaw_cmd = gamma(phi) - wrap180(phi - frame), with ``frame`` =
        ``wd_frame`` (the live ``fs.wind_direction`` under a wd schedule) or
        ``wd_nominal`` when None (static harness runs; identical numbers).
        """
        frame = self.wd_nominal if wd_frame is None else float(wd_frame)
        if self.uses_estimate and wd_hat is not None and np.isfinite(wd_hat):
            phi = float(wd_hat)
        else:
            phi = frame
        target = self.gamma(phi) - float(wrap180(phi - frame))
        if self.last_target is not None and self.deadband_deg > 0:
            move = np.abs(target - self.last_target) > self.deadband_deg
            target = np.where(move, target, self.last_target)
        self.last_phi = phi
        self.last_target = np.asarray(target, dtype=float)
        return self.last_target.copy()

    def describe(self):
        g0 = self.gamma(self.wd_nominal)
        return (f"LUT mode {self.mode}: {self.lut_name} table fed the {self.wd_feed} wd; "
                f"row at {self.wd_nominal:g}: {np.array2string(g0, precision=2, suppress_small=True)}"
                f"; deadband {self.deadband_deg:g} deg")


class LutPolicy:
    """eval_single_fast adapter for ``LutYawController`` (EvalPretrainedAgentGif).

    ``UseEnv = True`` so eval_single_fast injects env/yaw limits; no
    ``pywakeagent`` attr, so its ``update_wind`` warm-up is skipped. Reads
    ``env.unwrapped.wd_est`` (FarmEvalWdEst) each decision, "yaw" branch =
    clip((target - current_yaw)/yaw_step_env) like PyWakeAgent, "wind" branch
    = scale_yaw(target). Keeps a (time, phi, target) trace for the .nc.
    """

    UseEnv = True

    def __init__(self, ctrl, env, yaw_max, yaw_min):
        self.ctrl = ctrl
        self.env = env
        self.yaw_max = float(yaw_max)
        self.yaw_min = float(yaw_min)
        self.trace = []

    def scale_yaw(self, yaws):
        return (yaws - self.yaw_min) / (self.yaw_max - self.yaw_min) * 2 - 1

    def predict(self, obs=None, deterministic=True):
        base = self.env.unwrapped
        if getattr(base, "derate_action", False):
            raise ValueError("LutPolicy produces yaw-only actions (derate_action=True env)")
        wd_hat = float(getattr(base, "wd_est", float("nan")))
        # The frame the target is expressed in = the agent's yaw frame: the env's
        # phi_frame under yaw_frame_source="est" (2026-10-29; the LUT then
        # converts gamma(phi_hat) by -wrap180(phi_hat - phi_frame), i.e. only by
        # the frame's slew lag), else the live simulator frame (equals
        # wd_nominal in a static run, follows the schedule otherwise).
        if getattr(base, "yaw_frame_source", "true") == "est":
            frame = float(base.phi_frame)
        else:
            frame = float(base.fs.wind_direction)
        target = self.ctrl.target_yaw(wd_hat, wd_frame=frame)
        self.trace.append((float(base.fs.time), float(self.ctrl.last_phi), target.copy()))
        if base.ActionMethod == "yaw":
            cur = np.asarray(getattr(base, "current_yaw_ctrl", base.current_yaw), dtype=float)
            action = np.clip((target - cur) / float(base.yaw_step_env), -1.0, 1.0)
        elif base.ActionMethod == "wind":
            action = self.scale_yaw(np.clip(target, self.yaw_min, self.yaw_max))
        else:
            raise ValueError(f"Unknown ActionMethod {base.ActionMethod!r}")
        return np.asarray(action, dtype=np.float32), None


def trace_on_grid(trace, times, n_turb):
    """Forward-fill a (time, phi, target) decision trace onto sim times.
    Returns (phi[time], target[time, turb]); nan before the first decision."""
    times = np.asarray(times, dtype=float)
    phi = np.full(len(times), np.nan)
    tgt = np.full((len(times), n_turb), np.nan)
    if not trace:
        return phi, tgt
    t_dec = np.array([t for t, _, _ in trace], dtype=float)
    idx = np.searchsorted(t_dec, times, side="right") - 1
    ok = idx >= 0
    phi[ok] = np.array([p for _, p, _ in trace])[idx[ok]]
    tgt[ok] = np.array([g for _, _, g in trace])[idx[ok]]
    return phi, tgt


# ----------------------------------------------------------------------------
# Turbine helper + the harness entry point
# ----------------------------------------------------------------------------

def load_turbine(turbtype):
    """PyWake turbine object for a turbtype string (the EvalPretrainedAgentGif
    mapping: iea22 -> IEA_22MW_H2S, iea22h2 -> IEA_22MW_HAWC2Surrogate)."""
    t = str(turbtype)
    if t == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif t == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    elif t.lower() in ("iea22", "iea22h2"):
        cls = "IEA_22MW_HAWC2Surrogate" if t.lower() == "iea22h2" else "IEA_22MW_H2S"
        try:
            import helpers.iea_22_rwt as mod   # tracking branch vendors it here
        except ImportError:
            import iea_22_rwt as mod           # LES branch: top-level
        WT = getattr(mod, cls)
    else:
        raise ValueError(f"Unknown turbine type: {turbtype}")
    return WT()



def build_controller(mode, turbine, x, y, ws, ti, yaw_max, sign="farm", sigma=2.3,
                     wd_nominal=270.0, deadband_deg=0.0, verbose=True):
    """Generate the table(s) ``mode`` needs and return a ``LutYawController``.

    The deterministic table is always built (seconds); the robust one only for
    modes 2/3 (they are the only readers). ``verbose`` prints one line per
    direction bin plus the elapsed time, so the run log carries the schedule.
    """
    mode = int(mode)
    if mode not in LutYawController.MODE_DESC:
        raise ValueError(f"lut mode must be 1..4, got {mode}")
    t0 = time.time()
    if verbose:
        print(f"[pywake_lut] mode {mode}: building the deterministic table "
              f"({type(turbine).__name__}, {len(x)} turbines, sign={sign})", flush=True)
    wd_grid, det = build_lut(turbine, x, y, ws, ti, sigma_wd=0.0, yaw_max=yaw_max,
                             sign=sign, verbose=verbose)
    rob = None
    if mode in (2, 3):
        if verbose:
            print(f"[pywake_lut] mode {mode}: building the robust table (sigma={sigma:g})",
                  flush=True)
        _, rob = build_lut(turbine, x, y, ws, ti, wd_grid=wd_grid, sigma_wd=sigma,
                           yaw_max=yaw_max, sign=sign, verbose=verbose)
    if verbose:
        print(f"[pywake_lut] tables ready in {time.time() - t0:.1f} s", flush=True)
    return LutYawController(mode, det, rob, wd_grid, len(x), wd_nominal=wd_nominal,
                            deadband_deg=deadband_deg)
