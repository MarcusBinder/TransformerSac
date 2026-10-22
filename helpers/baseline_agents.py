"""Duck-typed PolicyEvaluator agents for ``--baseline_eval_policy`` (optuna sweep).

The sweep scores every trial as ``(E_rl - E_greedy) / (E_lut3 - E_greedy)`` per
eval spec. Those two references must be MEASURED through the exact trainer
eval path (same env factories, seeds, burn-in, step count, spec names), not
read off a table: the harness LUT's internal wake model (CrespoHernandez)
differs from the PyWake backend's (STF2017), and the PyWake backend has no wd
estimator, so "lut3" here is the robust (sigma 2.3) table fed the TRUE live wd
(``wd_feed = "true"``; recorded in the metrics so nobody mistakes it for the
DWM cell's estimated-wd lut3).

``PolicyEvaluator`` only needs ``eval()``, ``train()`` and
``act(envs, obs, deterministic)``; the agents below read ``wd``,
``current_yaw``, ``yaw_step_env`` and ``ActionMethod`` through the vector env's
``get_attr`` (MultiLayoutEnv properties) and mirror ``LutPolicy.predict``'s
action math (pywake_lut.py), so the per-env action equals what the harness LUT
policy would send to the same WindFarmEnv. Eval envs are never shuffled
(PolicyEvaluator default); a shuffled env is refused because the table is in
physical turbine order.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def _vec(envs):
    """The vector env that answers get_attr (RecordEpisodeVals wraps it as .env)."""
    return envs.env if hasattr(envs, "env") else envs


class _TargetYawAgent:
    """Base: per-env physical yaw target -> env action, padded to max_turbines."""

    def __init__(self, yaw_min: float, yaw_max: float):
        self.yaw_min = float(yaw_min)
        self.yaw_max = float(yaw_max)

    # PolicyEvaluator protocol (no networks: nothing to switch)
    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def target(self, wd: float, n_turb: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def scale_yaw(self, yaws: np.ndarray) -> np.ndarray:
        return (yaws - self.yaw_min) / (self.yaw_max - self.yaw_min) * 2 - 1

    def act(self, envs, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        vec = _vec(envs)
        wds = vec.get_attr("wd")
        cur = vec.get_attr("current_yaw")
        steps = vec.get_attr("yaw_step_env")
        methods = vec.get_attr("ActionMethod")
        shuffled = vec.get_attr("is_shuffled")
        if any(bool(s) for s in shuffled):
            raise RuntimeError("baseline agents need unshuffled eval envs (LUT is in "
                               "physical turbine order); PolicyEvaluator never shuffles")
        n_env, max_turb = int(obs.shape[0]), int(obs.shape[1])
        actions = np.zeros((n_env, max_turb), dtype=np.float32)
        for e in range(n_env):
            cur_e = np.asarray(cur[e], dtype=float)
            n = len(cur_e)
            tgt = np.asarray(self.target(float(wds[e]), n), dtype=float)
            if methods[e] == "yaw":
                a = np.clip((tgt - cur_e) / float(steps[e]), -1.0, 1.0)
            elif methods[e] == "wind":
                a = self.scale_yaw(np.clip(tgt, self.yaw_min, self.yaw_max))
            else:
                raise ValueError(f"Unknown ActionMethod {methods[e]!r}")
            actions[e, :n] = a
        return actions


class LutEvalAgent(_TargetYawAgent):
    """PyWake-LUT controller fed the TRUE live wd: target = ctrl.gamma(wd)."""

    def __init__(self, ctrl, yaw_min: float, yaw_max: float):
        super().__init__(yaw_min, yaw_max)
        if float(getattr(ctrl, "deadband_deg", 0.0)) != 0.0:
            raise ValueError("LutEvalAgent shares one controller across envs: deadband must be 0")
        self.ctrl = ctrl

    def target(self, wd: float, n_turb: int) -> np.ndarray:
        # LutPolicy.predict: target_yaw(wd_hat, wd_frame=live wd); with
        # wd_hat == frame == wd the frame term vanishes and target == gamma(wd).
        return self.ctrl.target_yaw(wd, wd_frame=wd)


class GreedyEvalAgent(_TargetYawAgent):
    """Yaw target 0 on every turbine (the greedy reference)."""

    def target(self, wd: float, n_turb: int) -> np.ndarray:
        return np.zeros(n_turb, dtype=float)


LUT_META = {"mode": 3, "wd_feed": "true", "sigma": 2.3, "sign": "farm"}


def make_baseline_agent(policy: str, *, wind_turbine, x, y, ws: float, ti: float,
                        yaw_min: float, yaw_max: float, wd_nominal: float, table_cache: dict = None):
    """Build the agent for one eval spec.

    The LUT tables depend on (turbine, layout, ws, ti, yaw_max, sign, sigma)
    but NOT on wd_nominal (it only sets gamma's unwrap centre), so one build
    serves every spec at the same ws: ``table_cache`` keeps the first
    controller and later specs re-wrap its tables with their own wd_nominal.
    """
    if policy == "greedy":
        return GreedyEvalAgent(yaw_min, yaw_max)
    if policy == "lut3":
        from helpers.pywake_lut import LutYawController, build_controller
        key = (float(ws), float(ti), float(yaw_max), tuple(np.round(x, 3)), tuple(np.round(y, 3)))
        cache = table_cache if table_cache is not None else {}
        if key not in cache:
            cache[key] = build_controller(LUT_META["mode"], wind_turbine, x, y, ws=float(ws), ti=float(ti),
                                          yaw_max=float(yaw_max), sign=LUT_META["sign"],
                                          sigma=LUT_META["sigma"], wd_nominal=float(wd_nominal))
        base = cache[key]
        ctrl = LutYawController(LUT_META["mode"], None, base.lut, base.wd_grid, base.n_turb,
                                wd_nominal=float(wd_nominal))
        return LutEvalAgent(ctrl, yaw_min, yaw_max)
    raise ValueError(f"--baseline_eval_policy must be 'lut3' or 'greedy', got {policy!r}")


def run_baseline_eval(policy: str, args, evaluators: Sequence, eval_specs: Sequence,
                      *, wind_turbine, config: Dict[str, Any], metrics_log) -> Dict[str, float]:
    """Run every evaluator once with the baseline agent and log ONE 'baseline' record.

    Mirrors transformer_sac_windfarm.run_all_evaluations' naming
    (``eval/wd/<spec>/...``) so the worker's score code reads trial and
    baseline records the same way. The LUT is rebuilt per spec at the spec's
    ws and the spec's wd_fn(0) as the nominal direction.
    """
    from helpers.layouts import get_layout_positions
    from helpers.wd_functions import get_wd_function
    if len(evaluators) != len(eval_specs):
        raise ValueError("one evaluator per eval spec expected")
    yaw_min, yaw_max = float(config["farm"]["yaw_min"]), float(config["farm"]["yaw_max"])
    ti = float(config["wind"]["TI_min"])
    metrics: Dict[str, float] = {}
    table_cache: Dict[Any, Any] = {}
    for ev, (wd_name, ws, spec_name) in zip(evaluators, eval_specs):
        if len(ev.eval_layout_names) != 1:
            raise ValueError("baseline eval needs exactly one eval layout "
                             f"(got {ev.eval_layout_names})")
        x, y = get_layout_positions(ev.eval_layout_names[0], wind_turbine)
        wd0 = float(get_wd_function(wd_name)(0.0))
        ev.agent = make_baseline_agent(policy, wind_turbine=wind_turbine, x=x, y=y, ws=ws, ti=ti,
                                       yaw_min=yaw_min, yaw_max=yaw_max, wd_nominal=wd0,
                                       table_cache=table_cache)
        m = ev.evaluate()
        metrics.update(m.to_dict(prefix=f"eval/wd/{spec_name}"))
        print(f"  [baseline {policy} / {spec_name}] mean power {m.mean_power:.1f} W, "
              f"baseline {m.mean_baseline_power:.1f} W, yaw travel {m.mean_yaw_travel:.3f} deg/step")
    meta = {"policy": policy, "ws_eval": [float(ws) for _, ws, _ in eval_specs], "ti": ti,
            "yaw_min": yaw_min, "yaw_max": yaw_max, "n_eval_steps": int(args.num_eval_steps),
            "num_eval_episodes": int(args.num_eval_episodes), "eval_seed": int(args.eval_seed)}
    if policy == "lut3":
        meta["lut"] = dict(LUT_META)
    metrics_log.write("baseline", **meta, **metrics)
    return metrics
