"""Optuna-sweep trainer flags (LESRL optuna_sweep, 2026-10-22).

Every flag added for the sweep defaults to the OLD behaviour: a trainer run that
passes none of them must build a bit-identical env config / obs width and the
same reward. These tests guard that default-equality and the wiring of each
flag (A1-A7 of the sweep plan):

  A1 --wd_min/--wd_max/--yaw_min/--yaw_max  (config overrides, eval_wd too)
  A2 --target_entropy_scale                 (multiplies the adaptive target)
  A3 --eval_force_baseline                  (Baseline_comp on the eval envs)
  A4 --metrics_jsonl                        (append-only event log)
  A5 --obs_sensors / --obs_yaw_current      (sensor gating)
  A6 --reversal_penalty                     (YawReversalPenaltyWrapper)
  A7 hold_ramp_270_305_short                (eval registry)
  A8 --eval_score_ref / --wandb_soft_fail   (live sweep score, W&B never kills a trial)
"""
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Args  # noqa: E402
from helpers.env_configs import (  # noqa: E402
    apply_config_overrides,
    make_env_config,
    obs_dim_per_turbine,
)


def parse(*argv):
    return tyro.cli(Args, args=list(argv))


def build(*argv, preset="les_recipe"):
    args = parse(*argv)
    config = make_env_config(preset)
    apply_config_overrides(config, args)
    return args, config


# ---------------------------------------------------------------------------
# A1: wind-direction band + yaw band overrides
# ---------------------------------------------------------------------------

def test_a1_flags_default_to_none():
    a = Args()
    for attr in ("wd_min", "wd_max", "yaw_min", "yaw_max"):
        assert getattr(a, attr) is None, attr


def test_a1_no_flags_leaves_les_recipe_untouched():
    _, cfg = build()
    assert cfg == make_env_config("les_recipe")


def test_a1_wd_and_yaw_overrides_reach_the_config():
    _, cfg = build("--wd_min", "235", "--wd_max", "305", "--yaw_min", "-30", "--yaw_max", "30")
    assert cfg["wind"]["wd_min"] == 235 and cfg["wind"]["wd_max"] == 305
    assert cfg["farm"]["yaw_min"] == -30 and cfg["farm"]["yaw_max"] == 30


def test_a1_apply_config_overrides_accepts_a_checkpoint_args_dict():
    """eval_wd.py carries the checkpoint args as a dict, not an Args."""
    cfg = make_env_config("les_recipe")
    apply_config_overrides(cfg, {"yaw_min": -25.0, "yaw_max": 25.0, "wd_min": None})
    assert cfg["farm"] == {"yaw_min": -25.0, "yaw_max": 25.0}
    assert cfg["wind"]["wd_min"] == 225


def test_a1_eval_wd_build_eval_config_applies_the_overrides():
    from eval_wd import build_eval_config
    args = {**vars(Args()), "config": "les_recipe", "history_length": 15,
            "yaw_min": -30.0, "yaw_max": 30.0}
    cfg = build_eval_config(args, wd0=270.0, ws=9.0)
    assert cfg["farm"] == {"yaw_min": -30.0, "yaw_max": 30.0}
    assert cfg["wind"]["wd_min"] == cfg["wind"]["wd_max"] == 270.0
    assert cfg["wind"]["ws_min"] == cfg["wind"]["ws_max"] == 9.0


def test_a1_eval_wd_build_eval_config_default_is_the_old_construction():
    from eval_wd import build_eval_config
    from helpers.env_configs import make_eval_wind_config
    args = {**vars(Args()), "config": "les_recipe", "history_length": 15}
    cfg = build_eval_config(args, wd0=270.0, ws=9.0)
    ref = make_env_config("les_recipe")
    for mes_type, prefix in (("ws_mes", "ws"), ("wd_mes", "wd"), ("yaw_mes", "yaw"),
                             ("power_mes", "power")):
        ref[mes_type][f"{prefix}_history_N"] = 15
        ref[mes_type][f"{prefix}_history_length"] = 15
    assert cfg == make_eval_wind_config(ref, 270.0, 9.0)


# ---------------------------------------------------------------------------
# A2: target entropy scale
# ---------------------------------------------------------------------------

def test_a2_target_entropy_scale_defaults_to_one_and_multiplies():
    from helpers.training_utils import compute_adaptive_target_entropy
    assert Args().target_entropy_scale == 1.0
    mask = torch.tensor([[False, False, False, True], [False, True, True, True]])
    base = compute_adaptive_target_entropy(mask, 1, "sum")
    assert torch.equal(compute_adaptive_target_entropy(mask, 1, "sum", scale=1.0), base)
    assert torch.allclose(compute_adaptive_target_entropy(mask, 1, "sum", scale=0.5), 0.5 * base)
    assert torch.allclose(compute_adaptive_target_entropy(mask, 1, "mean", scale=2.0),
                          2.0 * compute_adaptive_target_entropy(mask, 1, "mean"))


# ---------------------------------------------------------------------------
# A3: force a baseline farm on the eval envs
# ---------------------------------------------------------------------------

def test_a3_eval_force_baseline_default_adds_no_kwargs():
    from helpers.env_configs import eval_env_extra_kwargs
    assert Args().eval_force_baseline is False
    assert eval_env_extra_kwargs(Args()) == {}
    assert eval_env_extra_kwargs(parse("--eval_force_baseline")) == {"Baseline_comp": True}


# ---------------------------------------------------------------------------
# A4: metrics jsonl
# ---------------------------------------------------------------------------

def test_a4_metrics_jsonl_defaults_to_none_and_is_a_noop():
    from helpers.metrics_jsonl import MetricsJsonl
    assert Args().metrics_jsonl is None
    m = MetricsJsonl(None)
    m.write("eval", step=1, **{"eval/wd/static_270/mean_power": 1.0})  # no error, no file


def test_a4_metrics_jsonl_appends_one_json_line_per_event(tmp_path):
    from helpers.metrics_jsonl import MetricsJsonl
    p = tmp_path / "sub" / "metrics.jsonl"
    m = MetricsJsonl(str(p))
    m.write("started", pid=123, obs_dim=4)
    m.write("eval", step=20000, **{"eval/wd/static_270/mean_power": 1.5e7})
    m.write("finished", step=20000, stop_requested=False)
    lines = p.read_text().splitlines()
    assert [json.loads(l)["event"] for l in lines] == ["started", "eval", "finished"]
    rec = json.loads(lines[1])
    assert rec["step"] == 20000 and rec["eval/wd/static_270/mean_power"] == 1.5e7
    assert "time" in rec
    # numpy scalars must serialise
    m.write("eval", step=np.int64(3), value=np.float32(1.25))
    assert json.loads(p.read_text().splitlines()[-1])["value"] == 1.25


# ---------------------------------------------------------------------------
# A5: sensor gating
# ---------------------------------------------------------------------------

def test_a5_sensor_flags_default_to_the_preset():
    from helpers.env_configs import apply_sensor_overrides
    a = Args()
    assert a.obs_sensors is None and a.obs_yaw_current is False
    cfg = make_env_config("les_recipe")
    apply_sensor_overrides(cfg, a)
    assert cfg == make_env_config("les_recipe")


@pytest.mark.parametrize("sensors, expect", [
    ("ws,wd,power", (True, True, True)),
    ("wd,power", (False, True, True)),
    ("ws", (True, False, False)),
    ("", (False, False, False)),
])
def test_a5_obs_sensors_gate_the_mes_level_flags(sensors, expect):
    from helpers.env_configs import apply_sensor_overrides
    args = parse("--obs_sensors", sensors)
    cfg = make_env_config("les_recipe")
    apply_sensor_overrides(cfg, args)
    mes = cfg["mes_level"]
    assert (mes["turb_ws"], mes["turb_wd"], mes["turb_power"]) == expect
    assert mes["turb_TI"] is False  # untouched
    # obs width follows: 15 rolling means per enabled channel + yaw
    assert obs_dim_per_turbine(cfg, 15) == 15 * (1 + sum(expect))


def test_a5_obs_sensors_rejects_unknown_names():
    from helpers.env_configs import apply_sensor_overrides
    with pytest.raises(ValueError):
        apply_sensor_overrides(make_env_config("les_recipe"), parse("--obs_sensors", "ws,ti"))


def test_a5_obs_yaw_current_adds_one_column():
    from helpers.env_configs import apply_sensor_overrides
    cfg = make_env_config("les_recipe")
    apply_sensor_overrides(cfg, parse("--obs_yaw_current"))
    assert cfg["yaw_mes"]["yaw_current"] is True
    assert obs_dim_per_turbine(cfg, 15) == 4 * 15 + 1


def test_a5_sensor_overrides_accept_a_dict():
    from helpers.env_configs import apply_sensor_overrides
    cfg = make_env_config("les_recipe")
    apply_sensor_overrides(cfg, {"obs_sensors": "wd", "obs_yaw_current": True})
    assert cfg["mes_level"]["turb_ws"] is False and cfg["yaw_mes"]["yaw_current"] is True


# ---------------------------------------------------------------------------
# A6: yaw reversal penalty
# ---------------------------------------------------------------------------

class _FakeYawEnv(gym.Env):
    """Minimal env exposing what the wrapper reads: current_yaw, yaw_step_env."""
    def __init__(self, traj):
        self.traj = [np.asarray(t, dtype=float) for t in traj]
        self.i = 0
        self.yaw_step_env = 5.0
        self.current_yaw = self.traj[0].copy()
        n = len(self.traj[0])
        self.observation_space = gym.spaces.Box(-1, 1, (n,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (n,), np.float32)

    def reset(self, *, seed=None, options=None):
        self.i = 0
        self.current_yaw = self.traj[0].copy()
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        self.i += 1
        self.current_yaw = self.traj[self.i].copy()
        return np.zeros(self.observation_space.shape, np.float32), 1.0, False, False, {}


def test_a6_reversal_penalty_defaults_to_zero_and_zero_means_no_wrapper():
    from helpers.reward_variants import maybe_wrap_reversal_penalty
    assert Args().reversal_penalty == 0.0
    env = _FakeYawEnv([[0, 0]])
    assert maybe_wrap_reversal_penalty(env, 0.0) is env
    assert maybe_wrap_reversal_penalty(env, 0.1) is not env


def test_a6_reversal_penalty_charges_sign_flips_scaled_by_step_size():
    from helpers.reward_variants import YawReversalPenaltyWrapper
    # turbine 0: +5, -2.5, -2.5 (one flip of 2.5 deg); turbine 1: +5, +5, 0 (none)
    traj = [[0, 0], [5, 5], [2.5, 10], [0, 10]]
    env = YawReversalPenaltyWrapper(_FakeYawEnv(traj), coef=2.0)
    env.reset()
    _, r1, *_ = env.step(None)   # first move: no previous delta -> no penalty
    _, r2, *_ = env.step(None)   # t0 flips (+5 -> -2.5): |d|/step = 0.5, mean over 2 turbines = 0.25
    _, r3, *_ = env.step(None)   # t0 continues (-2.5 -> -2.5): no flip; t1 stops: no flip
    assert r1 == 1.0
    assert r2 == pytest.approx(1.0 - 2.0 * 0.25)
    assert r3 == 1.0
    env.reset()
    _, r1b, *_ = env.step(None)
    assert r1b == 1.0  # history cleared on reset


def test_a6_reversal_penalty_sits_before_reward_scale():
    """combined_wrapper inserts it before TransformReward, so --reward_scale
    multiplies the penalised reward: check the composition order used there."""
    from helpers.reward_variants import YawReversalPenaltyWrapper
    traj = [[0], [5], [0]]
    env = YawReversalPenaltyWrapper(_FakeYawEnv(traj), coef=1.0)
    env = gym.wrappers.TransformReward(env, lambda r: r * 10.0)
    env.reset()
    env.step(None)
    _, r2, *_ = env.step(None)
    assert r2 == pytest.approx(10.0 * (1.0 - 1.0))


# ---------------------------------------------------------------------------
# A7: eval registry
# ---------------------------------------------------------------------------

def test_a7_hold_ramp_270_305_short_mirrors_the_235_case():
    from helpers.wd_functions import WD_FUNCTIONS
    f = WD_FUNCTIONS["hold_ramp_270_305_short"]
    g = WD_FUNCTIONS["hold_ramp_270_235_short"]
    t = np.array([0.0, 299.0, 300.0, 600.0, 900.0, 2240.0])
    assert np.allclose(f(t), [270, 270, 270, 287.5, 305, 305])
    assert np.allclose(f(t) - 270.0, -(g(t) - 270.0))


# ---------------------------------------------------------------------------
# --num_eval_envs: decouple the eval vector-env size from the training one
# ---------------------------------------------------------------------------

def test_num_eval_envs_defaults_to_the_training_num_envs():
    from helpers.env_configs import eval_num_envs
    assert Args().num_eval_envs is None
    assert eval_num_envs(parse("--num_envs", "30")) == 30
    assert eval_num_envs(parse("--num_envs", "30", "--num_eval_envs", "8")) == 8
    with pytest.raises(ValueError):
        eval_num_envs(parse("--num_eval_envs", "0"))


# ---------------------------------------------------------------------------
# A8: live sweep score (--eval_score_ref) + W&B soft-fail (--wandb_soft_fail)
# ---------------------------------------------------------------------------

def _score_ref_file(tmp_path):
    ref = {"specs": ["static_270", "static_235"],
           "per_spec": {"static_270": {"greedy": 10.0, "lut3": 12.0, "greedy_agent": 10.0},
                        "static_235": {"greedy": 20.0, "lut3": 25.0, "greedy_agent": 20.0}}}
    p = tmp_path / "baselines.json"
    p.write_text(json.dumps(ref))
    return p


def test_a8_new_flags_default_to_the_old_behaviour():
    a = Args()
    assert a.eval_score_ref is None and a.wandb_soft_fail is False and a.wandb_init_json is None
    b = parse("--eval_score_ref", "x.json", "--wandb_soft_fail", "--wandb_init_json", "w.json")
    assert b.eval_score_ref == "x.json" and b.wandb_soft_fail is True and b.wandb_init_json == "w.json"


def test_a8_load_score_ref_reads_the_baselines_per_spec_layout(tmp_path):
    from helpers.eval_score import load_score_ref
    ref = load_score_ref(str(_score_ref_file(tmp_path)))
    assert ref == {"static_270": (10.0, 12.0), "static_235": (20.0, 25.0)}
    assert list(ref) == ["static_270", "static_235"]


def test_a8_eval_scores_is_the_unclamped_normalised_energy(tmp_path):
    from helpers.eval_score import eval_scores, load_score_ref
    ref = load_score_ref(str(_score_ref_file(tmp_path)))
    ev = {"eval/wd/static_270/mean_power": 13.0,      # (13-10)/(12-10) = 1.5
          "eval/wd/static_235/mean_power": 0.0,       # (0-20)/(25-20) = -4 (not clamped)
          "eval/mean_reward": 0.3}
    out = eval_scores(ev, ref)
    assert out == pytest.approx({"eval/score/static_270": 1.5, "eval/score/static_235": -4.0,
                                 "eval/score/mean": -1.25})


def test_a8_eval_scores_weighted_mean_follows_the_baselines_weights(tmp_path):
    """baselines.json["weights"] (spec -> weight, default 1.0) makes eval/score/mean
    the weighted mean, the same formula as optuna_sweep.objective.score_record."""
    from helpers.eval_score import eval_scores, load_score_ref
    ref = {"specs": ["static_270", "static_235"], "weights": {"static_235": 3.0},
           "per_spec": {"static_270": {"greedy": 10.0, "lut3": 12.0},
                        "static_235": {"greedy": 20.0, "lut3": 25.0}}}
    p = tmp_path / "baselines.json"
    p.write_text(json.dumps(ref))
    r = load_score_ref(str(p))
    assert r == {"static_270": (10.0, 12.0, 1.0), "static_235": (20.0, 25.0, 3.0)}
    ev = {"eval/wd/static_270/mean_power": 13.0,      # 1.5
          "eval/wd/static_235/mean_power": 22.5}      # 0.5
    out = eval_scores(ev, r)
    assert out == pytest.approx({"eval/score/static_270": 1.5, "eval/score/static_235": 0.5,
                                 "eval/score/mean": (1.5 * 1.0 + 0.5 * 3.0) / 4.0})
    # a 2-tuple ref (no weights key) still means weight 1
    assert eval_scores(ev, {k: v[:2] for k, v in r.items()})["eval/score/mean"] == pytest.approx(1.0)


def test_a8_eval_scores_skips_missing_specs_and_then_has_no_mean(tmp_path):
    from helpers.eval_score import eval_scores, load_score_ref
    ref = load_score_ref(str(_score_ref_file(tmp_path)))
    out = eval_scores({"eval/wd/static_270/mean_power": 11.0}, ref)
    assert out == pytest.approx({"eval/score/static_270": 0.5})
    assert eval_scores({}, None) == {}


class _FakeWandb:
    """Stand-in for the wandb module: records init calls, fails the first ``fail`` of them."""

    class Settings:
        def __init__(self, **kw):
            self.kw = kw

    def __init__(self, fail=0):
        self.fail, self.calls = fail, []

    def init(self, **kw):
        self.calls.append(kw)
        if len(self.calls) <= self.fail:
            raise RuntimeError("wandb down")
        return "RUN"


def test_a8_init_wandb_without_soft_fail_is_the_old_call(monkeypatch):
    from helpers import wandb_utils
    fake = _FakeWandb(fail=1)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    args = parse()
    with pytest.raises(RuntimeError):
        wandb_utils.init_wandb(args, soft_fail=False, project="p", name="n")
    assert fake.calls == [{"project": "p", "name": "n"}]       # no settings, no retry
    fake2 = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake2)
    assert wandb_utils.init_wandb(args, soft_fail=False, project="p") == "RUN"
    assert fake2.calls == [{"project": "p"}]


def test_a8_init_wandb_soft_fail_retries_offline(monkeypatch):
    from helpers import wandb_utils
    fake = _FakeWandb(fail=1)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    args = parse()
    assert wandb_utils.init_wandb(args, soft_fail=True, project="p", tags=["t"]) == "RUN"
    assert args.track is True
    # no explicit wandb.Settings: building one re-validates the WANDB_* env (0.21 rejected a
    # WANDB_CONFIG_PATHS string there, so every online init failed); the worker sets WANDB_INIT_TIMEOUT
    assert fake.calls[0] == {"project": "p", "tags": ["t"]}
    assert fake.calls[1] == {"project": "p", "tags": ["t"], "mode": "offline"}


def test_a8_init_wandb_soft_fail_gives_up_and_disables_tracking(monkeypatch):
    from helpers import wandb_utils
    fake = _FakeWandb(fail=2)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    args = parse()
    assert wandb_utils.init_wandb(args, soft_fail=True, project="p") is None
    assert args.track is False and len(fake.calls) == 2


def test_a8_wandb_init_json_adds_id_tags_and_config(tmp_path):
    from helpers.wandb_utils import wandb_init_kwargs
    base = {"lr": 1.0, "debug/x": 2}
    assert wandb_init_kwargs(None, base) == {"config": base}           # no flag: the old call
    p = tmp_path / "wandb_init.json"
    p.write_text(json.dumps({"id": "s-t0001-ab12", "tags": ["optuna", "s"],
                             "config": {"optuna/trial": 1, "lr": "clash"}}))
    kw = wandb_init_kwargs(str(p), base)
    assert kw["id"] == "s-t0001-ab12" and kw["tags"] == ["optuna", "s"]
    assert kw["config"] == {"lr": 1.0, "debug/x": 2, "optuna/trial": 1}  # trainer args win a clash
    assert base == {"lr": 1.0, "debug/x": 2}


def test_a8_real_wandb_offline_run_carries_the_id_tags_and_config(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    from helpers import wandb_utils
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    p = tmp_path / "wandb_init.json"
    p.write_text(json.dumps({"id": "s-t0002-cd34", "tags": ["optuna", "s", "raw"],
                             "config": {"optuna/trial": 2, "optuna/lr": 3e-4}}))
    args = parse("--wandb_soft_fail")
    run = wandb_utils.init_wandb(args, soft_fail=True, project="p", group="s", name="optuna_s_t0002",
                                 **wandb_utils.wandb_init_kwargs(str(p), {"seed": 1}))
    try:
        assert run is not None and run.id == "s-t0002-cd34" and run.group == "s"
        assert set(run.tags) == {"optuna", "s", "raw"}
        assert run.config["optuna/trial"] == 2 and run.config["seed"] == 1
    finally:
        run.finish()


def test_a8_eval_groups_chart_against_global_step():
    from helpers.wandb_utils import STEP_METRIC_GROUPS
    for g in ("perf/*", "losses/*", "charts/*", "eval/*", "eval_dr/*"):
        assert g in STEP_METRIC_GROUPS


# ---------------------------------------------------------------------------
# Regression: --obs_norm crashed every run since the async step refactor
# (obs_normalizer.update read next_obs before step_result was unpacked).
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("overlap", [False, True])
def test_obs_norm_training_run_completes(tmp_path, overlap):
    import subprocess
    metrics = tmp_path / "m.jsonl"
    argv = [sys.executable, "-u", "transformer_sac_windfarm.py", "--backend", "pywake", "--dt_sim", "10",
            "--dt_env", "10", "--layouts", "les_3x3", "--max_turbines", "9", "--turbtype", "iea22",
            "--config", "les_recipe", "--train_ws_min", "9", "--train_ws_max", "9", "--TI_type", "Random",
            "--num_envs", "2", "--num_eval_envs", "1", "--total_timesteps", "60", "--learning_starts", "20",
            "--eval_interval", "60", "--eval_wd_function", "static_270", "--eval_ws", "9", "--num_eval_steps", "5",
            "--embed_dim", "32", "--num_heads", "2", "--num_layers", "1", "--algorithm", "sac", "--batch_size", "16",
            "--buffer_size", "100", "--obs_norm", "--no-track", "--no-save_model", "--no-cuda",
            "--metrics_jsonl", str(metrics), "--exp_name", f"obsnorm_test_{int(overlap)}"]
    if overlap:
        argv.append("--async_overlap")
    r = subprocess.run(argv, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    events = [json.loads(l)["event"] for l in metrics.read_text().splitlines()]
    assert events[0] == "started" and events[-1] == "finished" and "eval" in events


# ---------------------------------------------------------------------------
# V3 (Optuna v3, 2026-10-28): --wd_est_deadband (the est-wd hold) and the
# fine-tune arm's actor freeze window.
# ---------------------------------------------------------------------------

def test_wd_est_deadband_reaches_the_env_kwargs_and_the_checkpoint_args():
    import transformer_sac_windfarm as tsw
    assert Args().wd_est_deadband == 0.0
    a = parse("--backend", "dynamiks", "--wd_source", "est", "--wd_est_tau", "60",
              "--wd_est_deadband", "1.0")
    kw = tsw.wd_source_env_kwargs(a)
    assert kw == {"wd_est_tau": 60.0, "wd_est_consensus": "median", "wd_est_deadband": 1.0}
    assert vars(a)["wd_est_deadband"] == 1.0   # save_checkpoint stores vars(args)
    # wd_source true: nothing forwarded (the old behaviour)
    assert tsw.wd_source_env_kwargs(parse("--backend", "dynamiks")) == {}


def test_wd_est_deadband_without_est_source_is_refused():
    import transformer_sac_windfarm as tsw
    a = parse("--backend", "dynamiks", "--wd_est_deadband", "1.0")
    with pytest.raises(ValueError, match="wd_est_deadband"):
        tsw.wd_source_env_kwargs(a)
    with pytest.raises(ValueError, match="wd_est_tau"):
        tsw.wd_source_env_kwargs(parse("--backend", "dynamiks", "--wd_source", "est"))


def test_eval_wd_cli_default_deadband_is_zero_and_a_namespace_without_it_still_builds():
    import types
    import eval_wd
    base = ["--checkpoint", "x.pt", "--wd_function", "static_270", "--out", "x.nc"]
    cli = eval_wd.build_parser().parse_args(base)
    assert cli.wd_est_deadband == 0.0
    cli2 = eval_wd.build_parser().parse_args(
        base + ["--wd_source", "est", "--wd_est_tau", "60", "--wd_est_deadband", "2"])
    assert eval_wd.wd_est_env_kwargs(cli2, "dynamiks") == {
        "wd_est_tau": 60.0, "wd_est_consensus": "median", "wd_est_deadband": 2.0}
    # test_obs_agg builds a SimpleNamespace without the new attribute
    old = types.SimpleNamespace(wd_source="est", wd_est_tau=30.0, wd_est_consensus="median")
    assert eval_wd.wd_est_env_kwargs(old, "dynamiks")["wd_est_deadband"] == 0.0
    assert eval_wd.wd_est_env_kwargs(types.SimpleNamespace(wd_source="true"), "pywake") == {}
    with pytest.raises(SystemExit):
        eval_wd.wd_est_env_kwargs(old, "pywake")


def test_actor_updates_are_skipped_for_the_freeze_window_after_learning_starts():
    from types import SimpleNamespace
    from helpers.training_utils import actor_update_allowed
    assert Args().finetune_freeze_actor_steps == 0
    off = SimpleNamespace(learning_starts=100, finetune_freeze_actor_steps=0)
    assert all(actor_update_allowed(s, off) for s in (0, 100, 150, 10_000))
    on = SimpleNamespace(learning_starts=100, finetune_freeze_actor_steps=50)
    assert not actor_update_allowed(100, on)
    assert not actor_update_allowed(149, on)
    assert actor_update_allowed(150, on)
    assert actor_update_allowed(10_000, on)
