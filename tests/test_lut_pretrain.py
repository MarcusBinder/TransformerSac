"""LUT pretraining (lutpre1, 2026-10-30): lut3 teacher in the warm-up, BC, critic warm-up, RL.

helpers/lut_pretrain.py: the teacher wrapper (clean lut3 label + AR(1)-noisy exploration action per
turbine), the MultiLayoutEnv pass-through (token order + padding), the replay-buffer label column and
the BC fit. A synthetic table keeps the tests off the PyWake optimiser (as test_baseline_agents.py).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.env_configs import make_env_config  # noqa: E402
from helpers.multi_layout_env import LayoutConfig, MultiLayoutEnv  # noqa: E402
from helpers.pywake_lut import LutYawController  # noqa: E402

X_POS = [0.0, 8 * 178.3]
Y_POS = [0.0, 0.0]
MAX_TURBINES = 3  # one padding slot


@pytest.fixture(scope="module")
def wind_turbine():
    from py_wake.examples.data.dtu10mw import DTU10MW
    return DTU10MW()


def _ctrl():
    wd_grid = np.arange(225.0, 316.0, 5.0)
    # turbine 0 steers 20 deg + (wd - 270) / 2, turbine 1 steers -10 deg
    lut = np.stack([20.0 + (wd_grid - 270.0) / 2.0, np.full_like(wd_grid, -10.0)], axis=1)
    return LutYawController(3, lut, lut, wd_grid, 2, wd_nominal=270.0)


def _config(action_method):
    cfg = make_env_config("les_recipe")
    cfg["ActionMethod"] = action_method
    cfg["yaw_init"] = "Zeros"
    for p in ("ws", "wd", "yaw", "power"):
        cfg[f"{p}_mes"][f"{p}_history_N"] = 3
        cfg[f"{p}_mes"][f"{p}_history_length"] = 3
        cfg[f"{p}_mes"][f"{p}_window_length"] = 1
    return cfg


def _windfarm_env(wind_turbine, action_method, x_pos=X_POS, y_pos=Y_POS):
    from WindGym import WindFarmEnv
    return WindFarmEnv(x_pos=np.asarray(x_pos), y_pos=np.asarray(y_pos), turbine=wind_turbine,
                       config=_config(action_method), backend="pywake", turbtype="Random",
                       TurbBox="Default", dt_sim=10, dt_env=10, yaw_step_sim=2.0, n_passthrough=5,
                       reset_init=False, wd_est_tau=30.0, prior_wd_est_tau=60.0,
                       prior_wd_est_consensus="front", prior_wd_est_deadband=0.0,
                       yaw_frame_source="est")


# --------------------------------------------------------------------------- label maths
def test_teacher_action_from_target_matches_the_hand_formula():
    from helpers.lut_pretrain import teacher_action_from_target
    tgt = np.array([20.0, -50.0, 3.0])
    # wind: (clip(t) - ymin) / (ymax - ymin) * 2 - 1 with band [-45, 45]
    got = teacher_action_from_target(tgt, "wind", yaw_min=-45.0, yaw_max=45.0)
    np.testing.assert_allclose(got, [20.0 / 45.0, -1.0, 3.0 / 45.0], atol=1e-7)
    # yaw: clip((clip(t) - cmd) / yaw_step_env, -1, 1)
    cmd = np.array([18.0, -40.0, 10.0])
    got = teacher_action_from_target(tgt, "yaw", yaw_min=-45.0, yaw_max=45.0, yaw_cmd=cmd,
                                     yaw_step_env=5.0)
    np.testing.assert_allclose(got, [0.4, -1.0, -1.0], atol=1e-7)
    with pytest.raises(ValueError, match="ActionMethod"):
        teacher_action_from_target(tgt, "absolute", yaw_min=-45.0, yaw_max=45.0)


# --------------------------------------------------------------------------- AR(1) noise
def test_ar1_noise_has_the_stationary_sigma_and_correlation_and_resets():
    from helpers.lut_pretrain import Ar1Noise
    n = Ar1Noise(n=4, sigma=3.0, tau_s=60.0, dt_s=10.0, seed=5)
    xs = []
    for _ in range(20):          # 20 episodes x 2000 steps
        n.reset()
        for _ in range(2000):
            xs.append(n.step().copy())
    xs = np.asarray(xs)
    assert abs(xs.std() - 3.0) < 0.1
    lag1 = np.mean([np.corrcoef(xs[:-1, i], xs[1:, i])[0, 1] for i in range(4)])
    assert abs(lag1 - np.exp(-10.0 / 60.0)) < 0.02
    # reset draws a fresh stationary state; the same seed reproduces the stream
    a, b = Ar1Noise(2, 3.0, 60.0, 10.0, seed=1), Ar1Noise(2, 3.0, 60.0, 10.0, seed=1)
    a.reset(); b.reset()
    np.testing.assert_array_equal(a.value, b.value)
    v0 = a.value.copy()
    a.reset()
    assert not np.allclose(a.value, v0)
    off = Ar1Noise(2, 0.0, 60.0, 10.0, seed=1)
    off.reset()
    assert np.all(off.step() == 0.0)


# --------------------------------------------------------------------------- teacher wrapper
@pytest.mark.parametrize("action_method", ["wind", "yaw"])
def test_teacher_wrapper_labels_are_the_lut3_target_and_drive_the_env_there(wind_turbine, action_method):
    from WindGym.core.yaw_frame import wind_action_from_target
    from helpers.action_prior import lut_frame
    from helpers.lut_pretrain import LutTeacherWrapper
    env = LutTeacherWrapper(_windfarm_env(wind_turbine, action_method), _ctrl(), noise_deg=0.0,
                            noise_tau_s=60.0)
    try:
        env.reset(seed=0)
        base = env.unwrapped
        label, explore = env.teacher_actions()
        np.testing.assert_array_equal(label, explore)          # sigma 0: explore == label
        want_tgt = np.clip(_ctrl().target_yaw(float(base.wd_est_prior), wd_frame=lut_frame(base)),
                           base.yaw_min, base.yaw_max)
        if action_method == "wind":
            np.testing.assert_allclose(label, wind_action_from_target(want_tgt, -45.0, 45.0), atol=1e-6)
        else:
            cmd = np.asarray(base.yaw_command, float) + base.frame_offset
            np.testing.assert_allclose(label, np.clip((want_tgt - cmd) / base.yaw_step_env, -1, 1),
                                       atol=1e-6)
        for _ in range(20):                                     # 2 deg/sim step: 10 steps reach 20
            _, explore = env.teacher_actions()
            *_, term, trunc, _ = env.step(explore)
            assert not (term or trunc)
        tgt = np.clip(_ctrl().target_yaw(float(base.wd_est_prior), wd_frame=lut_frame(base)),
                      base.yaw_min, base.yaw_max)
        np.testing.assert_allclose(np.asarray(base.yaw_command) + base.frame_offset, tgt, atol=0.05)
    finally:
        env.close()


def test_teacher_wrapper_noise_moves_the_target_and_resets_per_episode(wind_turbine):
    from helpers.lut_pretrain import LutTeacherWrapper
    env = LutTeacherWrapper(_windfarm_env(wind_turbine, "wind"), _ctrl(), noise_deg=3.0, noise_tau_s=60.0)
    try:
        env.reset(seed=0)
        label, explore = env.teacher_actions()
        # wind action = target / 45: the exploration offset is the AR(1) state in degrees
        np.testing.assert_allclose((explore - label) * 45.0, env.noise.value, atol=1e-4)
        v0 = env.noise.value.copy()
        env.step(explore)
        assert not np.allclose(env.noise.value, v0)             # advanced by the step
        env.reset(seed=1)
        env2 = LutTeacherWrapper(_windfarm_env(wind_turbine, "wind"), _ctrl(), noise_deg=3.0,
                                 noise_tau_s=60.0)
        env2.reset(seed=1)
        np.testing.assert_array_equal(env.noise.value, env2.noise.value)   # seeded by the reset
        env2.close()
    finally:
        env.close()


def test_teacher_wrapper_refusals(wind_turbine):
    from WindGym import WindFarmEnv
    from helpers.lut_pretrain import LutTeacherWrapper
    kw = dict(x_pos=np.asarray(X_POS), y_pos=np.asarray(Y_POS), turbine=wind_turbine,
              config=_config("wind"), backend="pywake", turbtype="Random", TurbBox="Default",
              dt_sim=10, dt_env=10, yaw_step_sim=0.5, n_passthrough=1, reset_init=False)
    env = WindFarmEnv(wd_est_tau=30.0, **kw)                    # no prior estimator
    with pytest.raises(ValueError, match="prior_wd_est_tau"):
        LutTeacherWrapper(env, _ctrl(), noise_deg=3.0, noise_tau_s=60.0)
    env.close()
    env = WindFarmEnv(prior_wd_est_tau=60.0, **kw)              # no agent estimator
    with pytest.raises(ValueError, match="wd_est"):
        LutTeacherWrapper(env, _ctrl(), noise_deg=3.0, noise_tau_s=60.0)
    env.close()
    env = WindFarmEnv(wd_est_tau=30.0, prior_wd_est_tau=60.0, **kw)
    with pytest.raises(ValueError, match="deadband"):
        LutTeacherWrapper(env, LutYawController(3, _ctrl().lut, _ctrl().lut, _ctrl().wd_grid, 2,
                                                deadband_deg=1.0), noise_deg=3.0, noise_tau_s=60.0)
    with pytest.raises(ValueError, match="noise"):
        LutTeacherWrapper(env, _ctrl(), noise_deg=-1.0, noise_tau_s=60.0)
    env.close()


# --------------------------------------------------------------------------- MultiLayoutEnv
@pytest.mark.parametrize("action_method", ["wind", "yaw"])
def test_multilayout_teacher_actions_are_token_ordered_padded_and_round_trip(wind_turbine, action_method):
    from WindGym.wrappers import PerTurbineObservationWrapper
    from helpers.action_prior import lut_frame
    from helpers.lut_pretrain import LutTeacherWrapper

    def factory(x_pos, y_pos):
        return LutTeacherWrapper(_windfarm_env(wind_turbine, action_method, x_pos, y_pos), _ctrl(),
                                 noise_deg=0.0, noise_tau_s=60.0)

    env = MultiLayoutEnv(layouts=[LayoutConfig(name="two", x_pos=np.array(X_POS), y_pos=np.array(Y_POS))],
                         env_factory=factory, per_turbine_wrapper=PerTurbineObservationWrapper,
                         seed=3, shuffle=True, max_turbines=MAX_TURBINES, max_episode_steps=200)
    try:
        for s in range(8):                                      # find a reset that swaps the pair
            env.reset(seed=s)
            if env.is_shuffled:
                break
        assert env.is_shuffled
        inner = env._current_env
        while not getattr(inner, "lut_teacher", False):
            inner = inner.env
        phys_label, _ = inner.teacher_actions()
        label, explore = env.teacher_actions()
        assert label.shape == (MAX_TURBINES,) and explore.shape == (MAX_TURBINES,)
        np.testing.assert_allclose(label[:2], phys_label[env._perm])
        assert label[2] == 0.0 and explore[2] == 0.0
        # through the vector-env call path (what the trainer uses)
        venv = gym.vector.SyncVectorEnv([lambda: env])
        (lab_v, exp_v), = venv.call("teacher_actions")
        np.testing.assert_allclose(lab_v, label)
        for _ in range(30):                                     # token-order actions reach the target
            _, explore = env.teacher_actions()
            *_, term, trunc, _ = env.step(explore)
            assert not (term or trunc)
        base = env._get_base_env()
        tgt = np.clip(_ctrl().target_yaw(float(base.wd_est_prior), wd_frame=lut_frame(base)),
                      base.yaw_min, base.yaw_max)
        np.testing.assert_allclose(np.asarray(base.yaw_command) + base.frame_offset, tgt, atol=0.05)
    finally:
        env.close()


# --------------------------------------------------------------------------- replay buffer
def _rb(label_dim, cap=50):
    from replay_buffer import TransformerReplayBuffer
    return TransformerReplayBuffer(capacity=cap, device=torch.device("cpu"), rotor_diameter=100.0,
                                   max_turbines=3, obs_dim=4, action_dim=1, label_dim=label_dim)


def _add(rb, i, label):
    mask = np.array([False, False, True])
    rb.add(np.full((3, 4), i, np.float32), np.full((3, 4), i + 1, np.float32), np.full((3, 1), 0.1 * i),
           float(i), False, np.zeros((3, 2), np.float32), mask, 270.0, label=label)


def test_buffer_labels_survive_save_load_and_mark_unlabelled_rows(tmp_path):
    rb = _rb(label_dim=1)
    for i in range(6):
        _add(rb, i, np.full((3, 1), 0.5 + i, np.float32) if i < 4 else None)
    np.testing.assert_array_equal(rb.labelled_indices(), [0, 1, 2, 3])
    batch = rb.sample_at(np.array([1, 4]))
    assert torch.allclose(batch["labels"][0], torch.full((3, 1), 1.5))
    assert torch.isnan(batch["labels"][1]).all()
    assert "labels" not in rb.sample(4) or rb.sample(4)["labels"].shape == (4, 3, 1)
    path = str(tmp_path / "b.npz")
    rb.save(path, extra_meta={"action_type": "yaw", "warmup_policy": "lut"})
    rb2 = _rb(label_dim=1)
    meta = rb2.load(path)
    assert meta["warmup_policy"] == "lut"
    np.testing.assert_array_equal(rb2.labelled_indices(), [0, 1, 2, 3])
    np.testing.assert_array_equal(rb2._labels[:6], rb._labels[:6])
    with pytest.raises(ValueError, match="label_dim"):
        _rb(label_dim=0).load(path)
    plain = _rb(label_dim=0)
    _add(plain, 0, None)
    with pytest.raises(ValueError, match="label"):
        _add(plain, 1, np.zeros((3, 1), np.float32))
    assert plain.labelled_indices().size == 0


# --------------------------------------------------------------------------- BC
def test_bc_fits_a_toy_buffer_below_half_a_degree():
    from config import Args
    from networks import TransformerActor
    from helpers.lut_pretrain import run_bc
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rb = _rb(label_dim=1, cap=4000)
    mask = np.array([False, False, True])
    w = np.array([0.3, -0.2, 0.1, 0.05])
    for i in range(4000):
        o = rng.uniform(-1, 1, (3, 4)).astype(np.float32)
        lab = np.tanh(o @ w)[:, None].astype(np.float32)     # a smooth function of the token's own obs
        lab[2] = 0.0
        rb.add(o, o, lab, 0.0, False, rng.uniform(-3, 3, (3, 2)).astype(np.float32), mask, 270.0,
               label=lab)
    actor = TransformerActor(obs_dim_per_turbine=4, action_dim_per_turbine=1, embed_dim=32,
                             num_heads=2, num_layers=1, args=Args())
    stats = run_bc(actor, rb, steps=1500, lr=3e-3, batch_size=256, std=0.1, std_coef=0.1,
                   deg_per_unit=45.0, holdout_frac=0.05, device=torch.device("cpu"))
    assert stats["n_labelled"] == 4000 and stats["n_holdout"] == 200
    assert stats["rms_deg_holdout"] < 0.5, stats
    # the std head went to the BC target (pre-tanh std 0.1)
    batch = rb.sample_at(np.arange(8))
    with torch.no_grad():
        _, log_std, _ = actor(batch["observations"], batch["positions"], batch["attention_mask"])
    real = ~batch["attention_mask"]
    assert abs(float(log_std[real].mean()) - np.log(0.1)) < 0.2


def test_bc_refuses_an_unlabelled_buffer():
    from config import Args
    from networks import TransformerActor
    from helpers.lut_pretrain import run_bc
    rb = _rb(label_dim=1)
    _add(rb, 0, None)
    actor = TransformerActor(obs_dim_per_turbine=4, embed_dim=16, num_heads=2, num_layers=1, args=Args())
    with pytest.raises(ValueError, match="labelled"):
        run_bc(actor, rb, steps=10, lr=1e-3, batch_size=8, std=0.1, std_coef=0.1, deg_per_unit=45.0,
               device=torch.device("cpu"))


# --------------------------------------------------------------------------- validation
def _args(**kw):
    base = dict(initial_exploration="lut", wd_source="est", backend="dynamiks", action_prior=None,
                load_buffer=None, action_type="yaw", learning_starts=1000, bc_steps=200,
                bc_batch_size=256, bc_std=0.1, lut_noise_deg=3.0, lut_noise_tau_s=60.0,
                finetune_freeze_actor_steps=0, buffer_size=10_000, resume_checkpoint=None)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("kw,match", [
    ({"wd_source": "true"}, "wd_source"),
    ({"backend": "pywake"}, "pywake"),
    ({"action_prior": "lut3"}, "action_prior"),
    ({"load_buffer": "x.npz"}, "load_buffer"),
    ({"action_type": "absolute"}, "action_type"),
    ({"learning_starts": 0}, "learning_starts"),
    ({"bc_steps": -1}, "bc_steps"),
    ({"bc_std": 0.0}, "bc_std"),
    ({"lut_noise_deg": -1.0}, "lut_noise"),
    ({"buffer_size": 500}, "buffer_size"),
    ({"resume_checkpoint": "runs/x/checkpoints/step_60000.pt"}, "resume"),
])
def test_validate_lut_args_refusals(kw, match):
    from helpers.lut_pretrain import validate_lut_args
    with pytest.raises(ValueError, match=match):
        validate_lut_args(_args(**kw), {"farm": {}})


def test_validate_lut_args_refuses_a_derate_env_and_passes_the_recipe():
    from helpers.lut_pretrain import validate_lut_args
    with pytest.raises(ValueError, match="derate"):
        validate_lut_args(_args(), {"derate_action": True})
    validate_lut_args(_args(), {"derate_action": False})
    validate_lut_args(_args(initial_exploration="random", wd_source="true"), {})   # off: no checks


def test_teacher_point_override_and_default():
    from helpers.lut_pretrain import teacher_point
    cfg = {"wind": {"ws_min": 8.0, "ws_max": 11.0, "TI_min": 0.038, "TI_max": 0.038}}
    assert teacher_point({}, cfg) == (9.5, 0.038)
    assert teacher_point({"lut_teacher_ws": 9.192, "lut_teacher_ti": 0.0498}, cfg) == (9.192, 0.0498)


def test_deg_per_action_unit():
    from helpers.lut_pretrain import deg_per_action_unit
    assert deg_per_action_unit("wind", yaw_min=-45.0, yaw_max=45.0, yaw_step_env=5.0) == 45.0
    assert deg_per_action_unit("yaw", yaw_min=-45.0, yaw_max=45.0, yaw_step_env=5.0) == 5.0
