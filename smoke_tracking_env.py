"""Fast self-test for the power-tracking / derate-only env stack.

Builds the full stack exactly as transformer_sac_windfarm_tracking.py does
  surrogate turbine -> "power_tracking" config -> WindFarmEnv(power_ref_function)
  -> PerTurbineObservationWrapper -> MultiLayoutEnv
then resets and steps with random actions. No SAC / no training -- this is a
seconds-long check that the environment, config, derating turbine, greedy probe,
and the MultiLayoutEnv obs/action shapes are all wired correctly.

Run under the pixi env:  pixi run smoke-env
"""
from functools import partial

import numpy as np

from WindGym import WindFarmEnv
from WindGym.wrappers import PerTurbineObservationWrapper

from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
from helpers.derating_turbine import make_derating_dtu10mw
from helpers.power_ref import stepwise_power_ref, measure_greedy, DEFAULT_SCHEDULE
from helpers.env_configs import make_env_config


def make_config():
    config = make_env_config("power_tracking")
    config["ActionMethod"] = "wind"  # mirrors the trainer
    return config


def build_raw_env(power_ref_function=None):
    wt = make_derating_dtu10mw()
    D = wt.diameter()
    kwargs = dict(
        turbine=wt, config=make_config(), backend="pywake",
        dt_sim=1, dt_env=1, max_time_steps=800, reset_init=False,
    )
    if power_ref_function is not None:
        kwargs["power_ref_function"] = power_ref_function
    return WindFarmEnv(x_pos=np.arange(3) * 6 * D, y_pos=np.zeros(3), **kwargs)


def main():
    # --- greedy probe + reference schedule ---
    greedy = measure_greedy(build_raw_env)
    print(f"GREEDY = {greedy/1e6:.3f} MW (notebook ~8.85 MW)")
    print("reference cycle (MW):", [round(f * greedy / 1e6, 2) for _, f in DEFAULT_SCHEDULE])
    assert 8.0 < greedy / 1e6 < 9.5, greedy

    ref_fn = partial(stepwise_power_ref, greedy=greedy, schedule=DEFAULT_SCHEDULE)

    # --- full wrapped stack via MultiLayoutEnv ---
    wt = make_derating_dtu10mw()
    D = wt.diameter()
    layout = LayoutConfig(name="track3", x_pos=np.arange(3) * 6 * D, y_pos=np.zeros(3))
    mle = MultiLayoutEnv(
        layouts=[layout],
        env_factory=lambda x, y: build_raw_env(power_ref_function=ref_fn),
        per_turbine_wrapper=PerTurbineObservationWrapper,
        seed=0, max_turbines=3, max_episode_steps=800,
    )

    # obs width = 3 temporal samples each for ws/power/yaw (was 2) + setpoint + error.
    assert mle.observation_space.shape == (3, 11), mle.observation_space.shape
    assert mle.action_space.shape == (3,), mle.action_space.shape

    obs, info = mle.reset(seed=0)
    assert obs.shape == (3, 11), obs.shape
    for key in ("Power reference", "Tracking error"):
        assert key in info, f"missing info key {key}"
    print(f"reset: obs {obs.shape}, setpoint {info['Power reference']/1e6:.2f} MW "
          f"(expect ~{0.8*greedy/1e6:.2f})")

    rng = np.random.default_rng(0)
    for t in range(5):
        a = rng.uniform(-1, 1, size=(3,)).astype(np.float32)
        obs, reward, term, trunc, info = mle.step(a)
        assert obs.shape == (3, 11), obs.shape
        assert -1.5 < reward <= 1e-6, reward
        print(f"step {t}: reward={reward:+.4f}  Pref={info['Power reference']/1e6:.3f}MW "
              f"err={info['Tracking error']/1e6:+.3f}MW")
    mle.close()
    print("\nSMOKE OK: obs (3,11), action (3,), reward in (~-1,0], tracking info present.")


if __name__ == "__main__":
    main()
