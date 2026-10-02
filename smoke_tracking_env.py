"""Fast self-test for the power-tracking / derate-only env stack.

Builds the full stack exactly as transformer_sac_windfarm_tracking.py does
  surrogate turbine -> "power_tracking" config -> WindFarmEnv(power_ref_function)
  -> PerTurbineObservationWrapper -> MultiLayoutEnv
then resets and steps with random actions. No SAC / no training -- this is a
seconds-long check that the environment, config, derating turbine, greedy probe,
and the MultiLayoutEnv obs/action shapes are all wired correctly.

Run under the pixi env:  pixi run smoke-env
Turbine selection:       python smoke_tracking_env.py --turbtype DTU10MW
"""
import argparse
from functools import partial

import numpy as np

from WindGym import WindFarmEnv
from WindGym.wrappers import PerTurbineObservationWrapper

from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
from helpers.derating_turbine import make_derating_turbine
from helpers.power_ref import stepwise_power_ref, measure_greedy, DEFAULT_SCHEDULE
from helpers.env_configs import make_env_config


def make_config():
    config = make_env_config("power_tracking")
    config["ActionMethod"] = "wind"  # mirrors the trainer
    return config


def make_turbine(cli):
    return make_derating_turbine(cli.turbtype, iea34_variant=cli.iea34_variant)


def build_raw_env(cli, power_ref_function=None):
    wt = make_turbine(cli)
    D = wt.diameter()
    kwargs = dict(
        turbine=wt, config=make_config(), backend="pywake",
        dt_sim=1, dt_env=1, max_time_steps=800, reset_init=False,
    )
    if power_ref_function is not None:
        kwargs["power_ref_function"] = power_ref_function
    return WindFarmEnv(x_pos=np.arange(3) * 6 * D, y_pos=np.zeros(3), **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Tracking-env smoke test")
    parser.add_argument("--turbtype", default="IEA34",
                        choices=["IEA34", "DTU10MW"])
    parser.add_argument("--iea34-variant", default="annrpm",
                        choices=["annrpm", "minct"])
    cli = parser.parse_args()

    n_turb = 3
    wt = make_turbine(cli)
    # Rated electrical power straight from the surrogate table (ws=15 is
    # comfortably above rated for both turbines) -> turbine-agnostic bands.
    P_rated = float(np.asarray(wt.power(np.array([15.0]))).ravel()[0])
    print(f"turbine: {wt.name()}  D={wt.diameter():.1f} m  "
          f"P_rated={P_rated / 1e6:.2f} MW")

    # --- greedy probe + reference schedule ---
    greedy = measure_greedy(partial(build_raw_env, cli))
    frac = greedy / (n_turb * P_rated)
    print(f"GREEDY = {greedy/1e6:.3f} MW "
          f"({frac:.1%} of {n_turb}x{P_rated/1e6:.2f} MW rated)")
    print("reference cycle (MW):", [round(f * greedy / 1e6, 2) for _, f in DEFAULT_SCHEDULE])
    # Wide farm-level band: waked 3-row farm below rated sits well under
    # nameplate but must produce a substantial fraction of it.
    assert 0.15 * n_turb * P_rated < greedy <= 1.02 * n_turb * P_rated, greedy

    ref_fn = partial(stepwise_power_ref, greedy=greedy, schedule=DEFAULT_SCHEDULE)

    # --- full wrapped stack via MultiLayoutEnv ---
    D = wt.diameter()
    layout = LayoutConfig(name="track3", x_pos=np.arange(3) * 6 * D, y_pos=np.zeros(3))
    mle = MultiLayoutEnv(
        layouts=[layout],
        env_factory=lambda x, y: build_raw_env(cli, power_ref_function=ref_fn),
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
    print(f"\nSMOKE OK ({cli.turbtype}): obs (3,11), action (3,), "
          "reward in (~-1,0], tracking info present.")


if __name__ == "__main__":
    main()
