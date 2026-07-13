"""Farm power-reference schedule + greedy probe for power-tracking training.

Ported from ``windgym/examples/Example 7 Power tracking RL setup.ipynb``
(cells 2a76cc2e / 49480083). WindFarmEnv's ``power_ref_function`` is a
``callable(t_seconds, env) -> float`` evaluated every ``env.delay`` seconds and
returning the farm power reference in watts.

Picklability for ``AsyncVectorEnv``: the env stores its reference callable, so
that callable must survive being pickled to worker processes. We keep
``stepwise_power_ref`` a MODULE-LEVEL function and bind its ``greedy``/``schedule``
args via ``functools.partial`` (a float + a list of tuples) -- both picklable.
A closure or lambda over the training scope would not be.
"""

import numpy as np

# (n_steps, fraction_of_greedy) per segment. 4 segments x 200 steps = 800-step
# episode. In DEFAULT_SCHEDULE all fractions are <= 1.0 so a derate-only agent
# can reach every target (greedy is the full-power, no-steering ceiling for a
# derate-only farm).
DEFAULT_SCHEDULE = [(200, 0.80), (200, 0.60), (200, 0.70), (200, 1.00)]

# Middle segment is 115% of greedy: reachable ONLY via yaw wake steering
# (greedy is the no-steering, full-power baseline). Derating alone cannot exceed
# it, so this schedule REQUIRES a yaw+derate agent (config power_tracking_yaw).
BOOST_SCHEDULE = [(200, 0.80), (200, 1.15), (200, 0.70), (200, 1.00)]


def stepwise_power_ref(t, env, *, greedy, schedule=DEFAULT_SCHEDULE):
    """Step-wise farm power reference in watts at episode time ``t`` seconds.

    ``env.delay`` is the seconds-per-agent-step, so ``step = round(t / delay)``
    is the current agent step. The reference holds the last segment's fraction
    for any step past the schedule horizon.
    """
    step = int(round(t / env.delay))
    bounds = np.cumsum([n for n, _ in schedule])  # e.g. [200, 400, 600, 800]
    fracs = [f for _, f in schedule]
    idx = min(int(np.searchsorted(bounds, step, side="right")), len(fracs) - 1)
    return fracs[idx] * greedy


def measure_greedy(make_probe_env, *, n_steps=100, burn_in=20) -> float:
    """Measure greedy (undereated) waked farm power [W] for the fixed inflow.

    Builds a throwaway env via ``make_probe_env()`` (a zero-arg callable
    returning a WindFarmEnv or a wrapped env), resets it, then steps it at the
    MINIMUM-DERATE action (``-1`` -> ``derate=0`` -> full power) and averages the
    settled farm power. Valid only because the inflow is fixed in the
    power-tracking config; if wind is ever randomized, greedy must be recomputed
    per episode instead.

    Why a time-average and not one snapshot: on a steady backend (pywake) the
    flow is constant, so one sample and the mean are identical -- this is a
    no-op there. On a dynamic backend (dynamiks/DWM) the flow is turbulent and
    wake-meandering, so a single post-reset snapshot is a NOISY draw that can sit
    ~10-20% off the settled mean (and swing that much between seeds), which would
    bias the whole fraction-of-greedy reference schedule low. The tracking reward
    scores a windowed MEAN of farm power (``farm_pow_deq``), so the reference must
    be a mean too; we read the same per-step quantity here (``farm_pow_deq[-1]``,
    which also survives the flow cleanup on a late truncation).
    """
    env = make_probe_env()
    env.reset(seed=0)
    u = env.unwrapped

    # -1 across the derate action -> derate=0 -> full power (see WindFarmEnv's
    # affine [-1,1] -> [derate_min, derate_max] map, derate_min=0). The probe env
    # is built YAW-DISABLED (derate-only) even when training uses yaw+derate, so
    # this all-min action is unambiguously the no-steering, full-power baseline --
    # the correct greedy ceiling for the >100% boost schedule. (See the probe-env
    # construction in transformer_sac_windfarm_tracking.py.)
    action = -np.ones(env.action_space.shape, dtype=np.float32)
    samples = []
    for _ in range(n_steps):
        _, _, terminated, truncated, _ = env.step(action)
        samples.append(float(u.farm_pow_deq[-1]))
        if terminated or truncated:
            break  # ws-derived time_max may be short; average what we have
    env.close()

    # Drop the settling prefix, but never drop so much we average <25% of the
    # trajectory (guards the short-episode / early-truncation case).
    drop = min(burn_in, len(samples) // 4)
    settled = samples[drop:] if len(samples) > drop else samples
    return float(np.mean(settled))
