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

# (n_steps, fraction_of_greedy) per segment; all <= 1.0 so a derate-only agent
# can reach every target. 4 segments x 100 steps = 400-step episode.
DEFAULT_SCHEDULE = [(100, 0.80), (100, 0.60), (100, 0.70), (100, 1.00)]


def stepwise_power_ref(t, env, *, greedy, schedule=DEFAULT_SCHEDULE):
    """Step-wise farm power reference in watts at episode time ``t`` seconds.

    ``env.delay`` is the seconds-per-agent-step, so ``step = round(t / delay)``
    is the current agent step. The reference holds the last segment's fraction
    for any step past the schedule horizon.
    """
    step = int(round(t / env.delay))
    bounds = np.cumsum([n for n, _ in schedule])  # e.g. [100, 200, 300, 400]
    fracs = [f for _, f in schedule]
    idx = min(int(np.searchsorted(bounds, step, side="right")), len(fracs) - 1)
    return fracs[idx] * greedy


def measure_greedy(make_probe_env) -> float:
    """Measure greedy (undereated) waked farm power [W] for the fixed inflow.

    Builds a throwaway env via ``make_probe_env()`` (a zero-arg callable
    returning a WindFarmEnv or a wrapped env), resets it, sums the per-turbine
    power from the settled flow, and closes it. Valid only because the inflow is
    fixed in the power-tracking config; if wind is ever randomized, greedy must
    be recomputed per episode instead.
    """
    env = make_probe_env()
    env.reset(seed=0)
    greedy = float(env.unwrapped.fs.windTurbines.power().sum())
    env.close()
    return greedy
