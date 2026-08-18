"""One factory for the load-penalty wrapper: NN DEL surrogate or proxy zoo.

Used by both trainers and interp.common.build_delmax_env so the three call
sites cannot drift. `args` is either config.Args (trainers) or the
back-filled checkpoint args dict (interp); old checkpoints back-fill
load_proxies=None and therefore keep the surrogate path.

    load_proxies unset -> del_surrogate.DELRewardWrapper(channels=del_channels)
    load_proxies set   -> proxy_zoo.ProxyRewardWrapper(proxies=load_proxies)

load_compare / load_penalty apply to both.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEL_SET_BY_TURBTYPE = {"IEA34": "iea34", "DTU10MW": "dtu10mw"}
_DEL_CHANNELS_DEFAULT = "Bl1Rad0FlpMnt"


def _get(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def _csv(value) -> list[str]:
    if value is None:
        return []
    return [c.strip() for c in str(value).split(",") if c.strip()]


def uses_proxy_reward(args) -> bool:
    return bool(_csv(_get(args, "load_proxies")))


def del_artifact_set(args) -> str:
    turbine = _get(args, "del_artifact_set")
    if turbine:
        return turbine
    turbtype = _get(args, "turbtype")
    if turbtype not in DEL_SET_BY_TURBTYPE:
        raise ValueError(
            f"No DEL surrogate set for turbtype {turbtype!r}; pass "
            "--del_artifact_set explicitly."
        )
    return DEL_SET_BY_TURBTYPE[turbtype]


def load_reward_channels(args) -> list[str]:
    """Names the penalty binds on, as the wrapper reports them in
    info["del_binding_channel"] / info["del_ratio_by_channel"]."""
    if uses_proxy_reward(args):
        return _csv(_get(args, "load_reward_proxies")) or _csv(_get(args, "load_proxies"))
    from del_surrogate import get_set
    tset = get_set(del_artifact_set(args))
    return [tset.canonical_channel(c) for c in _csv(_get(args, "del_channels"))]


def build_load_reward_wrapper(env, args, **limit_kwargs):
    """Wrap `env` (already per-turbine-observed) in the configured load
    reward wrapper. `limit_kwargs` = limit_range / limit_obs_ref /
    fixed_limit / limit_fn passthroughs (the interp path pins limits)."""
    common = dict(
        penalty_scale=float(_get(args, "del_penalty_scale", 0.0)),
        allowed_increase=float(_get(args, "del_allowed_increase", 0.10)),
        compare=_get(args, "load_compare", "farm_max") or "farm_max",
        penalty=_get(args, "load_penalty", "hinge") or "hinge",
        ti_window=float(_get(args, "del_ti_window", 60.0)),
        # Reduced rotor template for training: 3*12=36 sample points per
        # turbine per dt_sim per farm (the 288-point eval default is
        # for offline sampling).
        n_r=3,
        n_theta=12,
        **limit_kwargs,
    )
    if uses_proxy_reward(args):
        del_channels = _get(args, "del_channels")
        if del_channels not in (None, "", _DEL_CHANNELS_DEFAULT):
            raise ValueError(
                "--load_proxies replaces the DEL surrogate; an explicit "
                f"--del_channels ({del_channels!r}) has no effect and is "
                "refused to avoid silent misconfiguration."
            )
        from proxy_zoo import ProxyRewardWrapper
        proxies = _csv(_get(args, "load_proxies"))
        reward_proxies = _csv(_get(args, "load_reward_proxies")) or proxies
        return ProxyRewardWrapper(
            env, proxies=proxies, reward_proxies=reward_proxies,
            turbine=del_artifact_set(args), **common,
        )
    from del_surrogate import DELRewardWrapper
    channels = _csv(_get(args, "del_channels")) or [_DEL_CHANNELS_DEFAULT]
    return DELRewardWrapper(
        env, turbine=del_artifact_set(args), channels=channels,
        reward_channels=channels, **common,
    )
