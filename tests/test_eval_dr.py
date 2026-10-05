"""--eval_dr: PolicyEvaluator wraps its eval envs in a seeded DWMRandomizationWrapper
(outside MultiLayoutEnv) and the token-order params match the wrapper's draw.
Needs WindGym (skipped otherwise); runs one real dynamiks reset per env."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WindGym = pytest.importorskip("WindGym")
from WindGym import WindFarmEnv  # noqa: E402
from WindGym.wrappers import PerTurbineObservationWrapper  # noqa: E402

from helpers.dr_turbine import make_turbine_sampler  # noqa: E402
from helpers.env_configs import make_env_config  # noqa: E402
from helpers.eval_utils import PolicyEvaluator  # noqa: E402
from helpers.param_cond import COND_TURB_KEYS  # noqa: E402


@pytest.fixture(scope="module")
def evaluator():
    from py_wake.examples.data.dtu10mw import DTU10MW
    turbine = DTU10MW()
    config = make_env_config("hard_2")
    config["wind"]["ws_min"] = config["wind"]["ws_max"] = 10.0
    config["wind"]["wd_min"] = config["wind"]["wd_max"] = 270.0
    config["dwm_setup"] = "les_calibrated"

    def env_factory(x, y):
        return WindFarmEnv(turbine=turbine, x_pos=x, y_pos=y, config=config,
                           backend="dynamiks", turbtype="Random", TurbBox="Default",
                           dt_sim=10, dt_env=10, reset_init=False, n_passthrough=3)

    ranges = {"yaw_exp": (-0.4, 0.6), "tau_yaw": (0.0, 20.0)}
    ev = PolicyEvaluator(
        agent=None, eval_layouts=["test_layout"], env_factory=env_factory,
        combined_wrapper=PerTurbineObservationWrapper, num_envs=2, num_eval_steps=2,
        num_eval_episodes=1, device=torch.device("cpu"), rotor_diameter=178.3,
        wind_turbine=turbine, seed=7, max_turbines=5,
        dr_sampler_factory=lambda: make_turbine_sampler(ranges, 5, per_turbine=True),
        metrics_prefix="eval_dr",
    )
    yield ev
    ev.close()


def test_dr_wrapped_eval_envs_draw_params_in_token_order(evaluator):
    envs = evaluator.eval_envs
    envs.reset(seed=7)
    thetas = envs.env.get_attr("last_theta")
    toks = envs.env.get_attr("turbine_params_tok")
    perms = envs.env.get_attr("current_permutation")
    masks = envs.env.get_attr("attention_mask")
    assert len(thetas) == 2 and all(t is not None for t in thetas)
    for th, tok, perm, mask in zip(thetas, toks, perms, masks):
        n = int((~mask).sum())
        assert set(th) >= {"yaw_exp", "tau_yaw"}
        assert np.allclose(tok[:n, COND_TURB_KEYS.index("yaw_exp")], np.asarray(th["yaw_exp"])[:n][perm[:n]])
        assert np.allclose(tok[:n, COND_TURB_KEYS.index("tau_yaw")], np.asarray(th["tau_yaw"])[:n][perm[:n]])
        assert np.all(tok[n:, 0] == 0.0)
    # the two envs are seeded differently -> different draws
    assert not np.allclose(thetas[0]["yaw_exp"], thetas[1]["yaw_exp"])
    assert evaluator.metrics_prefix == "eval_dr"


def test_evaluator_without_factory_has_no_wrapper():
    from helpers.eval_utils import PolicyEvaluator as PE
    import inspect
    sig = inspect.signature(PE.__init__)
    assert sig.parameters["dr_sampler_factory"].default is None
    assert sig.parameters["metrics_prefix"].default == "eval"
    assert sig.parameters["autoreset_mode"].default is None


def test_evaluate_runs_with_cond_agent_through_dr_wrapper(evaluator):
    """Whole eval_dr path: CondFetcher resolves e from the DR-wrapped AsyncVectorEnv
    (turbine_params_tok / farm_params get_attrs through the wrapper) for a cond actor."""
    from config import Args
    from helpers.agent import WindFarmAgent
    from helpers.param_cond import ParamNormalizer
    from networks import TransformerActor
    obs_dim = evaluator.eval_envs.single_observation_space.shape[-1]
    args = Args(cond_source="turbine", cond_latent_dim=8, turb_dr=("yaw_exp=-0.4:0.6", "tau_yaw=0:20"))
    actor = TransformerActor(obs_dim_per_turbine=obs_dim, args=args)
    norm = ParamNormalizer({"yaw_exp": (-0.4, 0.6), "tau_yaw": (0.0, 20.0)}, source="turbine")
    evaluator.agent = WindFarmAgent(actor, torch.device("cpu"), 178.3, use_wind_relative=True,
                                    cond_normalizer=norm)
    m = evaluator.evaluate()
    d = m.to_dict(prefix=evaluator.metrics_prefix)
    assert any(k.startswith("eval_dr/") for k in d) and np.isfinite(m.power_ratio)
