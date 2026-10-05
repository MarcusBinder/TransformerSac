"""in_training_eval_enabled(args): the single gate that decides whether the
trainer builds its resident PolicyEvaluator AsyncVectorEnvs at all.

Why a gate exists: every evaluator forks num_envs workers at startup and keeps
them for the whole run (fork-safety, see transformer_sac_windfarm.py). With
in-training eval off (eval_interval > total_timesteps, the LUMI Stage-10
setting) those 3 x 30 workers are pure memory (~170 GB per trainer, the reason
only two trainers fit a 480 GB node). The gate must be False exactly when no
evaluation can ever run: no initial eval and the first periodic eval lies past
the end of training.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Args  # noqa: E402
import transformer_sac_windfarm as tsw  # noqa: E402


def _args(**kw):
    a = Args()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_off_when_first_periodic_eval_is_past_training_end():
    a = _args(eval_initial=False, eval_interval=1_000_000_000, total_timesteps=50_000)
    assert tsw.in_training_eval_enabled(a) is False


def test_on_when_periodic_eval_fits_in_training():
    a = _args(eval_initial=False, eval_interval=50_000, total_timesteps=300_000)
    assert tsw.in_training_eval_enabled(a) is True


def test_on_at_the_boundary_interval_equals_total():
    # global_step >= next_eval_step fires at the last step, so == counts as on.
    a = _args(eval_initial=False, eval_interval=300_000, total_timesteps=300_000)
    assert tsw.in_training_eval_enabled(a) is True


def test_eval_initial_forces_on_even_with_huge_interval():
    a = _args(eval_initial=True, eval_interval=1_000_000_000, total_timesteps=50_000)
    assert tsw.in_training_eval_enabled(a) is True
