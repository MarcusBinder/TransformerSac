"""Run one tracking checkpoint through the powertracking evaluation runner.

The powertracking project (the power tracking paper repo) evaluates every
controller with its own runner, ``powertrack.eval``: one episode on the
tracking env with a greedy shadow farm, saved as a netCDF of full time series.
That repo never imports TransformerSac (script-style, generic top-level module
names), so the bridge lives here: this script loads a checkpoint, wraps the
actor as a ``powertrack.eval`` controller (``reset(facts)``, ``act(obs, info)``,
normalized flat env action out) and calls ``powertrack.eval.config.evaluate``
in-process -- the same import direction as ``--power_ref_factory``. powertrack
launches it as a subprocess (``python -m powertrack.agents.transformersac
evaluate``).

The policy rebuilds the checkpoint's actor and per-turbine observation
transform exactly as ``eval_tracking_figs.py`` does (it reuses that script's
builders), so the flat observation of the powertrack env is turned into the
token layout the actor was trained on. The env itself comes from powertrack;
``reset`` refuses one whose layout, turbine count or yaw rate differs from the
checkpoint's training env, and ``act`` refuses an observation of another width.

Usage (normally through powertrack):
    python eval_powertrack.py --checkpoint runs/<run>/checkpoints/step_N.pt \
        --eval-config <powertracking>/configs/eval/<cfg>.yaml --seed 1 --name <controller>
"""

import argparse
import sys

import numpy as np
import torch

from eval_tracking_figs import (
    TRACK3_WD,
    build_agent,
    build_obs_transform,
    build_track3_env,
    compute_track3_profiles,
    load_checkpoint,
)


class TrackingPolicy:
    """A tracking checkpoint as a ``powertrack.eval`` controller (deterministic by default)."""

    def __init__(self, checkpoint_path: str, deterministic: bool = True):
        device = torch.device("cpu")
        checkpoint, args = load_checkpoint(checkpoint_path, device)
        obs_width = checkpoint["actor_state_dict"]["obs_encoder.0.weight"].shape[1]
        # The training env, rebuilt only for its observation layout and wrapper
        # stack; it is never stepped (the rollout runs in powertrack's env).
        base_env, wrapped_env, rotor_diameter, _ = build_track3_env(
            args, "./boxes/", obs_width=obs_width
        )
        actor, agent = build_agent(args, wrapped_env, rotor_diameter, device)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()

        self.agent = agent
        self.deterministic = deterministic
        self.transform = build_obs_transform(base_env, wrapped_env)
        self.obs_shape = tuple(base_env.observation_space.shape)
        self.n_turb = int(base_env.n_turb)
        self.x_pos = np.asarray(base_env.x_pos, dtype=float)
        self.y_pos = np.asarray(base_env.y_pos, dtype=float)
        self.yaw_step = float(args["yaw_step"])
        self.act_var = int(getattr(base_env, "act_var", 1))
        self.raw_positions = np.stack([self.x_pos, self.y_pos]).T[None].astype(np.float32)
        self.wind_dirs = np.array([TRACK3_WD], dtype=np.float32)
        self.masks = np.zeros((1, self.n_turb), dtype=bool)
        recep, infl = compute_track3_profiles(args, self.x_pos, self.y_pos, base_env.turbine)
        self.receptivity = None if recep is None else np.asarray(recep, np.float32)[None]
        self.influence = None if infl is None else np.asarray(infl, np.float32)[None]
        self.spec = {
            "checkpoint": str(checkpoint_path),
            "config": args["config"],
            "action_type": args["action_type"],
            "yaw_step": self.yaw_step,
            "power_ref_factory": args.get("power_ref_factory"),
            "power_ref_kwargs": args.get("power_ref_kwargs"),
            "deterministic": deterministic,
        }

    def reset(self, facts) -> None:
        """Refuse an env that differs from the checkpoint's training env."""
        if facts.n_turb != self.n_turb or not (
            np.allclose(facts.x_pos, self.x_pos) and np.allclose(facts.y_pos, self.y_pos)
        ):
            raise ValueError(
                f"layout differs from training: {facts.x_pos}, {facts.y_pos} "
                f"vs {self.x_pos}, {self.y_pos}"
            )
        if facts.yaw_step_sim != self.yaw_step:
            raise ValueError(
                f"yaw rate differs from training: {facts.yaw_step_sim} vs {self.yaw_step} "
                "deg per sim step (set yaw_step_sim in the eval config)"
            )

    def act(self, obs, info) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != self.obs_shape:
            raise ValueError(f"observation shape {obs.shape}, trained on {self.obs_shape}")
        per_turb = self.transform(obs)[None].astype(np.float32)  # (1, n_turb, obs_dim)
        with torch.no_grad():
            a = self.agent.act(
                None,
                per_turb,
                deterministic=self.deterministic,
                wind_dirs=self.wind_dirs,
                raw_positions=self.raw_positions,
                masks=self.masks,
                receptivity=self.receptivity,
                influence=self.influence,
            )
        a = np.asarray(a)
        # turbine-major (1, n_turb, act_dim) -> the env's variable-major
        # [yaw_0..yaw_n | derate_0..derate_n] (as _TransformerEvalModel does)
        if a.ndim == 3:
            return a[0].T.reshape(-1)
        return a.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-config", required=True, help="powertracking configs/eval/*.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", required=True, help="controller name in the output path")
    parser.add_argument("--out", default=None, help="eval output root (powertrack default)")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace an existing result")
    cli = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    from powertrack.eval.config import evaluate

    policy = TrackingPolicy(cli.checkpoint, deterministic=not cli.stochastic)
    ds, path = evaluate(
        cli.eval_config,
        cli.seed,
        controller=cli.name,
        instance=policy,
        instance_spec=policy.spec,
        out_dir=cli.out,
        force=cli.force,
    )
    print(f"[eval_powertrack] wrote {path}")


if __name__ == "__main__":
    main()
