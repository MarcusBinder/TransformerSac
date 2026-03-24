#!/usr/bin/env python3
"""
extract_attention.py
====================
Load a trained transformer-SAC checkpoint, run it on a specified layout,
and save the raw attention data (weights, positions, actions, metadata)
to a .npz file for offline plotting.

Usage
-----
    python extract_attention.py \
        --checkpoint checkpoints/step_150000.pt \
        --layout E3 \
        --wd 273 --ws 10 \
        --seed 22 \
        --warmup 10 \
        --outdir attention_data

Produces:  attention_data/E3_wd273_step150000.npz
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# ── Project imports ──────────────────────────────────────────────────────────
from helpers.helper_funcs import (
    transform_to_wind_relative,
    rotate_profiles_tensor,
    EnhancedPerTurbineWrapper,
)
from helpers.layouts import get_layout_positions
from helpers.env_configs import make_env_config
from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
from networks import TransformerActor
from config import Args

import gymnasium as gym
from WindGym import WindFarmEnv
from WindGym.wrappers import PerTurbineObservationWrapper


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Extract attention weights from a trained checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--layout", type=str, default="E3", help="Layout name (e.g. T1, E3)")
    p.add_argument("--wd", type=float, default=273.0, help="Wind direction (deg)")
    p.add_argument("--ws", type=float, default=10.0, help="Wind speed (m/s)")
    p.add_argument("--seed", type=int, default=22, help="Random seed")
    p.add_argument("--warmup", type=int, default=10, help="Warmup steps before snapshot")
    p.add_argument("--outdir", type=str, default="attention_data", help="Output directory")
    p.add_argument("--device", type=str, default=None, help="Device (auto-detect if omitted)")
    return p.parse_args()


# ── Turbine loader ───────────────────────────────────────────────────────────

def load_wind_turbine(turbtype: str):
    if turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    else:
        raise ValueError(f"Unknown turbine type: {turbtype}")
    return WT()


# ── Profile computation ─────────────────────────────────────────────────────

def compute_profiles(ckpt_args, x_pos, y_pos, rotor_diameter, wind_turbine):
    """Return (recep, influ) arrays or (None, None) if profiles are disabled."""
    if ckpt_args.get("profile_encoding_type", None) is None:
        return None, None

    n_dirs = ckpt_args.get("n_profile_directions", 360)
    source = ckpt_args.get("profile_source", "PyWake").lower()

    if source == "geometric":
        from helpers.geometric_profiles import compute_layout_profiles_vectorized
        return compute_layout_profiles_vectorized(
            x_pos, y_pos,
            rotor_diameter=rotor_diameter,
            k_wake=0.04,
            n_directions=n_dirs,
            sigma_smooth=10.0,
            scale_factor=15.0,
        )
    else:
        from helpers.receptivity_profiles import compute_layout_profiles
        return compute_layout_profiles(
            x_pos, y_pos, wind_turbine, n_directions=n_dirs,
        )


# ── Batch builder ────────────────────────────────────────────────────────────

def prepare_batch(env, obs, rotor_diameter, n_turbines, n_turbines_max,
                  use_wind_relative, use_profiles, recep_profiles,
                  influ_profiles, n_profile_directions, rotate, device):
    """Build actor-ready tensors from a single env observation."""
    obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32, device=device)

    wd = np.float32(env.wd)
    pos_t = torch.tensor(
        (env.turbine_positions / rotor_diameter)[np.newaxis],
        dtype=torch.float32, device=device,
    )
    if use_wind_relative:
        wd_t = torch.tensor([wd], dtype=torch.float32, device=device)
        pos_t = transform_to_wind_relative(pos_t, wd_t)

    mask_t = torch.tensor(env.attention_mask[np.newaxis], dtype=torch.bool, device=device)

    rec_t, inf_t = None, None
    if use_profiles:
        padded_r = np.zeros((n_turbines_max, n_profile_directions), dtype=np.float32)
        padded_i = np.zeros((n_turbines_max, n_profile_directions), dtype=np.float32)
        padded_r[:n_turbines] = recep_profiles
        padded_i[:n_turbines] = influ_profiles
        rec_t = torch.tensor(padded_r[np.newaxis], dtype=torch.float32, device=device)
        inf_t = torch.tensor(padded_i[np.newaxis], dtype=torch.float32, device=device)
        if rotate:
            wd_t = torch.tensor([wd], dtype=torch.float32, device=device)
            rec_t = rotate_profiles_tensor(rec_t, wd_t)
            inf_t = rotate_profiles_tensor(inf_t, wd_t)

    return obs_t, pos_t, mask_t, rec_t, inf_t


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]
    if hasattr(ckpt_args, "__dict__"):
        ckpt_args = vars(ckpt_args)
    step = checkpoint["step"]

    # Turbine
    turbtype = ckpt_args.get("turbtype", "DTU10MW")
    wind_turbine = load_wind_turbine(turbtype)
    rotor_diameter = float(wind_turbine.diameter())
    print(f"Turbine: {turbtype}  (D = {rotor_diameter:.1f} m)")

    # Layout
    x_pos, y_pos = get_layout_positions(args.layout, wind_turbine)
    n_turbines = len(x_pos)
    print(f"Layout: {args.layout}  ({n_turbines} turbines)")

    # Environment config
    config = make_env_config(ckpt_args.get("config", "default"))
    config["wind"]["ws_min"] = args.ws
    config["wind"]["ws_max"] = args.ws
    config["wind"]["wd_min"] = args.wd
    config["wind"]["wd_max"] = args.wd

    history_length = ckpt_args.get("history_length", 15)
    for prefix in ("ws", "wd", "yaw", "power"):
        config[f"{prefix}_mes"][f"{prefix}_history_N"] = history_length
        config[f"{prefix}_mes"][f"{prefix}_history_length"] = history_length

    base_env_kwargs = dict(
        turbine=wind_turbine,
        n_passthrough=ckpt_args.get("max_eps", 20),
        TurbBox=os.environ.get("TURBBOX_PATH", "/work/users/manils/rl_timestep/Boxes/V80env/"),
        config=config,
        turbtype=ckpt_args.get("TI_type", "Random"),
        dt_sim=ckpt_args.get("dt_sim", 5),
        dt_env=ckpt_args.get("dt_env", 10),
        yaw_step_sim=ckpt_args.get("yaw_step", 5.0),
    )

    # Profiles
    use_profiles = ckpt_args.get("profile_encoding_type", None) is not None
    n_profile_directions = ckpt_args.get("n_profile_directions", 360)
    rotate_profiles = ckpt_args.get("rotate_profiles", True)
    recep_profiles, influ_profiles = compute_profiles(
        ckpt_args, x_pos, y_pos, rotor_diameter, wind_turbine,
    )
    print(f"Profiles: {'enabled' if use_profiles else 'disabled'}")

    # Build environment
    layout = LayoutConfig(name=args.layout, x_pos=x_pos, y_pos=y_pos)
    use_wd_deviation = ckpt_args.get("use_wd_deviation", False)
    wd_scale_range = ckpt_args.get("wd_scale_range", 90.0)

    def env_factory(xp, yp):
        env = WindFarmEnv(x_pos=xp, y_pos=yp, reset_init=False, **base_env_kwargs)
        env.action_space.seed(args.seed)
        return env

    def combined_wrapper(env):
        env = PerTurbineObservationWrapper(env)
        if use_wd_deviation:
            env = EnhancedPerTurbineWrapper(env, wd_scale_range=wd_scale_range)
        return env

    env = MultiLayoutEnv(
        layouts=[layout], env_factory=env_factory,
        per_turbine_wrapper=combined_wrapper, seed=args.seed, shuffle=False,
    )

    n_turbines_max = env.max_turbines
    obs_dim_per_turbine = env.observation_space.shape[-1]

    # Build actor
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    @dataclass
    class _MinimalArgs:
        profile_encoder_kwargs: str = ckpt_args.get("profile_encoder_kwargs", "{}")

    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=1,
        embed_dim=ckpt_args.get("embed_dim", 128),
        pos_embed_dim=ckpt_args.get("pos_embed_dim", 32),
        num_heads=ckpt_args.get("num_heads", 4),
        num_layers=ckpt_args.get("num_layers", 2),
        mlp_ratio=ckpt_args.get("mlp_ratio", 2.0),
        dropout=ckpt_args.get("dropout", 0.0),
        action_scale=action_scale,
        action_bias=action_bias,
        pos_encoding_type=ckpt_args.get("pos_encoding_type", None),
        rel_pos_hidden_dim=ckpt_args.get("rel_pos_hidden_dim", 64),
        rel_pos_per_head=ckpt_args.get("rel_pos_per_head", True),
        pos_embedding_mode=ckpt_args.get("pos_embedding_mode", "concat"),
        profile_encoding=ckpt_args.get("profile_encoding_type", None),
        profile_encoder_hidden=ckpt_args.get("profile_encoder_hidden", 128),
        n_profile_directions=n_profile_directions,
        profile_fusion_type=ckpt_args.get("profile_fusion_type", "add"),
        profile_embed_mode=ckpt_args.get("profile_embed_mode", "add"),
        args=_MinimalArgs(),
    ).to(device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    print(f"Actor loaded: {sum(p.numel() for p in actor.parameters()):,} parameters")

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_wind_relative = ckpt_args.get("use_wind_relative_pos", True)

    # Run environment
    obs, _ = env.reset(seed=args.seed)
    wind_dir = np.float32(env.wd)
    print(f"Wind direction: {wind_dir:.1f} deg")

    batch_kwargs = dict(
        env=env, rotor_diameter=rotor_diameter,
        n_turbines=n_turbines, n_turbines_max=n_turbines_max,
        use_wind_relative=use_wind_relative, use_profiles=use_profiles,
        recep_profiles=recep_profiles, influ_profiles=influ_profiles,
        n_profile_directions=n_profile_directions, rotate=rotate_profiles,
        device=device,
    )

    # Warmup with agent's own policy
    for _ in range(args.warmup):
        obs_t, pos_t, mask_t, rec_t, inf_t = prepare_batch(obs, **batch_kwargs)
        with torch.no_grad():
            action_t, _, _, _ = actor.get_action(
                obs_t, pos_t, mask_t, deterministic=True,
                recep_profile=rec_t, influence_profile=inf_t,
            )
        obs, _, _, _, _ = env.step(action_t.squeeze(-1).squeeze(0).cpu().numpy())

    # Final snapshot
    obs_t, pos_t, mask_t, rec_t, inf_t = prepare_batch(obs, **batch_kwargs)
    with torch.no_grad():
        actions, log_prob, mean_action, attn_weights = actor.get_action(
            obs_t, pos_t, mask_t, deterministic=True,
            recep_profile=rec_t, influence_profile=inf_t,
            need_weights=True,
        )

    n_layers = len(attn_weights)
    n_heads = attn_weights[0].shape[1]
    print(f"Attention: {n_layers} layer(s), {n_heads} head(s)")
    print(f"Mean yaw: {mean_action[0, :n_turbines, 0].cpu().numpy().round(2)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    # Pack attention weights: key "attn_L{l}" has shape (n_heads, T, T)
    save_dict = {}
    for l_i in range(n_layers):
        save_dict[f"attn_L{l_i}"] = attn_weights[l_i][0].cpu().numpy()  # (H, T, T)

    # Positions in wind-relative, normalised coordinates (what the model sees)
    save_dict["positions"] = pos_t[0].cpu().numpy()              # (T, 2)
    save_dict["mean_action"] = mean_action[0].cpu().numpy()      # (T, 1)
    save_dict["mask"] = mask_t[0].cpu().numpy()                  # (T,)

    # Metadata as JSON string
    metadata = dict(
        layout=args.layout,
        n_turbines=n_turbines,
        n_turbines_max=n_turbines_max,
        n_layers=n_layers,
        n_heads=n_heads,
        wind_dir=float(wind_dir),
        wind_speed=args.ws,
        rotor_diameter=rotor_diameter,
        checkpoint=os.path.basename(args.checkpoint),
        step=int(step),
        seed=args.seed,
        warmup_steps=args.warmup,
    )
    save_dict["metadata_json"] = np.array(json.dumps(metadata))

    # Filename encodes the key parameters
    stem = f"{args.layout}_wd{int(args.wd)}_step{step}"
    outpath = os.path.join(args.outdir, f"{stem}.npz")
    np.savez_compressed(outpath, **save_dict)
    print(f"\nSaved: {outpath}")
    print(f"  Keys: {list(save_dict.keys())}")


if __name__ == "__main__":
    main()