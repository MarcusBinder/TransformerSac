# Transformer-Based SAC for Wind Farm Yaw Control

A transformer-based Soft Actor-Critic (SAC) agent that learns generalizable yaw control policies for wind farms. Trained on diverse farm layouts, it can deploy zero-shot to unseen configurations.

## Key Features

- **Zero-shot generalization** — Train on small farms (3-6 turbines), deploy on larger ones (20-25 turbines) without retraining
- **Layout transfer** — Learn from grid layouts, transfer to irregular and circular arrangements
- **Interpretable attention** — Attention weights reveal which turbines influence each control decision
- **Wind-relative encoding** — Positions are transformed to a canonical wind frame, making the policy invariant to absolute wind direction
- **Modular architecture** — Pluggable positional encodings (MLP, sinusoidal, polar, ALiBi, relative bias) and profile encoders (Fourier, CNN, dilated, attention-based)


## Quick Start

**Training:**
```bash
python transformer_sac_windfarm.py \
    --train_layouts T1 T2 T3 T4 T5 T6 \
    --eval_layouts E1 E2 E3 \
    --total_timesteps 500000 \
    --pos_encoding_type absolute_mlp \
    --use_profiles \
    --seed 1
```

**Evaluation:**
```bash
python evaluate.py \
    --checkpoint runs/<run_name>/checkpoints/step_500000.pt \
    --eval_layouts E1 E2 E3 E4 E5
```

## Project Structure

| Path | Description |
|------|-------------|
| `transformer_sac_windfarm.py` | Main training script (SAC + transformer architecture) |
| `agent.py` | `WindFarmAgent` — wraps the actor for inference |
| `evaluate.py` | Evaluation pipeline |
| `eval_utils.py` | Evaluation helper functions |
| `helper_funcs.py` | Checkpoint I/O, coordinate transforms, env utilities |
| `MultiLayoutEnv.py` | Multi-layout environment for training across farm configurations |
| `positional_encodings/` | Positional encoding modules (absolute, bias, GAT, neighborhood) |
| `profile_encodings/` | Wake profile encoders (Fourier, CNN, dilated, attention) |
| `receptivity_profiles.py` | Compute turbine receptivity/influence profiles via PyWake |
| `geometric_profiles.py` | Geometry-based profile approximations |
| `pretrain.py` | Behavioral cloning pretraining |
| `extract_attention.py` | Extract attention weights for analysis |
| `Notebooks/` | Plotting and analysis notebooks (WES paper figures) |
| `archive/` | Historical development code (old iterations, experiments) |

## Approach

Each turbine is treated as a **token** in a transformer sequence:

1. **Per-turbine tokenization** — Local observations (wind speed, direction, yaw) become token features
2. **Wind-relative positional encoding** — Turbine positions are rotated so wind always comes from a canonical direction, then encoded via MLP (or other schemes)
3. **Wake profile conditioning** — Optional Fourier-encoded receptivity/influence profiles provide layout-aware context
4. **Permutation-equivariant output** — Shared actor/critic heads produce actions for all turbines simultaneously

The transformer naturally handles variable-length sequences, enabling a single policy to control farms of different sizes.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
