# Transformer-Based Wind Farm Control

This project develops transformer-based reinforcement learning approaches for wind farm yaw control that can generalize across different farm configurations.

## Overview

Wind turbines in a farm create wakes that reduce power output for downstream turbines. By coordinating yaw angles across turbines, we can steer wakes and increase total farm power production. Traditional control methods require retraining for each new farm layout, limiting their practical deployment.

This work uses transformer architectures to learn control policies that:

- **Generalize across farm sizes**: Train on small farms, deploy on larger ones
- **Transfer between layouts**: Learn from square grids, apply to circular arrangements
- **Provide interpretability**: Attention weights reveal which turbines the policy considers when making decisions

## Approach

The core idea is to treat each turbine as a token in a transformer sequence:

- **Per-turbine tokenization**: Each turbine's local observations (wind speed, direction, current yaw) become a token
- **Wind-relative positional encoding**: Turbine positions are transformed relative to wind direction and normalized by rotor diameter
- **Permutation equivariance**: The policy outputs actions for all turbines simultaneously, with shared weights ensuring the same turbine in different positions receives consistent treatment

The transformer architecture naturally handles variable-length sequences, enabling a single trained policy to control farms with different numbers of turbines.

## Goal

Develop a zero-shot generalizable wind farm controller that can be trained on a diverse set of farm configurations and deployed to new, unseen layouts without retraining—reducing the engineering effort required for real-world wind farm optimization.


## Current status:
`transformer_sac_windfarm_v4` was able to train a agent, and it lookes like it was working. This was however only for a single wind condition and a single farm. 

`transformer_sac_windfarm_v5` should be able to train an agent on varrying wind farms, but it is not yet tested.  

### Future work/tasks

Priority | Issue | Type | Effort | Explanation | 
-----------------------------------------------
1 | Wind direction is fixed at 270° for positional encoding | 🔴  Bug | Medium | 
    The wind-relative positional encoding is designed to rotate turbine coordinates so wind always "comes from" a canonical direction. But the code hardcodes 270° instead of reading the actual wind direction from the observation. This means whenever the wind isn't exactly 270° (which is most of the time, given the 265-275° range), the positional encoding is wrong. The model learns with corrupted spatial information.

2 | Critic architecture: actions through attention | Design choice | Medium | 
    Currently, each turbine token is (obs, action) concatenated, and the transformer attends over all tokens. This means turbine A's representation is influenced by turbine B's action through attention. While wake physics could justify this (B's yaw affects A's inflow), it's unconventional—standard Q-networks condition on actions after state encoding, not through attention. An alternative is: encode states with the transformer, pool, then concatenate the flat action vector before the Q-head. Worth testing which works better.

3 | No attention masking (needed for multi-layout) | Feature gap | Low | 
    The attention mask infrastructure exists (attn_mask parameter flows through the network) but is never used. For single fixed-size farms this is fine. For your goal of training across variable-size farms with padding, you'll need masks to prevent the model from attending to padded "fake" turbines. Should be straightforward to add when needed.

4 | Mean pooling vs CLS token for critic | Design choice | Low | 
    The critic aggregates turbine representations by taking the mean across all tokens. An alternative is to prepend a learnable [CLS] token to the sequence; after the transformer, only the CLS token's output is used for the Q-value. CLS tokens can learn to aggregate information more flexibly than fixed mean pooling. Low priority since mean pooling is a reasonable default.

5 | Absolute vs relative positional encoding | Enhancement | Medium | 
    Currently, positions are encoded as absolute (x, y) coordinates (rotated to wind-relative frame). Relative positional encoding—where attention weights depend on the distance between turbines rather than their absolute locations—could improve transfer to unseen layouts. This is a more significant architectural change and might be worth exploring after basics are solid.

6 | No gradient clipping | Stability | Low | 
    The optimizer updates have no gradient clipping. If gradients explode during training (especially early on or with high learning rates), this can destabilize learning. Adding torch.nn.utils.clip_grad_norm_ with a max norm of 0.5-1.0 is a simple safeguard. Low priority if training is already stable.

7 | No reward normalization | Stability | Low | 
    Rewards are used raw from the environment. If reward magnitudes vary significantly (e.g., different farm sizes produce different total power), this can affect learning. Running reward normalization (tracking mean/std and normalizing) can help. Not urgent for single-layout training, but useful for multi-layout.

8 | Positional encoding could use Fourier features | Enhancement | Low | 
    The current positional encoder is a simple 2-layer MLP on (x, y). Fourier/sinusoidal features (like in NeRF or the original Transformer) can help the network represent high-frequency spatial patterns and generalize better across scales. This is a "nice to have" refinement, not critical.


Priority  | Direction                  | Novelty    | Effort | Key Benefit
-------------------------------------------------------
1         | Polar coordinate encoding  | Low        | Low    | Physical interpretability, wind-invariant representation
2         | GTrXL                      | Low        | Medium | Training stability, established RL-transformer solution
3         | Wind-relative RoPE         | Medium-High| Medium | Relative distance encoding, novel domain adaptation
4         | Hybrid GNN-Transformer     | High       | High   | Physics-informed structure, strong generalization


Recommended Implementation Order
Phase       | Task                                       | Rationale | 
Now         | Fix wind direction bug (A1)                 | Critical correctness issue
Now         | Add gradient clipping (A6)                 | Simple stability safeguard
Soon        | Implement attention masking (A3)           | Required for multi-layout training
Soon        | Polar coordinates (B1)                     | Low effort, good physical grounding
Medium-term | Test critic architecture alternatives (A2) | Could improve learning
Medium-term | GTrXL (B2)                                 | Stabilize training for harder problems
Research    | Wind-relative RoPE (B3)                    | Novel contribution potential
Research    | Hybrid GNN-Transformer (B4)                | Strongest paper contribution