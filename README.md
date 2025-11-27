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