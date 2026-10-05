#!/bin/bash
# Stage 10 CPU smokes (minutes each, dynamiks 3x3). Run from TransformerSac/:
#   bash tests/smoke_cond.sh <name> [extra trainer flags]
# Recipes (plan P2 verification):
#   bash tests/smoke_cond.sh smoke_rma   --cond_source turbine_farm --cond_latent_dim 8 --cond_mode concat --cond_critic raw --cond_action_hist --eval_dr --adapt_rounds 1 --adapt_round_steps 20 --adapt_warm_steps 10 --adapt_fit_steps 10
#   bash tests/smoke_cond.sh smoke_uposi --cond_source turbine_farm --cond_latent_dim 0 --cond_mode concat --cond_critic raw --cond_action_hist --eval_dr --adapt_rounds 1 --adapt_round_steps 20 --adapt_warm_steps 10 --adapt_fit_steps 10
#   bash tests/smoke_cond.sh smoke_film  --cond_source turbine_farm --cond_latent_dim 8 --cond_mode film   --cond_critic none --cond_action_hist --eval_dr --adapt_rounds 1 --adapt_round_steps 20 --adapt_warm_steps 10 --adapt_fit_steps 10 --phase2_loss latent_action --phase2_train_actor
#   bash tests/smoke_cond.sh smoke_s9                       # Stage-9 recipe unchanged (regression: step_0.pt bitwise vs main)
# Expect: "[Cond] source=..." banner (cond runs), "[eval_dr] 1 evaluator(s)", "[adapt] round 1/1", no Traceback,
# runs/<name>/checkpoints/step_<final>.pt with adapt_state_dict (cond runs).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1; shift
PY="${PYTHON:-/home/marcus/miniconda3/bin/conda run -n lesrl python}"
POST="${POSTERIOR:-/home/marcus/Documents/paper-derating/posterior_les_data_samples.npz}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" $PY -u transformer_sac_windfarm.py --no-cuda --no-track --save_model --save_interval 1000000 \
  --num_envs 2 --total_timesteps 48 --learning_starts 8 --batch_size 4 --max_episode_steps 12 \
  --eval_interval 1000000000 --TI_type Random --exp_name "$NAME" --config les_recipe_pin270 --turbtype iea22h2 \
  --backend dynamiks --action_type yaw --algorithm tqc --tqc_share_trunk --dt_sim 5 --dt_env 10 --yaw_step 2.5 \
  --history_length 3 --obs_agg raw15span --obs_agg_len 60 --action_penalty 0.05 --action_penalty_type Change \
  --train_wd_function cycle_270_235_slow_phase --dr_posterior_path "$POST" --dr-keys k1 k2 d_particle \
  --turb_dr yaw_exp=-0.4:0.6 cp_gain=0.95:1.05 ct_gain=0.90:1.10 tau_yaw=0:20 delay_yaw=0:10 tau_power=0:30 \
  --veer_min 0 --veer_max 2 --tilt 5 --utd_ratio 1 --buffer_size 3000 --max_eps 100 --shuffle_turbs \
  --layouts les_3x3 --max_turbines 9 --max_turb_move 30 --wd_source est --wd_est_tau 15 --wd_est_consensus front \
  --eval_wd_function cycle_270_235_slow_short --eval_ws 9 --num_eval_steps 2 --num_eval_episodes 1 \
  --eval_layouts les_3x3 --eval_seed 42 --seed 1 "$@"
