### Run offline pre-training and online fine-tuning for exploratory data collection

Kitchen:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_calql/conservative_sac_main.py --exp_name=calql_rnd10_kitchen_{dataset} \
                --env=kitchen-{dataset:partial,complete,mixed}-v0 \
                --seed=0 \
                --cql_min_q_weight=5.0 \
                --cql.cql_importance_sample=False \
                --policy_arch=512-512-512 \
                --qf_arch=512-512-512 \
                --n_pretrain_epochs=500 \
                --max_online_env_steps={dataset:1e6,4e6,1e6} \
                --mixing_ratio=0.25 \
                --reward_bias=-5 \
                --logging.output_dir=./results/calql_rnd10_kitchen_{dataset}/seed_0/ \
                --cql.use_rnd=True \
                --cql.rnd_reward_scale=10 \
                --cql.bound_q_functions=True \
                --cql.max_reward=0.5
```


Antmaze-goal-missing-large-v2:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_calql/conservative_sac_main.py --exp_name=calql_rnd50_antmaze_goal_removed \
                --env=antmaze-large-diverse-v2 \
                --seed=0 \
                --cql_min_q_weight=5.0 \
                --cql.cql_target_action_gap=0.8 \
                --cql.cql_lagrange=True \
                --policy_arch=256-256 \
                --qf_arch=256-256-256-256 \
                --n_pretrain_epochs=1000 \
                --max_online_env_steps=2e6 \
                --mixing_ratio=0.5 \
                --reward_scale=10.0 \
                --reward_bias=-5 \
                --logging.output_dir=./results/calql_rnd50_antmaze_goal_removed/seed_0/ \
                --dataset_path=./datasets/antmaze-large-diverse-v2-goal-data-removed_radius_2.5/seed_0.npz \
                --cql.use_rnd=True \
                --cql.rnd_reward_scale=50.0 \
                --cql.bound_q_functions=False
```

The intrinsic reward scale is 50 instead of 10 to account for the extrinsic reward scaling.

Maze2d-missing-data-large-v1:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_calql/conservative_sac_main.py --exp_name=calql_rnd10_maze2d_dataset_goal_topleft_bottomleft_removed \
                --env maze2d-large-v1 \
                --seed=0 \
                --cql_min_q_weight=5.0 \
                --cql.cql_target_action_gap=0.8 \
                --cql.cql_lagrange=True \
                --policy_arch=256-256 \
                --qf_arch=256-256-256-256 \
                --n_pretrain_epochs=500 \
                --max_online_env_steps=1e6 \
                --mixing_ratio=0.5 \
                --reward_scale=10 \
                --reward_bias=-5 \
                --logging.output_dir=./results/calql_rnd10_maze2d_dataset_goal_topleft_bottomleft_removed/seed_0 \
                --dataset_path=./datasets/maze2d-large-v1-500k_downsample_goal_and_top_left/seed_0.npz \
                --cql.use_rnd=True \
                --cql.rnd_reward_scale=10 \
                --cql.bound_q_functions=True \
                --cql.min_reward=-5 \
                --cql.max_reward=6
```


Hammer-truncated-expert-v1:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_calql/conservative_sac_main.py --exp_name=calql_rnd0.5_hammer_truncated \
                --env=hammer-expert-v1 \
                --seed=0 \
                --cql_min_q_weight=1.0 \
                --policy_arch=512-512 \
                --qf_arch=512-512-512 \
                --n_pretrain_epochs=20 \
                --max_online_env_steps=1e6 \
                --mixing_ratio=0.5 \
                --logging.output_dir=./results/calql_rnd10_hammer_truncated/seed_0 \
                --max_len=20 \
                --cql.use_rnd=True \
                --cql.rnd_reward_scale=0.5  \
                --cql.bound_q_functions=True \
                --cql.max_reward=103
```


Adroit binary tasks (relocate, door, pen):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_calql/conservative_sac_main.py --exp_name={task:relocate,door,pen}_rnd0.5 \
                --env={task}-binary-v0 \
                --seed=0 \
                --cql_min_q_weight=1.0 \
                --policy_arch=512-512 \
                --qf_arch=512-512-512 \
                --n_pretrain_epochs=20 \
                --max_online_env_steps=1e6 \
                --mixing_ratio=0.5 \
                --reward_scale=10.0 \
                --reward_bias=5.0 \
                --logging.output_dir=./results/{task}_rnd0.5/seed_0 \
                --cql.use_rnd=True \
                --cql.rnd_reward_scale=0.5
```


### Run Offline RL for Exploitation phase

Kitchen:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_calql/conservative_sac_exploitation.py --exp_name=cql_exploitation_kitchen_{dataset:partial,complete,mixed}_timestep_{timestep:100000,250000,500000,1000000,2000000,3000000,4000000} \
                --env=kitchen-{dataset}-v0 \
                --seed=0 \
                --cql_min_q_weight=5.0 \
                --cql.cql_importance_sample=False \
                --policy_arch=512-512-512 \
                --qf_arch=512-512-512 \
                --mixing_ratio=0.25 \
                --reward_bias=-4.0 \
                --replay_buffer_original_bias=-5 \
                --cql.use_rnd=False \
                --logging.output_dir=./results/cql_exploitation_***env***/timestep_{timestep}/seed_0 \
                --exploitation_timestep={timestep} \
                --replay_buffer_path=./results/calql_rnd10_kitchen_{dataset}/seed_0/replay_buffer.npz \
                --replay_buffer_size=4000000 \
                --bound_q_functions_according_to_data=True \
                --cql.feature_normalization=True
```


Maze2d-missing-data-large-v1:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_calql/conservative_sac_exploitation.py --exp_name=cql_exploitation_maze2d_dataset_goal_topleft_bottomleft_removed_timestep_{timestep:250000,500000} \
                --env=maze2d-large-v1 \
                --seed=0 \
                --cql_min_q_weight=5.0 \
                --cql.cql_target_action_gap=0.8 \
                --cql.cql_lagrange=True \
                --policy_arch=256-256 \
                --qf_arch=256-256-256-256 \
                --mixing_ratio=-1 \
                --reward_scale=10 \
                --reward_bias=-5 \
                --replay_buffer_original_bias=-5 \
                --replay_buffer_original_scale=10 \
                --cql.use_rnd=False \
                --logging.output_dir=./results/cql_exploitation_maze2d_dataset_goal_topleft_bottomleft_removed/timestep_{timestep}/seed_0 \
                --exploitation_timestep={timestep} \
                --replay_buffer_path=./results/calql_rnd50_maze2d_dataset_goal_topleft_bottomleft_removed/seed_0/replay_buffer.npz \
                --bound_q_functions_according_to_data=True \
                --cql.feature_normalization=True \
                --upsample_batch_size=32
```


Hammer-truncated-expert-v1:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_calql/conservative_sac_exploitation.py --exp_name=cql_exploitation_hammer_truncated_timestep_{timestep:250000,500000,1000000} \
                --env=hammer-expert-v1  \
                --seed=0 \
                --cql_min_q_weight=1.0 \
                --policy_arch=512-512 \
                --qf_arch=512-512-512 \
                --mixing_ratio=-1 \
                --enable_calql=False \
                --max_len=20 \
                --cql.use_rnd=False \
                --logging.output_dir=./results/cql_exploitation_hammer_truncated/timestep_{timestep}/seed_0 \
                --exploitation_timestep={timestep} \
                --replay_buffer_path=./results/calql_rnd0.5_hammer_truncated/seed_0/replay_buffer.npz \
                --bound_q_functions_according_to_data=True \
                --cql.feature_normalization=True \
                --reward_scale=0.01 \
                --upsample_batch_size=32
```


Adroit binary tasks (relocate, door, pen):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_calql/conservative_sac_exploitation.py --exp_name=cql_exploitation_{task:relocate,door,pen}_timestep_{timestep:2.5e5,5e5,7.5e5,1e6} \
                --env=***env***-binary-v0 \
                --logging.online \
                --seed=0 \
                --cql_min_q_weight=1.0 \
                --policy_arch=512-512 \
                --qf_arch=512-512-512 \
                --mixing_ratio=0.5 \
                --reward_scale=10 \
                --reward_bias=5.0 \
                --replay_buffer_original_bias=5.0 \
                --replay_buffer_original_scale=10 \
                --enable_calql=False \
                --cql.use_rnd=False \
                --logging.output_dir=./results/cql_exploitation_{task}/timestep_{timestep}/seed_0 \
                --exploitation_timestep={timestep} \
                --replay_buffer_path=./results/{task}_rnd0.5/seed_0/replay_buffer.npz \
                --bound_q_functions_according_to_data=True \
                --cql.feature_normalization=True
```
