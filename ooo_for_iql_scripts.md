# OOO for IQL

### Run offline pre-training and online fine-tuning for exploratory data collection

Kitchen:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_finetune_decoupled.py --env_name=kitchen-{dataset:partial,complete,mixed}-v0 \
                --config=OOO_for_iql/configs/kitchen_decoupled_finetune_config.py \
                --exp_name=iql_rnd10_kitchen_{dataset} \
                --replay_buffer_size=4500000 \
                --max_steps=4000000 \
                --num_pretraining_steps=1000000 \
                --seed=0 \
                --rewards_bias=-4 \
                --use_rnd=True \
                --intrinsic_reward_scale=10
```

Antmaze-goal-missing-large-v2:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_finetune_decoupled.py --env_name=antmaze-large-diverse-v2 \
                --config=OOO_for_iql/configs/antmaze_decoupled_finetune_config.py \
                --exp_name=iql_rnd10_antmaze-large-diverse-v2_dataset_goal_removed_radius_2.5 \
                --seed=0 \
                --dataset_path=./datasets/antmaze-large-diverse-v2-goal-data-removed_radius_2.5/seed_0.npz \
                --max_steps=2000000 \
                --replay_buffer_size=3000000 \
                --rewards_bias=-1 \
                --use_rnd=True \
                --intrinsic_reward_scale=10
```

Maze2d-missing-data-large-v1:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_finetune_decoupled.py --env_name=maze2d-large-v1 \
                --config=OOO_for_iql/configs/maze2d_decoupled_finetune_config.py \
                --exp_name=iql_rnd10_maze2d_dataset_goal_topleft_bottomleft_removed \
                --seed=0 \
                --dataset_path=./datasets/maze2d-large-v1-500k_downsample_goal_and_top_left/seed_0.npz \
                --max_steps=500000 \
                --rewards_bias=-1 \
                --use_rnd=True \
                --intrinsic_reward_scale=10
```

Hammer-truncated-expert-v1

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_finetune_decoupled.py --exp_name=iql_rnd0.5_hammer_truncated \
                --env_name hammer-expert-v1 \
                --max_len 20 \
                --seed=0 \
                --num_pretraining_steps 50000 \
                --max_steps=500000 \
                --config=OOO_for_iql/configs/hammer_decoupled_finetune_config.py \
                --use_rnd=True \
                --intrinsic_reward_scale=0.5
```

Adroit binary tasks (relocate, door, pen). Requires AWAC datasets and `mjrl` installation (see [Adroit Manipulation Suite Setup](README.md#adroit-manipulation-suite-setup)).

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_finetune_decoupled.py --env_name={task:relocate,door,pen}-binary-v0 \
                --config=OOO_for_iql/configs/adroit_manipulation_decoupled_finetune_config.py \
                --exp_name=iql_rnd0.5_{task} \
                --seed=0 \
                --max_steps=1000000 \
                --num_pretraining_steps=25000 \
                --use_rnd=True \
                --intrinsic_reward_scale=0.5
```


### Run Offline RL for Exploitation phase

Kitchen:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_only_exploitation.py --exp_name=iql_rnd10_exploitation_kitchen_{dataset:partial,complete,mixed} \
                --env_name=kitchen-{dataset}-v0 \
                --seed=0 \
                --config=OOO_for_iql/configs/kitchen_exploitation_only_upsampling_config.py \
                --max_steps=2000000 \
                --replay_buffer_path=./results/iql_rnd10_kitchen_{dataset}/0/replay_buffer.npz \
                --offline_dataset_size={dataset:136950,3680,136950} \
                --online_eval_timesteps=1000000,2000000,3000000,4000000 \
                --rewards_bias=-4
```

Antmaze-goal-missing-large-v2:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_only_exploitation.py --exp_name=iql_rnd10_exploitation_antmaze_large-diverse-v2_dataset_goal_removed_radius_2.5 \
                --env_name=antmaze-large-diverse-v2 \
                --seed=0 \
                --config=OOO_for_iql/configs/antmaze_exploitation_only_config.py \
                --max_steps=2000000 \
                --replay_buffer_path=./results/iql_rnd10_antmaze-large-diverse-v2_dataset_goal_removed_radius_2.5/0/replay_buffer.npz \
                --offline_dataset_size=942157 \
                --online_eval_timesteps=500000,750000,1000000,1250000,1500000 \
                --rewards_bias=-1
```

Maze2d-missing-data-large-v1:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_only_exploitation.py --env_name=maze2d-large-v1 \
                --config=OOO_for_iql/configs/maze2d_exploitation_only_config.py \
                --exp_name=iql_rnd10_exploitation_maze2d_dataset_goal_topleft_bottomleft_removed \
                --seed=0 \
                --max_steps=2000000 \
                --replay_buffer_path=./results/iql_rnd10_maze2d_dataset_goal_topleft_bottomleft_removed/0/replay_buffer.npz \
                --offline_dataset_size=481287 \
                --online_eval_timesteps=250000,500000 \
                --rewards_bias=-1
```


Hammer-truncated-expert-v1

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_only_exploitation.py --exp_name=iql_rnd0.5_exploitation_hammer_truncated \
                --env_name=hammer-expert-v1 \
                --max_len=20 \
                --seed=0 \
                --config=OOO_for_iql/configs/manipulation_exploitation_only_config.py \
                --max_steps=2000000 \
                --replay_buffer_path=./results/iql_rnd0.5_hammer_truncated/0/replay_buffer.npz \
                --offline_dataset_size=100000 \
                --online_eval_timesteps=250000,500000
```

Adroit binary tasks (relocate, door, pen). Requires AWAC datasets and `mjrl` installation (see [Adroit Manipulation Suite Setup](README.md#adroit-manipulation-suite-setup)).

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python OOO_for_iql/train_only_exploitation.py --exp_name=iql_rnd0.5_exploitation_{task:relocate,door,pen} \
                --env_name={task}-binary-v0 \
                --seed=0 \
                --config=OOO_for_iql/configs/manipulation_exploitation_only_config.py \
                --max_steps=2000000 \
                --replay_buffer_path=./results/iql_rnd0.5_***task***/0/replay_buffer.npz \
                --offline_dataset_size={task:98937,96000,94378} \
                --online_eval_timesteps=250000,500000,750000,1000000
```
