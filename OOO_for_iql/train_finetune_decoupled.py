######################################################
#             train_finetune_decoupled.py            #
#              The first two O's of OOO              #
#  Offline pretraining followed by online finetuning #
######################################################

import os
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags, ConfigDict
import wandb

from dataset_utils import (
    Batch,
    ReplayBuffer,
)
from evaluation import evaluate
from learner import Learner
from rnd_net import calculate_intrinsic_reward, PointMassStateActionVisitationTracker
from env_utils import make_env_and_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Experiment name for WandB and logging path.")
flags.DEFINE_string("env_name", "point_mass_wall", "Environment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer(
    "max_len", None, "Trim every trajectory in offline dataset to this length."
)
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 50000, "Eval interval.")
flags.DEFINE_integer("save_interval", 100000, "Save interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("num_pretraining_steps", int(5e5), "Number of pretraining steps.")
flags.DEFINE_integer(
    "replay_buffer_size", None, "Replay buffer size (=max_steps if unspecified)."
)
flags.DEFINE_integer(
    "init_dataset_size", None, "Offline data size (uses all data if unspecified)."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
config_flags.DEFINE_config_file(
    "config",
    "configs/decoupled_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_string(
    "dataset_path",
    None,
    "Path to dataset for D4RL tasks. If None, uses default D4RL dataset",
)
flags.DEFINE_boolean("use_rnd", False, "Use RND intrinsic reward.")
flags.DEFINE_float("intrinsic_reward_scale", 0, "Intrinsic reward scale")
flags.DEFINE_integer("rewards_bias", 0, "Rewards bias.")
flags.DEFINE_integer("rewards_scale", 1, "Rewards scale.")


def main(_):
    print("Environment name: ", FLAGS.env_name)
    assert FLAGS.exp_name is not None
    # Put all flags into a dictionary for logging
    flags_dict = FLAGS.flag_values_dict()
    flags_dict["config"] = {
        k: v if not isinstance(v, ConfigDict) else v.to_dict()
        for k, v in FLAGS.config.items()
    }

    wandb.init(
        project="OOO-iql",
        settings=wandb.Settings(start_method="fork", _service_wait=300),
        name=FLAGS.exp_name,
        config=flags_dict,
    )
    print(f"Experiment Name: {FLAGS.exp_name}")
    save_dir = f"./results/{FLAGS.exp_name}/{FLAGS.seed}/"
    os.makedirs(save_dir, exist_ok=True)

    env, offline_dataset, eval_env = make_env_and_dataset(
        FLAGS.env_name,
        FLAGS.seed,
        max_len=FLAGS.max_len,
        dataset_path=FLAGS.dataset_path,
    )

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(
        env.observation_space, action_dim, FLAGS.replay_buffer_size or FLAGS.max_steps
    )
    replay_buffer.initialize_with_dataset(offline_dataset, FLAGS.init_dataset_size)

    # Initialize count-based visitation bonus specifically for point_mass_wall
    count_bonus_tracker = None
    if FLAGS.env_name == "point_mass_wall":
        visitation_path = os.path.join(save_dir, "visitation_plots")
        os.makedirs(visitation_path, exist_ok=True)

        count_bonus_tracker = PointMassStateActionVisitationTracker(
            state_discretization=100, action_discretization=10
        )

        for offline_obs, offline_action in zip(
            offline_dataset.observations, offline_dataset.actions
        ):
            count_bonus_tracker.update(offline_obs, offline_action)
        count_bonus_tracker.plot_agent_visitation(
            os.path.join(visitation_path, "offline_visitation.png")
        )
        count_bonus_tracker.save(
            os.path.join(visitation_path, "count_bonus_offline.npy")
        )

    # setup exploration agent used for online data collection
    exploration_kwargs = dict(FLAGS.config.exploration_agent_config)
    if exploration_kwargs["bound_q_functions"]:
        min_reward = FLAGS.config.min_reward
        max_reward = FLAGS.config.max_reward
        assert min_reward is not None and max_reward is not None
        min_q = min_reward / (1 - FLAGS.config.get("discount", 0.99))
        max_q = max_reward / (1 - FLAGS.config.get("discount", 0.99))
    else:
        min_q = None
        max_q = None
    exploration_agent = Learner(
        FLAGS.seed,
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        visitation_tracker=count_bonus_tracker,
        minimum_q=min_q,
        maximum_q=max_q,
        use_rnd=FLAGS.use_rnd,
        intrinsic_reward_scale=FLAGS.intrinsic_reward_scale,
        **exploration_kwargs,
    )

    observation, done = env.reset(), False

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(
        range(1 - FLAGS.num_pretraining_steps, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
        desc="Training iteration",
    ):
        if (
            FLAGS.env_name == "point_mass_wall"
            and i >= 1
            and (i + FLAGS.num_pretraining_steps - 1) % 10000 == 0
        ):
            # Make plots of visitation and count-based reward
            count_bonus_tracker.plot_agent_visitation(
                os.path.join(
                    visitation_path,
                    f"online_visitation_{i + FLAGS.num_pretraining_steps - 1}.png",
                )
            )
            count_bonus_tracker.plot_count_based_reward(
                os.path.join(
                    visitation_path,
                    f"intrinsic_reward_{i + FLAGS.num_pretraining_steps - 1}.png",
                )
            )
            count_bonus_tracker.save(
                os.path.join(
                    visitation_path,
                    f"count_bonus_{i + FLAGS.num_pretraining_steps - 1}.npy",
                )
            )

        if i >= 1:
            # The first O (Offline pre-training) is over - this does the second O (Online data collection)
            action = exploration_agent.sample_actions(
                observation,
                temperature=exploration_kwargs.get("online_sample_temperature", 1.0),
            )
            next_observation, reward, done, info = env.step(action)
            if FLAGS.env_name == "point_mass_wall":
                # update visitation counts
                count_bonus_tracker.update(observation, action)

            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(
                observation, action, reward, mask, float(done), next_observation
            )
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info["episode"].items():
                    wandb.log({f"train/{k}": v}, step=i + FLAGS.num_pretraining_steps)
                wandb.log(
                    {"train/online_step": i}, step=i + FLAGS.num_pretraining_steps
                )

        # Get data for agent update
        batch = replay_buffer.sample(FLAGS.batch_size)
        scaled_rewards = (batch.rewards + FLAGS.rewards_bias) * FLAGS.rewards_scale
        masks = batch.masks
        """
        Kitchen dataset has terminations when the demonstrations end. Using this is incorrect,
        because we do want to bootstrap from the final state of the demonstration.
        We modify the data to only terminate when the 4 tasks are completed.
        """
        if "kitchen" in FLAGS.env_name:
            masks = (batch.rewards != 4.0).astype(np.float32)
        batch = Batch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=scaled_rewards,
            masks=masks,
            next_observations=batch.next_observations,
        )
        if "antmaze" in FLAGS.env_name:
            assert FLAGS.rewards_bias == -1 and FLAGS.rewards_scale == 1, (
                "Original IQL implementation manually uses rewards_bias=-1 and rewards_scale=1."
                "If you really want to change this, feel free to remove this assertion."
            )

        update_info = exploration_agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb.log({f"train/{k}": v}, step=i + FLAGS.num_pretraining_steps)
                else:
                    wandb.log(
                        {f"train/{k}": wandb.Histogram(v)},
                        step=i + FLAGS.num_pretraining_steps,
                    )

            # log the RND average intrinsic reward for the last 1000 steps.
            if FLAGS.use_rnd and i >= 1:
                next_observations = replay_buffer.next_observations[
                    replay_buffer.size - 1000 : replay_buffer.size
                ]
                intrinsic_rewards = calculate_intrinsic_reward(
                    exploration_agent.rnd_predictor,
                    exploration_agent.rnd_target,
                    next_observations,
                    exploration_agent.rnd_obs_rms.to_container(),
                    exploration_agent.rnd_reward_rms.to_container(),
                )
                wandb.log(
                    {
                        "train/average_intrinsic_reward": np.mean(intrinsic_rewards),
                        "train/average_scaled_intrinsic_reward": (
                            np.mean(intrinsic_rewards) * FLAGS.intrinsic_reward_scale
                        ),
                        "train/std_intrinsic_reward": np.std(intrinsic_rewards),
                    },
                    step=i + FLAGS.num_pretraining_steps,
                )

        if i % FLAGS.eval_interval == 0:
            # evaluate the exploration agent
            eval_stats = evaluate(exploration_agent, eval_env, FLAGS.eval_episodes)
            for k, v in eval_stats.items():
                wandb.log(
                    {f"eval/average_{k}": v},
                    step=i + FLAGS.num_pretraining_steps,
                )

        if i % FLAGS.save_interval == 0:
            exploration_agent.save_checkpoint(
                os.path.join(save_dir, f"exploration_agent_{i}")
            )

    print("Finished training.")
    # Save replay buffer.
    replay_buffer.save(os.path.join(save_dir, "replay_buffer.npz"))
    # Save final exploration agent.
    exploration_agent.save_checkpoint(os.path.join(save_dir, "exploration_agent"))


if __name__ == "__main__":
    app.run(main)
