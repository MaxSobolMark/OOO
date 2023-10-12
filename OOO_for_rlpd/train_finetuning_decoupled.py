######################################################
#           train_finetuning_decoupled.py            #
#              The first two O's of OOO              #
#  Offline pretraining followed by online finetuning #
#    (some environments, like halfcheetah and ant    #
#       sparse, do not do offline pretraining)       #
######################################################

import os
from pathlib import Path
import pickle
import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion  # noqa: F401 (register environments with gym)
import numpy as np
import tqdm
from absl import app, flags

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer

from rlpd.evaluation import evaluate
from rlpd.env_utils import make_env_and_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 50000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", True, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", True, "Save agent replay buffer on evaluation."
)
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
flags.DEFINE_string("dataset_path", None, "Path to dataset.")
flags.DEFINE_integer(
    "max_len", None, "Trim every trajectory in offline dataset to this length."
)


def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined


def main(_):
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    wandb.init(
        project="OOO-rlpd",
        settings=wandb.Settings(start_method="fork", _service_wait=300),
        name=FLAGS.exp_name,
    )
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"

    log_dir = f"./results/{FLAGS.exp_name}/{FLAGS.seed}"
    os.makedirs(log_dir, exist_ok=True)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    env, ds, eval_env = make_env_and_dataset(
        FLAGS.env_name,
        FLAGS.seed,
        include_bc_data=FLAGS.binary_include_bc,
        max_len=FLAGS.max_len,
        dataset_path=FLAGS.dataset_path,
    )

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    # First O - Offline pre-training phase
    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v
            if "antmaze" in FLAGS.env_name and k == "rewards":
                batch[k] -= 1
            elif "kitchen" in FLAGS.env_name and k == "rewards":
                reward_bias = -5 if FLAGS.config.get("use_rnd", False) else -4
                batch[k] += reward_bias

        agent, update_info = agent.update(
            batch, FLAGS.utd_ratio, use_rnd=FLAGS.config.get("use_rnd", False)
        )

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)

    # Second O - Online exploratory data collection phase
    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            if FLAGS.offline_ratio > 0:
                offline_batch = ds.sample(
                    int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
                )

                batch = combine(offline_batch, online_batch)
            else:
                batch = online_batch

            if "antmaze" in FLAGS.env_name:
                batch["rewards"] -= 1
            elif "kitchen" in FLAGS.env_name:
                """
                Kitchen dataset has terminations when the demonstrations end. Using this is incorrect,
                because we do want to bootstrap from the final state of the demonstration.
                We modify the data to only terminate when the 4 tasks are completed.
                """
                batch["masks"] = (batch["rewards"] != 4.0).astype(np.float32)
                reward_bias = -5 if FLAGS.config.get("use_rnd", False) else -4
                batch["rewards"] += reward_bias

            agent, update_info = agent.update(
                batch, FLAGS.utd_ratio, use_rnd=FLAGS.config.get("use_rnd", False)
            )

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)
            wandb.log({"evaluation/online_step": i}, step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except Exception as e:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        # pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                        pickle.dump(replay_buffer.dataset_dict, f)
                except:
                    print("Could not save agent buffer.")


if __name__ == "__main__":
    app.run(main)
