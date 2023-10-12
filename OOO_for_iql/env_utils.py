######################################################
#                    env_utils.py                    #
######################################################

from typing import Optional, Tuple
from dataset_utils import (
    Dataset,
    D4RLDataset,
    trim_dataset,
)
import gym
import numpy as np

import wrappers


def make_env_and_dataset(
    env_name: str,
    seed: int,
    dataset_path: Optional[str] = None,
    max_len: Optional[int] = None,
) -> Tuple[gym.Env, D4RLDataset, gym.Env]:
    """
    Make an environment and a dataset from a given environment name.

    Args:
        env_name: Name of the environment.
        seed: Random seed.
        dataset_path (optional): if given, loads from npz file instead of default D4RL dataset.
        max_len (optional): if given, trims each trajectory to the given length.

    Returns:
        The training environment, the dataset, and the evaluation environment.
    """
    if env_name == "point_mass_wall":
        from envs.point_mass_wall import PointMassWallEnv, steps_to_goal_reward_function

        train_env = PointMassWallEnv()
        train_env = gym.wrappers.TimeLimit(train_env, 150)
        eval_env = PointMassWallEnv(reward_function=steps_to_goal_reward_function)
        eval_env = gym.wrappers.TimeLimit(eval_env, 150)

        dataset_path = "./datasets/point_mass_wall.npz"
        dataset = np.load(dataset_path, allow_pickle=True).item()
        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
        dataset["rewards"] = dataset["rewards"][..., None]
        dataset["terminals"] = dataset["terminals"][..., None]
        # The dataset contains rotation info and the goal, so truncate them
        dataset["observations"] = dataset["observations"][:, :2]
        dataset["next_observations"] = dataset["next_observations"][:, :2]

        # Clip actions
        actions_limit = 1 - 1e-5
        dataset["actions"] = np.clip(dataset["actions"], -actions_limit, actions_limit)
        dataset = Dataset(
            observations=dataset["observations"],
            actions=dataset["actions"],
            rewards=dataset["rewards"].reshape(-1),
            masks=1 - dataset["terminals"].reshape(-1),
            dones_float=dataset["terminals"].reshape(-1),
            next_observations=dataset["next_observations"],
            size=len(dataset["rewards"]),
        )
    else:
        if "binary" in env_name:
            import mj_envs  # noqa: F401
            from binary_datasets import BinaryDataset

            env = gym.make(env_name)

            dataset = BinaryDataset(env, include_bc_data=True)
        else:
            env = gym.make(env_name)
            dataset = D4RLDataset(env, load_from_path=dataset_path)
        eval_env = gym.make(env_name)

    if max_len is not None:
        dataset = trim_dataset(dataset, max_len)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    eval_env = wrappers.EpisodeMonitor(eval_env)
    eval_env = wrappers.SinglePrecision(eval_env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env, dataset, eval_env


def load_replay_buffer(
    data_path: str,
):
    dataset = dict(np.load(data_path))
    offline_dataset = Dataset(
        observations=dataset["observations"],
        actions=dataset["actions"],
        rewards=dataset["rewards"],
        masks=dataset["masks"],
        dones_float=dataset["dones_float"],
        next_observations=dataset["next_observations"],
        size=len(dataset["rewards"]),
    )
    return offline_dataset


def get_kitchen_successful_samples(dataset):
    terminals = dataset["dones_float"]
    indices = np.nonzero(terminals)[0] + 1
    trajectories = {k: np.split(v, indices) for k, v in dataset.items()}
    if len(trajectories["terminals"][-1]) == 0:
        for k, v in trajectories.items():
            trajectories[k] = v[:-1]
    trajectory_to_max_reward = {}
    for i in range(len(trajectories["rewards"])):
        trajectory_to_max_reward[i] = trajectories["rewards"][i].max()
    # Get a list where max_reward == 4
    complete_success_indeces = [
        k for k, v in trajectory_to_max_reward.items() if v == 4
    ]
    observations, actions, rewards, masks, dones_float, next_observations = (
        np.concatenate([trajectories[k][i] for i in complete_success_indeces])
        for k in [
            "observations",
            "actions",
            "rewards",
            "masks",
            "dones_float",
            "next_observations",
        ]
    )

    return (
        observations,
        actions,
        rewards,
        masks,
        dones_float,
        next_observations,
    )
