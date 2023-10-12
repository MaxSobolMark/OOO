from typing import Optional, Tuple
from tqdm import tqdm
from rlpd.data.dataset import Dataset
from rlpd.data.d4rl_datasets import D4RLDataset

try:
    from rlpd.data.binary_datasets import BinaryDataset
except:
    print("not importing binary dataset")
from rlpd.wrappers import wrap_gym

import gym
import numpy as np
import math


def split_into_trajectories(
    observations, actions, rewards, masks, dones, next_observations
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones[i],
                next_observations[i],
            )
        )
        if dones[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones = []
    next_observations = []

    for traj in trajs:
        for obs, act, rew, mask, done, next_obs in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones.append(done)
            next_observations.append(next_obs)

    return {
        "observations": np.stack(observations),
        "actions": np.stack(actions),
        "rewards": np.stack(rewards),
        "masks": np.stack(masks),
        "dones": np.stack(dones),
        "next_observations": np.stack(next_observations),
    }


def trim_dataset(dataset, max_len):
    trajs = split_into_trajectories(
        dataset.dataset_dict["observations"],
        dataset.dataset_dict["actions"],
        dataset.dataset_dict["rewards"],
        dataset.dataset_dict["masks"],
        dataset.dataset_dict["dones"],
        dataset.dataset_dict["next_observations"],
    )

    new_trajs = []
    for traj in trajs:
        new_traj = traj[:max_len]
        new_traj[-1] = list(new_traj[-1])
        new_traj[-1][4] = 1.0  # set done to 1.0
        new_traj[-1] = tuple(new_traj[-1])
        new_trajs.append(new_traj)

    merged = merge_trajectories(new_trajs)
    return Dataset(merged)


def make_env_and_dataset(
    env_name: str,
    seed: int,
    include_bc_data: bool,
    dataset_path: Optional[str] = None,
    max_len: Optional[int] = None,
) -> Tuple[gym.Env, D4RLDataset, gym.Env]:
    if env_name == "halfcheetah-sparse-v2":
        from rlpd.envs.half_cheetah_sparse import SparseHalfCheetahEnv
        from gym.wrappers import TimeLimit

        env = SparseHalfCheetahEnv()
        env = TimeLimit(env, max_episode_steps=1000)
        eval_env = SparseHalfCheetahEnv()
        eval_env = TimeLimit(eval_env, max_episode_steps=1000)
        dataset = None
        assert max_len is None
    elif env_name == "ant-sparse-v2":
        from rlpd.envs.ant_sparse import SparseAntEnv
        from gym.wrappers import TimeLimit

        env = SparseAntEnv(include_xpos_in_obs=True)
        env = TimeLimit(env, max_episode_steps=1000)
        eval_env = SparseAntEnv(include_xpos_in_obs=True)
        eval_env = TimeLimit(eval_env, max_episode_steps=1000)
        dataset = None
        assert max_len is None
    else:
        env = gym.make(env_name)
        eval_env = gym.make(env_name)
        if "binary" in env_name:
            assert dataset_path is None
            dataset = BinaryDataset(env, include_bc_data=include_bc_data)
        elif env_name == "hammer-v0":
            dataset = None
            assert max_len is None
        else:
            dataset = D4RLDataset(env, load_from_path=dataset_path)

    if max_len is not None:
        dataset = trim_dataset(dataset, max_len)

    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env, deque_size=1)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    eval_env.seed(seed + 1)
    eval_env.action_space.seed(seed + 1)
    eval_env.observation_space.seed(seed + 1)

    return env, dataset, eval_env


def load_replay_buffer(
    data_path: str,
    return_upsample_dataset: bool = False,
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
    if return_upsample_dataset:
        reward1_idxs = np.where(dataset["rewards"] == 1.0)[0]
        reward1_dataset = Dataset(
            observations=dataset["observations"][reward1_idxs],
            actions=dataset["actions"][reward1_idxs],
            rewards=dataset["rewards"][reward1_idxs],
            masks=dataset["masks"][reward1_idxs],
            dones_float=dataset["dones_float"][reward1_idxs],
            next_observations=dataset["next_observations"][reward1_idxs],
            size=len(reward1_idxs),
        )
        return offline_dataset, reward1_dataset
    return offline_dataset
