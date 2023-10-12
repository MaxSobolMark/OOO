from typing import Dict, Optional

import d4rl
import gym
import numpy as np
from tqdm import tqdm
from flax import struct


@struct.dataclass
class Batch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray
    weights: Optional[np.ndarray] = None


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for obs, act, rew, mask, done, next_obs in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.stack(masks),
        np.stack(dones_float),
        np.stack(next_observations),
    )


class Dataset(object):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
        )


class ReturnWeightedDataset(Dataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trajectory_returns = np.zeros_like(self.rewards)
        self.weights = np.ones_like(self.rewards)
        self.indxs = None
        self.indxs_remaining_samples = 0

    def calculate_trajectory_returns(self):
        current_trajectory_start_index = 0
        current_trajectory_return = 0
        num_trajectories = 0
        for i in range(self.size):
            current_trajectory_return += self.rewards[i]
            if self.dones_float[i] == 1.0:
                self.trajectory_returns[
                    current_trajectory_start_index : i + 1
                ] = current_trajectory_return
                current_trajectory_start_index = i + 1
                current_trajectory_return = 0
                num_trajectories += 1
        if current_trajectory_start_index < self.size:
            self.trajectory_returns[
                current_trajectory_start_index : self.size
            ] = current_trajectory_return
            num_trajectories += 1

    def _split_weights_into_trajectories(self):
        trajs = [[]]

        for i in range(self.size):
            trajs[-1].append(self.weights[i])
            if self.dones_float[i] == 1.0 and i + 1 < self.size:
                trajs.append([])
        return trajs

    def weight_dataset(self, temperature: float = 0.1):
        self.calculate_trajectory_returns()
        self.weights = np.exp(self.trajectory_returns / temperature)
        self.weights /= self.weights.sum()

    def validate_weights(self):
        trajs_weights = self._split_weights_into_trajectories()
        for traj_weights in trajs_weights:
            assert np.allclose(traj_weights, traj_weights[0])

    def sample(self, batch_size: int) -> Batch:
        if self.indxs_remaining_samples == 0:
            self.indxs = np.random.choice(
                self.size, size=(1000, batch_size), p=self.weights
            )
            self.indxs_remaining_samples = 1000
        self.indxs_remaining_samples -= 1
        indx = self.indxs[self.indxs_remaining_samples]
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=self.masks[indx],
            next_observations=self.next_observations[indx],
            weights=self.weights[indx],
        )


class D4RLDataset(Dataset):
    def __init__(
        self,
        env: gym.Env,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
        load_from_path: Optional[str] = None,
    ):
        if load_from_path is not None:
            print("Loading dataset from path: {}".format(load_from_path))
        dataset = (
            d4rl.qlearning_dataset(env)
            if load_from_path is None
            else dict(np.load(load_from_path, allow_pickle=True))
        )

        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

        dones_float = np.zeros_like(dataset["rewards"])

        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            size=len(dataset["observations"]),
        )


class ReplayBuffer(Dataset):
    def __init__(
        self, observation_space: gym.spaces.Box, action_dim: int, capacity: int
    ):
        observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity,), dtype=np.float32)
        dones_float = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            dones_float=dones_float,
            next_observations=next_observations,
            size=0,
        )

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(
        self, dataset: Dataset, num_samples: Optional[int], shuffle_samples=False
    ):
        assert (
            self.insert_index == 0
        ), "Can insert a batch online in an empty replay buffer."

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert (
            self.capacity >= num_samples
        ), "Dataset cannot be larger than the replay buffer capacity."

        if num_samples < dataset_size:
            if shuffle_samples:
                perm = np.random.permutation(dataset_size)
                indices = perm[:num_samples]
            else:
                indices = np.arange(num_samples)
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        done_float: float,
        next_observation: np.ndarray,
    ):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def save(self, path: str):
        np.savez_compressed(
            path,
            observations=self.observations[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            masks=self.masks[: self.size],
            dones_float=self.dones_float[: self.size],
            next_observations=self.next_observations[: self.size],
            capacity=self.capacity,
        )

    def load(self, path: str, max_step: Optional[int] = None):
        data = np.load(path)
        size = len(data["rewards"])
        if max_step is not None:
            size = min(size, max_step)
        self.observations[:size] = data["observations"][:size]
        self.actions[:size] = data["actions"][:size]
        self.rewards[:size] = data["rewards"][:size]
        self.masks[:size] = data["masks"][:size]
        self.dones_float[:size] = data["dones_float"][:size]
        self.next_observations[:size] = data["next_observations"][:size]
        self.size = size
        self.insert_index = self.size
        self.capacity = data["capacity"]


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def to_container(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def from_container(self, container):
        self.mean = container["mean"]
        self.var = container["var"]
        self.count = container["count"]

    def save(self, path: str):
        np.savez(path, **self.to_container())

    def load(self, path: str):
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        data = np.load(path)
        self.from_container(data)


def functional_running_mean_std_update(
    container: Dict[str, np.ndarray], x: np.ndarray
) -> Dict[str, np.ndarray]:
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    batch_count = x.shape[0]
    return functional_running_mean_std_update_from_moments(
        container, batch_mean, batch_var, batch_count
    )


def functional_running_mean_std_update_from_moments(
    container: Dict[str, np.ndarray],
    batch_mean: np.ndarray,
    batch_var: np.ndarray,
    batch_count: int,
) -> Dict[str, np.ndarray]:
    delta = batch_mean - container["mean"]
    tot_count = container["count"] + batch_count

    new_mean = container["mean"] + delta * batch_count / tot_count
    m_a = container["var"] * (container["count"])
    m_b = batch_var * (batch_count)
    M2 = (
        m_a
        + m_b
        + np.square(delta)
        * container["count"]
        * batch_count
        / (container["count"] + batch_count)
    )
    new_var = M2 / (container["count"] + batch_count)

    new_count = batch_count + container["count"]

    container["mean"] = new_mean
    container["var"] = new_var
    container["count"] = new_count
    return container
