######################################################
#                     rnd_net.py                     #
#   Random Network Distillation models and training  #
######################################################

from typing import Callable, Dict, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt

from common import Batch, InfoDict, MLP, Model, Params, PRNGKey


class RNDNet(nn.Module):
    """Random Network Distillation (RND) network."""

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the RND network.

        Args:
            observations: Batch of observations.

        Returns:
            The output of the RND network.
        """
        return MLP(self.hidden_dims, activations=self.activations)(observations)


def update_rnd(
    rng: PRNGKey,
    predictor: Model,
    target: Model,
    batch: Batch,
    observations_running_mean_std: Dict[str, jnp.ndarray],
    update_proportion: float,
) -> Tuple[Model, InfoDict]:
    """Update the RND network.

    Args:
        predictor: The RND predictor network.
        target: The RND target network.
        batch: The batch of observations.

    Returns:
        The updated RND predictor network and the RND loss.
    """
    normalized_next_obs = (
        batch.next_observations - observations_running_mean_std["mean"]
    ) / jnp.sqrt(observations_running_mean_std["var"])
    normalized_next_obs = jnp.clip(normalized_next_obs, -5, 5)
    target_next_state_features = target(normalized_next_obs)

    def rnd_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        """Loss function for the RND network."""
        predictor_next_state_features = predictor.apply(
            {"params": params}, normalized_next_obs
        )
        loss = jnp.mean(
            jnp.square(predictor_next_state_features - target_next_state_features),
            axis=-1,
        )
        mask = jax.random.uniform(rng, shape=(len(loss),)) < update_proportion
        loss = (loss * mask).sum() / jnp.maximum(mask.sum(), 1)
        return loss, {"rnd_loss": loss}

    new_predictor, info = predictor.apply_gradient(rnd_loss_fn)
    return new_predictor, info


def calculate_intrinsic_reward(
    predictor: Model,
    target: Model,
    next_observations: jnp.ndarray,
    observations_running_mean_std: Dict[str, jnp.ndarray],
    rewards_running_mean_std: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Calculate the intrinsic reward.

    Args:
        predictor: The RND predictor network.
        target: The RND target network.
        next_observations: The batch of next observations.
        observations_running_mean_std: The running mean and standard deviation of the observations.
        rewards_running_mean_std: The running mean and standard deviation of the rewards.

    Returns:
        The intrinsic reward.
    """
    assert next_observations.ndim == 2
    normalized_next_obs = (
        next_observations - observations_running_mean_std["mean"]
    ) / jnp.sqrt(observations_running_mean_std["var"])
    normalized_next_obs = jnp.clip(normalized_next_obs, -5, 5)
    target_next_state_features = target(normalized_next_obs)
    predictor_next_state_features = predictor(normalized_next_obs)
    feature_diff = jnp.square(
        predictor_next_state_features - target_next_state_features
    )
    assert feature_diff.ndim == 2
    # intrinsic_reward = jnp.sum(feature_diff, axis=1)
    intrinsic_reward = jnp.max(feature_diff, axis=1)

    # Normalize the intrinsic reward.
    intrinsic_reward = intrinsic_reward / jnp.sqrt(rewards_running_mean_std["var"])
    return intrinsic_reward


class PointMassStateActionVisitationTracker:
    def __init__(
        self, state_discretization: int = 100, action_discretization: int = 10
    ):
        self.state_discretization = state_discretization
        self.action_discretization = action_discretization

        # Create a numpy array with all the possible states in the grid
        self.min_x, self.max_x, self.min_y, self.max_y = -0.25, 1.25, -0.25, 1.45
        self.min_a, self.max_a = -1, 1
        self.x = np.linspace(self.min_x, self.max_x, self.state_discretization)
        self.y = np.linspace(self.min_y, self.max_y, self.state_discretization)
        self.a1 = np.linspace(self.min_a, self.max_a, self.action_discretization)
        self.a2 = np.linspace(self.min_a, self.max_a, self.action_discretization)
        self.state_action_visitation_counts = np.zeros(
            (
                self.state_discretization,
                self.state_discretization,
                self.action_discretization,
                self.action_discretization,
            )
        )

    # increment the visitation count of the closest state
    def update(self, state, action):
        x_idx = np.argmin(np.abs(self.x - state[0]))
        y_idx = np.argmin(np.abs(self.y - state[1]))
        a1_idx = np.argmin(np.abs(self.a1 - action[0]))
        a2_idx = np.argmin(np.abs(self.a2 - action[1]))
        self.state_action_visitation_counts[x_idx, y_idx, a1_idx, a2_idx] += 1

    # vectorize getting the visitation counts
    def get_state_visitation_counts(self, states):
        """
        self.states: np.array of shape (N, 2)
        states: np.array of shape (B, 2)

        Returns: visitation counts as np.array of shape (B,)
        """
        x_indices = np.argmin(np.abs(self.x[:, np.newaxis] - states[:, 0]), axis=0)
        y_indices = np.argmin(np.abs(self.y[:, np.newaxis] - states[:, 1]), axis=0)
        return self.state_action_visitation_counts.sum(axis=(2, 3))[
            x_indices, y_indices
        ]

    def get_state_action_visitation_counts(self, states, actions):
        """
        self.states: np.array of shape (N, 2)
        states: np.array of shape (B, 2)

        Returns: visitation counts as np.array of shape (B,)
        """
        x_indices = np.argmin(np.abs(self.x[:, np.newaxis] - states[:, 0]), axis=0)
        y_indices = np.argmin(np.abs(self.y[:, np.newaxis] - states[:, 1]), axis=0)
        a1_indices = np.argmin(np.abs(self.a1[:, np.newaxis] - actions[:, 0]), axis=0)
        a2_indices = np.argmin(np.abs(self.a2[:, np.newaxis] - actions[:, 1]), axis=0)
        return self.state_action_visitation_counts[
            x_indices, y_indices, a1_indices, a2_indices
        ]

    def plot_agent_visitation(self, save_path):
        visitation_img = self.state_action_visitation_counts.sum(axis=(2, 3))
        visitation_img = visitation_img.swapaxes(0, 1)
        visitation_img = np.flip(visitation_img, axis=0)

        # Plot the intrinsic reward
        fig, ax = plt.subplots()
        im = ax.imshow(
            visitation_img, extent=[self.min_x, self.max_x, self.min_y, self.max_y]
        )
        fig.colorbar(im)
        ax.set_title("Agent Visitation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.savefig(save_path)
        plt.close()

    def plot_count_based_reward(self, save_path):
        count_based_reward = self.state_action_visitation_counts.sum(axis=(2, 3))
        count_based_reward = 1 / np.sqrt(count_based_reward + 1)
        count_based_reward = count_based_reward.swapaxes(0, 1)
        count_based_reward = np.flip(count_based_reward, axis=0)

        # Plot the intrinsic reward
        fig, ax = plt.subplots()
        im = ax.imshow(
            count_based_reward, extent=[self.min_x, self.max_x, self.min_y, self.max_y]
        )
        fig.colorbar(im)
        ax.set_title("Count Based Reward")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.savefig(save_path)
        plt.close()

    def save(self, save_path):
        np.save(save_path, self.state_action_visitation_counts)

    def load(self, load_path):
        self.state_action_visitation_counts = np.load(load_path)
