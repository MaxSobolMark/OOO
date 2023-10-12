######################################################
#                     rnd_net.py                     #
#    Random Network Distillation models, training    #
#    and utilities for Cal-QL intrinsic MC returns   #
######################################################


from typing import Any, Callable, Dict, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import flax
from flax.training.train_state import TrainState

from model import FullyConnectedNetwork

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]


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
        hidden_dims = [str(h) for h in self.hidden_dims]
        return FullyConnectedNetwork(
            output_dim=self.hidden_dims[-1],
            arch="-".join(hidden_dims),
            orthogonal_init=True,
        )(observations)[0]


def update_rnd(
    rng: PRNGKey,
    predictor: TrainState,
    target: TrainState,
    next_observations: jnp.ndarray,
    observations_running_mean_std: Dict[str, jnp.ndarray],
    update_proportion: float,
) -> Tuple[TrainState, Dict[str, float]]:
    """Update the RND network.

    Args:
        rng: Jax Random number generator.
        predictor: The RND predictor network.
        target: The RND target network.
        next_observations: Batch of next observations.
        observations_running_mean_std: Running mean and std of the observations.
        update_proportion: Proportion of the batch to use for the update.

    Returns:
        The updated RND predictor network and the RND loss.
    """
    normalized_next_obs = (
        next_observations - observations_running_mean_std["mean"]
    ) / jnp.sqrt(observations_running_mean_std["var"])
    normalized_next_obs = jnp.clip(normalized_next_obs, -5, 5)
    target_next_state_features = target.apply_fn(
        {"params": target.params}, normalized_next_obs
    )

    def rnd_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Loss function for the RND network."""
        predictor_next_state_features = predictor.apply_fn(
            {"params": params}, normalized_next_obs
        )
        loss = jnp.mean(
            jnp.square(predictor_next_state_features - target_next_state_features),
            axis=-1,
        )
        max_loss = jnp.max(loss)
        min_loss = jnp.min(loss)
        loss_std = jnp.std(loss)
        # Only use a proportion of the batch for the update.
        mask = jax.random.uniform(rng, shape=(len(loss),)) < update_proportion
        loss = (loss * mask).sum() / jnp.maximum(mask.sum(), 1)
        return loss, {
            "rnd_loss": loss,
            "max_rnd_loss": max_loss,
            "min_rnd_loss": min_loss,
            "std_rnd_loss": loss_std,
        }

    grads, info = jax.grad(rnd_loss_fn, has_aux=True)(predictor.params)
    new_predictor = predictor.apply_gradients(grads=grads)
    return new_predictor, info


def calculate_intrinsic_reward(
    predictor: TrainState,
    target: TrainState,
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
    target_next_state_features = target.apply_fn(
        {"params": target.params}, normalized_next_obs
    )
    predictor_next_state_features = predictor.apply_fn(
        {"params": predictor.params}, normalized_next_obs
    )
    feature_diff = jnp.square(
        predictor_next_state_features - target_next_state_features
    )
    assert feature_diff.ndim == 2
    intrinsic_reward = jnp.max(feature_diff, axis=1)

    # Normalize the intrinsic reward.
    intrinsic_reward = intrinsic_reward / jnp.sqrt(rewards_running_mean_std["var"])
    return intrinsic_reward


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = jnp.zeros(shape, "float64")
        self.var = jnp.ones(shape, "float64")
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


def functional_running_mean_std_update(
    container: Dict[str, jnp.ndarray], x: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    batch_count = x.shape[0]
    return functional_running_mean_std_update_from_moments(
        container, batch_mean, batch_var, batch_count
    )


def functional_running_mean_std_update_from_moments(
    container: Dict[str, jnp.ndarray],
    batch_mean: jnp.ndarray,
    batch_var: jnp.ndarray,
    batch_count: int,
) -> Dict[str, jnp.ndarray]:
    delta = batch_mean - container["mean"]
    tot_count = container["count"] + batch_count

    new_mean = container["mean"] + delta * batch_count / tot_count
    m_a = container["var"] * (container["count"])
    m_b = batch_var * (batch_count)
    M2 = (
        m_a
        + m_b
        + jnp.square(delta)
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
