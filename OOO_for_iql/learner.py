######################################################
#                     learner.py                     #
#              Implements the IQL agent              #
######################################################


from typing import Dict, Optional, Sequence, Tuple
from pathlib import Path
import os
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import awr_policy_update, td3_policy_update
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_q, update_v, td3_update_q
from rnd_net import RNDNet, calculate_intrinsic_reward
from rnd_net import update_rnd as update_rnd_predictor
from dataset_utils import RunningMeanStd


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    """
    Update the target critic network using soft interpolation with critic.

    Args:
        critic (Model): Current critic weights.
        target_critic (Model): The target critic model to be updated.
        tau (float): The interpolation parameter for the soft update. Should be in the range [0, 1].

    Returns:
        Model: The updated target critic model.
    """
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )

    return target_critic.replace(params=new_target_params)


# Jax jit compilation calls


@functools.partial(
    jax.jit,
    static_argnames=(
        "intrinsic_reward_scale",
        "td3_update",
        "minimum_q",
        "maximum_q",
    ),
)
def _update_critic_and_actor_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    target_critic: Model,
    batch: Batch,
    discount: float,
    tau: float,
    expectile: float,
    temperature: float,
    intrinsic_reward: Optional[jnp.ndarray] = None,
    intrinsic_reward_scale: float = 0.0,
    td3_update: bool = False,
    minimum_q: Optional[float] = None,
    maximum_q: Optional[float] = None,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    key, rng = jax.random.split(rng)

    new_value, value_info = update_v(
        target_critic, value, batch, expectile, minimum_q=minimum_q, maximum_q=maximum_q
    )

    if td3_update:
        # value functions are no longer used in TD3
        new_critic, critic_info = td3_update_q(
            key,
            critic,
            target_critic,
            actor,
            batch,
            discount,
            intrinsic_reward,
            intrinsic_reward_scale,
        )
        new_target_critic = target_update(new_critic, target_critic, tau)

        new_actor, actor_info = td3_policy_update(
            key,
            actor,
            critic,
            batch,
        )
    else:
        new_critic, critic_info = update_q(
            critic,
            new_value,
            batch,
            discount,
            intrinsic_reward,
            intrinsic_reward_scale,
            minimum_q=minimum_q,
            maximum_q=maximum_q,
        )
        new_target_critic = target_update(new_critic, target_critic, tau)

        new_actor, actor_info = awr_policy_update(
            key,
            actor,
            target_critic,
            new_value,
            batch,
            temperature,
        )

    return (
        rng,
        new_actor,
        new_critic,
        new_value,
        new_target_critic,
        {**critic_info, **value_info, **actor_info},
    )


@jax.jit
def _update_rnd_jit(
    rng: PRNGKey,
    rnd_predictor: Model,
    rnd_target: Model,
    batch: Batch,
    observations_running_mean_std: Dict[str, jnp.ndarray],
    update_proportion: float,
) -> Tuple[PRNGKey, Model, jnp.ndarray, InfoDict]:
    rnd_key, rng = jax.random.split(rng)
    new_rnd_predictor, rnd_predictor_info = update_rnd_predictor(
        rnd_key,
        rnd_predictor,
        rnd_target,
        batch,
        observations_running_mean_std,
        update_proportion,
    )

    return rng, new_rnd_predictor, rnd_predictor_info


@jax.jit
def _calculate_rnd_reward_jit(
    rnd_predictor: Model,
    rnd_target: Model,
    batch: Batch,
    observations_running_mean_std: Dict[str, jnp.ndarray],
    rewards_running_mean_std: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    intrinsic_reward = calculate_intrinsic_reward(
        rnd_predictor,
        rnd_target,
        batch.next_observations,
        observations_running_mean_std,
        rewards_running_mean_std,
    )
    return intrinsic_reward


@jax.jit
def _calculate_state_action_count_bonus_jit(
    x: jnp.ndarray,
    y: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
    counts: jnp.ndarray,
    batch: Batch,
) -> jnp.ndarray:
    def _euclidean_distance(x, y):
        return jnp.sum((x - y) ** 2)

    # Vectorized implementation for finding the closest bin for each observation
    distances_x = jax.vmap(
        lambda x_row: jax.vmap(lambda b: _euclidean_distance(x_row, b))(
            batch.observations[:, 0]
        )
    )(x)
    distances_y = jax.vmap(
        lambda y_row: jax.vmap(lambda b: _euclidean_distance(y_row, b))(
            batch.observations[:, 1]
        )
    )(y)
    distances_a1 = jax.vmap(
        lambda a1_row: jax.vmap(lambda b: _euclidean_distance(a1_row, b))(
            batch.actions[:, 0]
        )
    )(a1)
    distances_a2 = jax.vmap(
        lambda a2_row: jax.vmap(lambda b: _euclidean_distance(a2_row, b))(
            batch.actions[:, 1]
        )
    )(a2)

    # Find the indices of the smallest distances for each row in the batch
    closest_indices_x = jnp.argmin(distances_x, axis=0)
    closest_indices_y = jnp.argmin(distances_y, axis=0)
    closest_indices_a1 = jnp.argmin(distances_a1, axis=0)
    closest_indices_a2 = jnp.argmin(distances_a2, axis=0)

    # return the intrinsic reward for each row in the batch
    intrinsic_rewards = 1.0 / jnp.sqrt(
        counts[
            closest_indices_x, closest_indices_y, closest_indices_a1, closest_indices_a2
        ]
        + 1.0
    )
    return intrinsic_rewards


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        dropout_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
        opt_decay_schedule: str = "cosine",
        policy_log_std_min: float = -5,
        td3_update: bool = False,  # defaults to AWR
        minimum_q: Optional[float] = None,
        maximum_q: Optional[float] = None,
        # parameters related to exploration / intrinsic rewards
        intrinsic_reward_scale: float = 0.0,
        # ensembling / disagreement parameters
        critic_ensemble_size: int = 2,
        # rnd parameters
        use_rnd: bool = False,
        rnd_hidden_dims: Sequence[int] = (512, 512),
        rnd_lr: float = 1e-4,
        rnd_update_proportion: float = 0.25,
        # count based bonus
        use_count_based_bonus: bool = False,
        visitation_tracker=None,
        **kwargs,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.td3_update = td3_update
        self.minimum_q = minimum_q
        self.maximum_q = maximum_q
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,  # not used when state_dependent_std=False
            log_std_min=policy_log_std_min,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=self.td3_update,
        )

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(
                optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
            )
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optimiser)

        critic_ensemble_def = value_net.CriticEnsemble(
            critic_ensemble_size, hidden_dims
        )
        critic_ensemble = Model.create(
            critic_ensemble_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )

        value_def = value_net.ValueCritic(hidden_dims)
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr),
        )

        target_critic_ensemble = Model.create(
            critic_ensemble_def,
            inputs=[critic_key, observations, actions],
        )

        # intrinsic rewards
        self.intrinsic_reward_scale = intrinsic_reward_scale

        # RND
        self.use_rnd = use_rnd
        if self.use_rnd:
            rng, rnd_key, rnd_target_key = jax.random.split(rng, 3)
            rnd_def = RNDNet(rnd_hidden_dims)
            self.rnd_predictor = Model.create(
                rnd_def,
                inputs=[rnd_key, observations],
                tx=optax.adam(learning_rate=rnd_lr),
            )
            self.rnd_target = Model.create(
                rnd_def, inputs=[rnd_target_key, observations]
            )
            self.rnd_obs_rms = RunningMeanStd(shape=observations.shape[1:])
            self.rnd_reward_rms = RunningMeanStd()
            self.rnd_update_proportion = rnd_update_proportion

        # count based bonus
        self.use_count_based_bonus = use_count_based_bonus
        if self.use_count_based_bonus:
            assert visitation_tracker is not None
            self.visitation_tracker = visitation_tracker

        self.actor = actor
        self.critic_ensemble = critic_ensemble
        self.value = value
        self.target_critic_ensemble = target_critic_ensemble
        self.rng = rng

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> jnp.ndarray:
        """
        Sample actions from the policy.

        Args:
            observations: the observations to sample actions for.
            temperature: factor that scales the policy std.

        Returns:
            the sampled actions
        """
        rng, actions = policy.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(
        self,
        batch: Batch,
        update_rnd: bool = True,
    ) -> InfoDict:
        """
        Update the policy, critic, and optionally RND networks.

        Args:
            batch: the batch of data to use for the update.
            update_rnd: whether to update the RND networks.

        Returns:
            a dictionary of metrics.
        """
        intrinsic_reward = None
        if self.use_rnd:
            if update_rnd:
                intrinsic_reward, _ = self.update_rnd(batch)
            else:
                intrinsic_reward = _calculate_rnd_reward_jit(
                    self.rnd_predictor,
                    self.rnd_target,
                    batch,
                    self.rnd_obs_rms.to_container(),
                    self.rnd_reward_rms.to_container(),
                )

        if self.use_count_based_bonus:
            intrinsic_reward = _calculate_state_action_count_bonus_jit(
                self.visitation_tracker.x,
                self.visitation_tracker.y,
                self.visitation_tracker.a1,
                self.visitation_tracker.a2,
                self.visitation_tracker.state_action_visitation_counts,
                batch,
            )

        (
            new_rng,
            new_actor,
            new_critic,
            new_value,
            new_target_critic,
            info,
        ) = _update_critic_and_actor_jit(
            self.rng,
            self.actor,
            self.critic_ensemble,
            self.value,
            self.target_critic_ensemble,
            batch,
            self.discount,
            self.tau,
            self.expectile,
            self.temperature,
            # reward configurations
            intrinsic_reward=intrinsic_reward,
            intrinsic_reward_scale=self.intrinsic_reward_scale,
            td3_update=self.td3_update,
            minimum_q=self.minimum_q,
            maximum_q=self.maximum_q,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic_ensemble = new_critic
        self.value = new_value
        self.target_critic_ensemble = new_target_critic

        return info

    def update_rnd(self, batch: Batch) -> Tuple[jnp.ndarray, InfoDict]:
        """
        Update the RND model exclusively

        Args:
            batch: the batch of data to use for the update.

        Returns:
            a tuple of the intrinsic reward and a dictionary of metrics.r
        """
        (
            new_rng,
            new_rnd_predictor,
            rnd_predictor_info,
        ) = _update_rnd_jit(
            self.rng,
            self.rnd_predictor,
            self.rnd_target,
            batch,
            self.rnd_obs_rms.to_container(),
            self.rnd_update_proportion,
        )
        self.rng = new_rng
        self.rnd_predictor = new_rnd_predictor

        intrinsic_reward = _calculate_rnd_reward_jit(
            self.rnd_predictor,
            self.rnd_target,
            batch,
            self.rnd_obs_rms.to_container(),
            self.rnd_reward_rms.to_container(),
        )

        # update running parameters
        self.rnd_reward_rms.update(intrinsic_reward)
        self.rnd_obs_rms.update(batch.observations)
        rnd_predictor_info["rnd_rewards_mean"] = self.rnd_reward_rms.mean
        rnd_predictor_info["rnd_rewards_var"] = self.rnd_reward_rms.var
        return intrinsic_reward, rnd_predictor_info

    def save_checkpoint(self, save_dir: str):
        print("Saving checkpoint to", save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = Path(save_dir)
        self.actor.save(save_path / "actor")
        self.critic_ensemble.save(save_path / "critic_ensemble")
        self.value.save(save_path / "value")
        self.target_critic_ensemble.save(save_path / "target_critic_ensemble")
        if self.use_rnd:
            self.rnd_predictor.save(save_path / "rnd_predictor")
            self.rnd_target.save(save_path / "rnd_target")
            self.rnd_obs_rms.save(save_path / "rnd_obs_rms")
            self.rnd_reward_rms.save(save_path / "rnd_reward_rms")

    def load_checkpoint(self, load_dir: str):
        print("Loading checkpoint from", load_dir)
        load_path = Path(load_dir)
        self.actor = self.actor.load(load_path / "actor")
        self.critic_ensemble = self.critic_ensemble.load(load_path / "critic_ensemble")
        self.value = self.value.load(load_path / "value")
        self.target_critic_ensemble = self.target_critic_ensemble.load(
            load_path / "target_critic_ensemble"
        )
        if self.use_rnd:
            self.rnd_predictor = self.rnd_predictor.load(load_path / "rnd_predictor")
            self.rnd_target = self.rnd_target.load(load_path / "rnd_target")
            self.rnd_obs_rms.load(load_path / "rnd_obs_rms")
            self.rnd_reward_rms.load(load_path / "rnd_reward_rms")

    def load_rnd_only(self, load_dir: str):
        print("Loading RND checkpoint from", load_dir)
        load_path = Path(load_dir)
        self.rnd_predictor = self.rnd_predictor.load(load_path / "rnd_predictor")
        self.rnd_target = self.rnd_target.load(load_path / "rnd_target")
        self.rnd_obs_rms.load(load_path / "rnd_obs_rms")
        self.rnd_reward_rms.load(load_path / "rnd_reward_rms")

    def copy_weights_from_other_agent(self, other_agent: Model):
        self.actor = other_agent.actor
        self.critic_ensemble = other_agent.critic_ensemble
        self.value = other_agent.value
        self.target_critic_ensemble = other_agent.target_critic_ensemble
        if self.use_rnd:
            self.rnd_predictor = other_agent.rnd_predictor
            self.rnd_target = other_agent.rnd_target
            self.rnd_obs_rms = other_agent.rnd_obs_rms
            self.rnd_reward_rms = other_agent.rnd_reward_rms
