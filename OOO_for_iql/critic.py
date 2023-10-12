######################################################
#                      critic.py                     #
#        functions to update Q and V functions       #
######################################################

from typing import Tuple, Optional
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(
    critic: Model,
    value: Model,
    batch: Batch,
    expectile: float,
    minimum_q: Optional[float] = None,
    maximum_q: Optional[float] = None,
) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    qs = critic(batch.observations, actions)
    # If min/max q values are specified, clip the target value
    if minimum_q is not None:
        qs = jnp.maximum(qs, minimum_q)
    if maximum_q is not None:
        qs = jnp.minimum(qs, maximum_q)
    assert len(qs.shape) == 2 and qs.shape[1] == batch.actions.shape[0]
    # TODO: Change this to correct for an ensembling version of IQL
    q1, q2 = qs[0], qs[1]
    q = jnp.minimum(q1, q2)
    assert q.shape == (batch.observations.shape[0],)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({"params": value_params}, batch.observations)
        value_loss = loss(q - v, expectile).mean()

        return value_loss, {
            "value_loss": value_loss,
            "value": v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)
    return new_value, info


def update_q(
    critic: Model,
    target_value: Model,
    batch: Batch,
    discount: float,
    intrinsic_rewards: Optional[jnp.ndarray] = None,
    intrinsic_rewards_scale: Optional[float] = None,
    minimum_q: Optional[float] = None,
    maximum_q: Optional[float] = None,
) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v
    # If min/max q values are specified, clip the target value
    if minimum_q is not None:
        target_q = jnp.maximum(target_q, minimum_q)
    if maximum_q is not None:
        target_q = jnp.minimum(target_q, maximum_q)
    if intrinsic_rewards is not None:
        assert batch.rewards.shape == intrinsic_rewards.shape == next_v.shape
        assert intrinsic_rewards_scale is not None
        target_q += intrinsic_rewards * intrinsic_rewards_scale

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        assert qs.shape[1:] == target_q.shape and len(qs.shape) == 2

        td_errors = qs - target_q
        assert td_errors.shape == qs.shape
        # switch from sum to mean
        # follows: https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/critic.py#L41)
        critic_loss = (td_errors**2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": qs[0].mean(),
            "q2": qs[1].mean(),
            "qs_mean": qs.mean(),
            "qs_std": qs.std(),
            "target_q": target_q.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    if intrinsic_rewards is not None:
        info["average_batch_intrinsic_rewards"] = intrinsic_rewards.mean()
    return new_critic, info


def td3_update_q(
    key: PRNGKey,
    critic: Model,
    target_critic: Model,
    actor: Model,
    batch: Batch,
    discount: float,
    intrinsic_rewards: Optional[jnp.ndarray] = None,
    intrinsic_rewards_scale: Optional[float] = None,
) -> Tuple[Model, InfoDict]:
    next_actions = actor(batch.next_observations).sample(seed=key)
    next_v = jnp.min(target_critic(batch.next_observations, next_actions), axis=0)

    # compute target values
    assert next_v.shape == batch.rewards.shape
    target_q = batch.rewards + discount * batch.masks * next_v
    if intrinsic_rewards is not None:
        assert batch.rewards.shape == intrinsic_rewards.shape
        assert intrinsic_rewards_scale is not None
        target_q += intrinsic_rewards * intrinsic_rewards_scale

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        assert qs.shape[1:] == target_q.shape and len(qs.shape) == 2

        td_errors = qs - target_q
        assert td_errors.shape == qs.shape
        # switch from sum to mean
        # follows: https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/critic.py#L41)
        critic_loss = (td_errors**2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": qs[0].mean(),
            "q2": qs[1].mean(),
            "qs_mean": qs.mean(),
            "qs_std": qs.std(),
            "target_q": target_q.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    if intrinsic_rewards is not None:
        info["average_batch_intrinsic_rewards"] = intrinsic_rewards.mean()

    return new_critic, info
