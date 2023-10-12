from typing import Tuple

import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey


def awr_policy_update(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    batch: Batch,
    temperature: float,
) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)
    qs = critic(batch.observations, batch.actions)
    assert len(qs.shape) == 2 and qs.shape[1] == batch.observations.shape[0]

    q1, q2 = qs[0], qs[1]
    q = jnp.minimum(q1, q2)
    assert q.shape == (batch.observations.shape[0],)

    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply(
            {"params": actor_params},
            batch.observations,
            training=True,
            rngs={"dropout": key},
        )
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss, {
            "actor_loss": actor_loss,
            "adv": q - v,
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def td3_policy_update(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    batch: Batch,
) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply(
            {"params": actor_params},
            batch.observations,
            training=True,
            rngs={"dropout": key},
        )
        sampled_actions = dist.sample(seed=key)
        critic_preds = critic(batch.observations, sampled_actions)

        actor_loss = -critic_preds.mean()

        return actor_loss, {
            "actor_loss": actor_loss,
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
