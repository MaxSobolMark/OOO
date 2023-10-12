"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from rlpd.agents.agent import Agent
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.networks import (
    MLP,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    subsample_ensemble,
)
from rlpd.networks.rnd_net import (
    RNDNet,
    calculate_intrinsic_reward,
    RunningMeanStd,
    functional_running_mean_std_update,
)
from rlpd.networks.rnd_net import update_rnd as update_rnd_predictor


# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


class SACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)
    # RND variables
    use_rnd: bool
    rnd_predictor: TrainState
    rnd_target: TrainState
    rnd_obs_rms: Dict[str, jnp.ndarray]
    rnd_reward_rms: Dict[str, jnp.ndarray]
    rnd_update_proportion: float
    intrinsic_reward_scale: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        # RND parameters
        use_rnd: bool = False,
        rnd_hidden_dims: Sequence[int] = (512, 512),
        rnd_lr: float = 1e-4,
        rnd_update_proportion: float = 0.25,
        intrinsic_reward_scale: float = 1.0,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        (
            rng,
            actor_key,
            critic_key,
            temp_key,
            rnd_key,
            rnd_target_key,
        ) = jax.random.split(rng, 6)

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        # RND variables
        if use_rnd:
            rnd_def = RNDNet(rnd_hidden_dims)
            rnd_params = rnd_def.init(rnd_key, observations)["params"]
            rnd_predictor = TrainState.create(
                apply_fn=rnd_def.apply,
                params=rnd_params,
                tx=optax.adam(learning_rate=rnd_lr),
            )
            rnd_target_params = rnd_def.init(rnd_target_key, observations)["params"]
            rnd_target = TrainState.create(
                apply_fn=rnd_def.apply,
                params=rnd_target_params,
                tx=optax.GradientTransformation(lambda _: None, lambda _: None),
            )
            assert len(observations.shape) == 1
            rnd_obs_rms = RunningMeanStd(shape=observations.shape).to_container()
            rnd_reward_rms = RunningMeanStd().to_container()

        else:
            rnd_predictor = None
            rnd_target = None
            rnd_obs_rms = None
            rnd_reward_rms = None

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            # RND variables
            use_rnd=use_rnd,
            rnd_predictor=rnd_predictor,
            rnd_target=rnd_target,
            rnd_obs_rms=rnd_obs_rms,
            rnd_reward_rms=rnd_reward_rms,
            rnd_update_proportion=rnd_update_proportion,
            intrinsic_reward_scale=intrinsic_reward_scale,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(
        self,
        batch: DatasetDict,
        use_rnd: bool,
        intrinsic_rewards: Optional[jnp.ndarray] = None,
        max_q: Optional[float] = None,
        min_q: Optional[float] = None,
    ) -> Tuple[TrainState, Dict[str, float]]:
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q
        # Optionally clamp the target value.
        if max_q is not None:
            target_q = jnp.minimum(target_q, max_q)
        if min_q is not None:
            target_q = jnp.maximum(target_q, min_q)

        # RND intrinsic reward
        if use_rnd:
            assert (
                intrinsic_rewards is not None
                and self.intrinsic_reward_scale is not None
            )
            assert batch["rewards"].shape == intrinsic_rewards.shape
            target_q += self.intrinsic_reward_scale * intrinsic_rewards

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        if use_rnd:
            info["average_batch_intrinsic_rewards"] = intrinsic_rewards.mean()

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update_rnd(
        self, batch: DatasetDict
    ) -> Tuple[Agent, jnp.ndarray, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        new_rnd_predictor, rnd_predictor_info = update_rnd_predictor(
            key,
            self.rnd_predictor,
            self.rnd_target,
            batch,
            self.rnd_obs_rms,
            self.rnd_update_proportion,
        )
        intrinsic_reward = calculate_intrinsic_reward(
            self.rnd_predictor,
            self.rnd_target,
            batch["next_observations"],
            self.rnd_obs_rms,
            self.rnd_reward_rms,
        )

        # Update running parameters
        rnd_reward_rms = functional_running_mean_std_update(
            self.rnd_reward_rms, intrinsic_reward
        )
        rnd_obs_rms = functional_running_mean_std_update(
            self.rnd_obs_rms, batch["observations"]
        )
        rnd_predictor_info.update(
            {
                "rnd_rewards_mean": rnd_reward_rms["mean"],
                "rnd_rewards_var": rnd_reward_rms["var"],
                "rnd_obs_mean": rnd_obs_rms["mean"],
                "rnd_obs_var": rnd_obs_rms["var"],
            }
        )

        return (
            self.replace(
                rnd_predictor=new_rnd_predictor,
                rnd_reward_rms=rnd_reward_rms,
                rnd_obs_rms=rnd_obs_rms,
                rng=rng,
            ),
            intrinsic_reward,
            rnd_predictor_info,
        )

    @partial(
        jax.jit,
        static_argnames=("utd_ratio", "use_rnd", "update_rnd", "max_q", "min_q"),
    )
    def update(
        self,
        batch: DatasetDict,
        utd_ratio: int,
        use_rnd: bool = False,
        update_rnd: bool = True,
        max_q: Optional[float] = None,
        min_q: Optional[float] = None,
    ):
        new_agent = self
        intrinsic_reward = None
        rnd_info = {}
        batch_info = {
            "batch_rewards_mean": batch["rewards"].mean(),
            "batch_rewards_std": batch["rewards"].std(),
            "batch_rewards_max": batch["rewards"].max(),
            "batch_rewards_min": batch["rewards"].min(),
        }
        if use_rnd:
            if update_rnd:
                new_agent, intrinsic_reward, rnd_info = self.update_rnd(batch)
            else:
                intrinsic_reward = calculate_intrinsic_reward(
                    self.rnd_predictor,
                    self.rnd_target,
                    batch["next_observations"],
                    self.rnd_obs_rms,
                    self.rnd_reward_rms,
                )
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            mini_intr_reward = (
                None if intrinsic_reward is None else slice(intrinsic_reward)
            )
            new_agent, critic_info = new_agent.update_critic(
                mini_batch, use_rnd, mini_intr_reward, max_q, min_q
            )

        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {
            **batch_info,
            **actor_info,
            **critic_info,
            **rnd_info,
            **temp_info,
        }
