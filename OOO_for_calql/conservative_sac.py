######################################################
#                 conservative_sac.py                #
#             Implements CQL and Cal-QL              #
######################################################

from copy import deepcopy
from functools import partial
from ml_collections import ConfigDict
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from jax.flatten_util import ravel_pytree

from jax_utils import (
    next_rng,
    value_and_multi_grad,
    mse_loss,
    JaxRNG,
    wrap_function_with_rng,
    collect_jax_metrics,
)
from model import Scalar, update_target_network
from rnd_net import (
    RNDNet,
    calculate_intrinsic_reward,
    RunningMeanStd,
    functional_running_mean_std_update,
)
from rnd_net import update_rnd as update_rnd_predictor


class ConservativeSAC(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 1e-4
        config.qf_lr = 3e-4
        config.optimizer_type = "adam"
        config.soft_target_update_rate = 5e-3
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_max_target_backup = True
        config.cql_clip_diff_min = -np.inf
        config.cql_clip_diff_max = np.inf
        # RND configs
        config.use_rnd = False
        config.add_intrinsic_rewards_to_mc_returns = True
        config.intrinsic_rewards_for_offline_dataset = True
        config.intrinsic_mc_recalculation_frequency = 10  # in epochs
        config.intrinsic_discount = 0.99
        config.rnd_lr = 1e-4
        config.rnd_update_proportion = 0.25
        config.rnd_reward_scale = 1.0
        config.rnd_hidden_dims = (512, 512)
        # Q-function extra configs
        config.bound_q_functions = False
        config.calculate_q_bounds_from_data = False
        config.min_reward = -np.inf
        config.max_reward = np.inf
        config.feature_normalization = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config, policy, qf):
        self.config = self.get_default_config(config)
        self.policy = policy
        self.qf = qf
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim

        self._train_states = {}

        optimizer_class = {
            "adam": optax.adam,
            "sgd": optax.sgd,
        }[self.config.optimizer_type]

        policy_params = self.policy.init(
            next_rng(self.policy.rng_keys()), jnp.zeros((10, self.observation_dim))
        )

        self._train_states["policy"] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=None,
        )

        qf1_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim)),
        )

        qf2_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim)),
        )

        self._train_states["qf1"] = TrainState.create(
            params=qf1_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        self._train_states["qf2"] = TrainState.create(
            params=qf2_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )

        # RND networks
        if self.config.use_rnd:
            rnd_key = next_rng()
            rnd_target_key = next_rng()
            rnd_def = RNDNet(hidden_dims=self.config.rnd_hidden_dims)
            rnd_params = rnd_def.init(rnd_key, jnp.zeros((self.observation_dim,)))[
                "params"
            ]
            rnd_predictor = TrainState.create(
                apply_fn=rnd_def.apply,
                params=rnd_params,
                tx=optimizer_class(self.config.rnd_lr),
            )
            rnd_target_params = rnd_def.init(
                rnd_target_key, jnp.zeros((self.observation_dim,))
            )["params"]
            rnd_target = TrainState.create(
                apply_fn=rnd_def.apply,
                params=rnd_target_params,
                tx=optax.GradientTransformation(lambda _: None, lambda _: None),
            )
            rnd_obs_rms = RunningMeanStd(shape=(self.observation_dim,)).to_container()
            rnd_reward_rms = RunningMeanStd().to_container()
            self._train_states["rnd_predictor"] = rnd_predictor
            self._train_states["rnd_target"] = rnd_target
            self._train_states["rnd_obs_rms"] = rnd_obs_rms
            self._train_states["rnd_reward_rms"] = rnd_reward_rms

        self._target_qf_params = deepcopy({"qf1": qf1_params, "qf2": qf2_params})
        model_keys = ["policy", "qf1", "qf2"]

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self._train_states["log_alpha"] = TrainState.create(
                params=self.log_alpha.init(next_rng()),
                tx=optimizer_class(self.config.policy_lr),
                apply_fn=None,
            )
            model_keys.append("log_alpha")

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self._train_states["log_alpha_prime"] = TrainState.create(
                params=self.log_alpha_prime.init(next_rng()),
                tx=optimizer_class(self.config.qf_lr),
                apply_fn=None,
            )
            model_keys.append("log_alpha_prime")

        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def train(
        self,
        batch,
        use_cql=True,
        cql_min_q_weight=5.0,
        enable_calql=False,
        bound_q_functions=False,
        min_q=-np.inf,
        max_q=np.inf,
    ):
        self._total_steps += 1
        self._train_states, self._target_qf_params, metrics = self._train_step(
            self._train_states,
            self._target_qf_params,
            next_rng(),
            batch,
            use_cql,
            cql_min_q_weight,
            enable_calql,
            bound_q_functions=bound_q_functions,
            min_q=min_q,
            max_q=max_q,
        )
        return metrics

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "use_cql",
            "cql_min_q_weight",
            "enable_calql",
            "bound_q_functions",
            "min_q",
            "max_q",
        ),
    )
    def _train_step(
        self,
        train_states,
        target_qf_params,
        rng,
        batch,
        use_cql=True,
        cql_min_q_weight=5.0,
        enable_calql=False,
        bound_q_functions=False,
        min_q=None,
        max_q=None,
    ):
        rng_generator = JaxRNG(rng)

        # Update RND
        if self.config.use_rnd:
            key, rng = jax.random.split(rng)
            new_rnd_predictor, rnd_predictor_info = update_rnd_predictor(
                key,
                self._train_states["rnd_predictor"],
                self._train_states["rnd_target"],
                batch["next_observations"],
                self._train_states["rnd_obs_rms"],
                self.config.rnd_update_proportion,
            )
            intrinsic_reward = calculate_intrinsic_reward(
                self._train_states["rnd_predictor"],
                self._train_states["rnd_target"],
                batch["next_observations"],
                self._train_states["rnd_obs_rms"],
                self._train_states["rnd_reward_rms"],
            )

            # Update running parameters
            new_rnd_reward_rms = functional_running_mean_std_update(
                self._train_states["rnd_reward_rms"], intrinsic_reward
            )
            new_rnd_obs_rms = functional_running_mean_std_update(
                self._train_states["rnd_obs_rms"], batch["observations"]
            )
        else:
            intrinsic_reward = jnp.zeros_like(batch["rewards"])

        def loss_fn(train_params):
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_observations = batch["next_observations"]
            dones = batch["dones"]

            loss_collection = {}

            @wrap_function_with_rng(rng_generator())
            def forward_policy(rng, *args, **kwargs):
                return self.policy.apply(
                    *args, **kwargs, rngs=JaxRNG(rng)(self.policy.rng_keys())
                )

            @wrap_function_with_rng(rng_generator())
            def forward_qf(rng, *args, **kwargs):
                return self.qf.apply(
                    *args,
                    **kwargs,
                    rngs=JaxRNG(rng)(self.qf.rng_keys()),
                )

            log_pi_data = forward_policy(
                train_params["policy"],
                observations,
                actions,
                method=self.policy.log_prob,
            )
            new_actions, log_pi = forward_policy(train_params["policy"], observations)

            if self.config.use_automatic_entropy_tuning:
                alpha_loss = (
                    -self.log_alpha.apply(train_params["log_alpha"])
                    * (log_pi + self.config.target_entropy).mean()
                )
                loss_collection["log_alpha"] = alpha_loss
                alpha = (
                    jnp.exp(self.log_alpha.apply(train_params["log_alpha"]))
                    * self.config.alpha_multiplier
                )
            else:
                alpha_loss = 0.0
                alpha = self.config.alpha_multiplier

            """ Policy loss """
            q_new_actions = jnp.minimum(
                forward_qf(train_params["qf1"], observations, new_actions)[0],
                forward_qf(train_params["qf2"], observations, new_actions)[0],
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()

            loss_collection["policy"] = policy_loss

            """ Q function loss """
            q1_pred, q1_penultimate_layer_activations = forward_qf(
                train_params["qf1"],
                observations,
                actions,
            )
            assert q1_penultimate_layer_activations.shape == (
                observations.shape[0],
                int(self.qf.arch.split("-")[-1]),
            )
            q2_pred, q2_penultimate_layer_activations = forward_qf(
                train_params["qf2"],
                observations,
                actions,
            )

            # get penultimate_layer_activations for next_observations, next_actions
            _, q1_penultimate_layer_activations_next = forward_qf(
                train_params["qf1"],
                next_observations,
                batch["next_actions"],
            )
            _, q2_penultimate_layer_activations_next = forward_qf(
                train_params["qf2"],
                next_observations,
                batch["next_actions"],
            )

            # Calculate dot products between current and next activations
            q1_penultimate_layer_activations_dot = jnp.einsum(
                "ij,ij->i",
                q1_penultimate_layer_activations,
                q1_penultimate_layer_activations_next,
            )
            q2_penultimate_layer_activations_dot = jnp.einsum(
                "ij,ij->i",
                q2_penultimate_layer_activations,
                q2_penultimate_layer_activations_next,
            )

            average_penultimate_layer_activation_dot = (
                q1_penultimate_layer_activations_dot
                + q2_penultimate_layer_activations_dot
            ) / 2.0

            if self.config.cql_max_target_backup:
                new_next_actions, next_log_pi = forward_policy(
                    train_params["policy"],
                    next_observations,
                    repeat=self.config.cql_n_actions,
                )
                target_q_values = jnp.minimum(
                    forward_qf(
                        target_qf_params["qf1"], next_observations, new_next_actions
                    )[0],
                    forward_qf(
                        target_qf_params["qf2"], next_observations, new_next_actions
                    )[0],
                )
                max_target_indices = jnp.expand_dims(
                    jnp.argmax(target_q_values, axis=-1), axis=-1
                )
                target_q_values = jnp.take_along_axis(
                    target_q_values, max_target_indices, axis=-1
                ).squeeze(-1)
                next_log_pi = jnp.take_along_axis(
                    next_log_pi, max_target_indices, axis=-1
                ).squeeze(-1)
            else:
                new_next_actions, next_log_pi = forward_policy(
                    train_params["policy"], next_observations
                )
                target_q_values = jnp.minimum(
                    forward_qf(
                        target_qf_params["qf1"], next_observations, new_next_actions
                    )[0],
                    forward_qf(
                        target_qf_params["qf2"], next_observations, new_next_actions
                    )[0],
                )

            # Bound the Q target
            if bound_q_functions:
                assert min_q is not None or max_q is not None
                if min_q is not None:
                    target_q_values = jnp.maximum(target_q_values, min_q)
                if max_q is not None:
                    target_q_values = jnp.minimum(target_q_values, max_q)
            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            assert intrinsic_reward.shape == rewards.shape
            td_target = jax.lax.stop_gradient(
                intrinsic_reward * self.config.rnd_reward_scale
                + rewards
                + (1.0 - dones) * self.config.discount * target_q_values
            )
            qf1_bellman_loss = mse_loss(q1_pred, td_target)
            qf2_bellman_loss = mse_loss(q2_pred, td_target)

            # CQL
            if use_cql:
                batch_size = actions.shape[0]
                cql_random_actions = jax.random.uniform(
                    rng_generator(),
                    shape=(batch_size, self.config.cql_n_actions, self.action_dim),
                    minval=-1.0,
                    maxval=1.0,
                )

                cql_current_actions, cql_current_log_pis = forward_policy(
                    train_params["policy"],
                    observations,
                    repeat=self.config.cql_n_actions,
                )
                cql_next_actions, cql_next_log_pis = forward_policy(
                    train_params["policy"],
                    next_observations,
                    repeat=self.config.cql_n_actions,
                )

                cql_q1_rand = forward_qf(
                    train_params["qf1"], observations, cql_random_actions
                )[0]
                cql_q2_rand = forward_qf(
                    train_params["qf2"], observations, cql_random_actions
                )[0]
                cql_q1_current_actions = forward_qf(
                    train_params["qf1"], observations, cql_current_actions
                )[0]
                cql_q2_current_actions = forward_qf(
                    train_params["qf2"], observations, cql_current_actions
                )[0]
                cql_q1_next_actions = forward_qf(
                    train_params["qf1"], observations, cql_next_actions
                )[0]
                cql_q2_next_actions = forward_qf(
                    train_params["qf2"], observations, cql_next_actions
                )[0]

                """ Cal-QL: prepare for Cal-QL, and calculate how much data will be lower bounded for logging """
                mc_returns = batch["mc_returns"]
                if (
                    self.config.use_rnd
                    and self.config.add_intrinsic_rewards_to_mc_returns
                ):
                    assert (
                        batch["intrinsic_mc_returns"].shape == batch["mc_returns"].shape
                    )
                    mc_returns += batch["intrinsic_mc_returns"]
                lower_bounds = jnp.repeat(
                    batch["mc_returns"].reshape(-1, 1),
                    cql_q1_current_actions.shape[1],
                    axis=1,
                )
                num_vals = jnp.sum(lower_bounds == lower_bounds)
                bound_rate_cql_q1_current_actions = (
                    jnp.sum(cql_q1_current_actions < lower_bounds) / num_vals
                )
                bound_rate_cql_q2_current_actions = (
                    jnp.sum(cql_q2_current_actions < lower_bounds) / num_vals
                )
                bound_rate_cql_q1_next_actions = (
                    jnp.sum(cql_q1_next_actions < lower_bounds) / num_vals
                )
                bound_rate_ql_q2_next_actions = (
                    jnp.sum(cql_q2_next_actions < lower_bounds) / num_vals
                )

                """ Cal-QL: bound Q-values with MC return-to-go """
                if enable_calql:
                    cql_q1_current_actions = jnp.maximum(
                        cql_q1_current_actions, lower_bounds
                    )
                    cql_q2_current_actions = jnp.maximum(
                        cql_q2_current_actions, lower_bounds
                    )
                    cql_q1_next_actions = jnp.maximum(cql_q1_next_actions, lower_bounds)
                    cql_q2_next_actions = jnp.maximum(cql_q2_next_actions, lower_bounds)
                    cql_q1_rand = jnp.maximum(cql_q1_rand, lower_bounds)
                    cql_q2_rand = jnp.maximum(cql_q2_rand, lower_bounds)

                cql_cat_q1 = jnp.concatenate(
                    [
                        cql_q1_rand,
                        jnp.expand_dims(q1_pred, 1),
                        cql_q1_next_actions,
                        cql_q1_current_actions,
                    ],
                    axis=1,
                )
                cql_cat_q2 = jnp.concatenate(
                    [
                        cql_q2_rand,
                        jnp.expand_dims(q2_pred, 1),
                        cql_q2_next_actions,
                        cql_q2_current_actions,
                    ],
                    axis=1,
                )
                cql_std_q1 = jnp.std(cql_cat_q1, axis=1)
                cql_std_q2 = jnp.std(cql_cat_q2, axis=1)

                if self.config.cql_importance_sample:
                    random_density = np.log(0.5**self.action_dim)
                    cql_cat_q1 = jnp.concatenate(
                        [
                            cql_q1_rand - random_density,
                            cql_q1_next_actions - cql_next_log_pis,
                            cql_q1_current_actions - cql_current_log_pis,
                        ],
                        axis=1,
                    )
                    cql_cat_q2 = jnp.concatenate(
                        [
                            cql_q2_rand - random_density,
                            cql_q2_next_actions - cql_next_log_pis,
                            cql_q2_current_actions - cql_current_log_pis,
                        ],
                        axis=1,
                    )

                cql_qf1_ood = (
                    jax.scipy.special.logsumexp(
                        cql_cat_q1 / self.config.cql_temp, axis=1
                    )
                    * self.config.cql_temp
                )
                cql_qf2_ood = (
                    jax.scipy.special.logsumexp(
                        cql_cat_q2 / self.config.cql_temp, axis=1
                    )
                    * self.config.cql_temp
                )

                """Subtract the log likelihood of data"""
                q1_pred_to_maximize = q1_pred
                cql_qf1_diff = jnp.clip(
                    cql_qf1_ood - q1_pred_to_maximize,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()
                q2_pred_to_maximize = q2_pred
                cql_qf2_diff = jnp.clip(
                    cql_qf2_ood - q2_pred_to_maximize,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()

                if self.config.cql_lagrange:
                    alpha_prime = jnp.clip(
                        jnp.exp(
                            self.log_alpha_prime.apply(train_params["log_alpha_prime"])
                        ),
                        a_min=0.0,
                        a_max=1000000.0,
                    )
                    cql_min_qf1_loss = (
                        alpha_prime
                        * cql_min_q_weight
                        * (cql_qf1_diff - self.config.cql_target_action_gap)
                    )
                    cql_min_qf2_loss = (
                        alpha_prime
                        * cql_min_q_weight
                        * (cql_qf2_diff - self.config.cql_target_action_gap)
                    )

                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5

                    loss_collection["log_alpha_prime"] = alpha_prime_loss

                else:
                    cql_min_qf1_loss = cql_qf1_diff * cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * cql_min_q_weight
                    alpha_prime_loss = 0.0
                    alpha_prime = 0.0

                qf1_loss = qf1_bellman_loss + cql_min_qf1_loss
                qf2_loss = qf2_bellman_loss + cql_min_qf2_loss
            else:
                # when use_cql = False
                qf1_loss = qf1_bellman_loss
                qf2_loss = qf2_bellman_loss

            loss_collection["qf1"] = qf1_loss
            loss_collection["qf2"] = qf2_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(
            loss_fn, len(self.model_keys), has_aux=True
        )(train_params)
        policy_loss_gradient = jnp.linalg.norm(
            ravel_pytree(grads[self.model_keys.index("policy")]["policy"])[0]
        )
        qf1_loss_gradient = jnp.linalg.norm(
            ravel_pytree(grads[self.model_keys.index("qf1")]["qf1"])[0]
        )
        qf2_loss_gradient = jnp.linalg.norm(
            ravel_pytree(grads[self.model_keys.index("qf2")]["qf2"])[0]
        )

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }
        new_target_qf_params = {}
        new_target_qf_params["qf1"] = update_target_network(
            new_train_states["qf1"].params,
            target_qf_params["qf1"],
            self.config.soft_target_update_rate,
        )
        new_target_qf_params["qf2"] = update_target_network(
            new_train_states["qf2"].params,
            target_qf_params["qf2"],
            self.config.soft_target_update_rate,
        )

        metrics = collect_jax_metrics(
            aux_values,
            [
                "log_pi",
                "policy_loss",
                "qf1_loss",
                "qf2_loss",
                "alpha_loss",
                "alpha",
                "q1_pred",
                "q2_pred",
                "target_q_values",
                "policy_loss_gradient",
                "qf1_loss_gradient",
                "qf2_loss_gradient",
                "average_penultimate_layer_activation_dot",
            ],
        )
        metrics.update(
            policy_loss_gradient=policy_loss_gradient,
            qf1_loss_gradient=qf1_loss_gradient,
            qf2_loss_gradient=qf2_loss_gradient,
        )
        metrics.update(
            use_cql=int(use_cql),
            enable_calql=int(enable_calql),
            cql_min_q_weight=cql_min_q_weight,
        )
        if self.config.use_rnd:
            metrics.update(rnd_predictor_info)
            new_train_states["rnd_predictor"] = new_rnd_predictor
            new_train_states["rnd_target"] = self._train_states["rnd_target"]
            new_train_states["rnd_reward_rms"] = new_rnd_reward_rms
            new_train_states["rnd_obs_rms"] = new_rnd_obs_rms

        if use_cql:
            metrics.update(
                collect_jax_metrics(
                    aux_values,
                    [
                        "cql_std_q1",
                        "cql_std_q2",
                        "cql_q1_rand",
                        "cql_q2_rand",
                        "cql_qf1_diff",
                        "cql_qf2_diff",
                        "cql_min_qf1_loss",
                        "cql_min_qf2_loss",
                        "cql_q1_current_actions",
                        "cql_q2_current_actions" "cql_q1_next_actions",
                        "cql_q2_next_actions",
                        "alpha_prime",
                        "alpha_prime_loss",
                        "qf1_bellman_loss",
                        "qf2_bellman_loss",
                        "bound_rate_cql_q1_current_actions",
                        "bound_rate_cql_q2_current_actions",
                        "bound_rate_cql_q1_next_actions",
                        "bound_rate_ql_q2_next_actions",
                        "log_pi_data",
                    ],
                    "cql",
                )
            )
        if self.config.use_rnd and self.config.add_intrinsic_rewards_to_mc_returns:
            metrics.update(
                intrinsic_mc_returns_mean=batch["intrinsic_mc_returns"].mean(),
                intrinsic_mc_returns_std=batch["intrinsic_mc_returns"].std(),
                intrinsic_mc_returns_max=batch["intrinsic_mc_returns"].max(),
                intrinsic_mc_returns_min=batch["intrinsic_mc_returns"].min(),
            )

        return new_train_states, new_target_qf_params, metrics

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
