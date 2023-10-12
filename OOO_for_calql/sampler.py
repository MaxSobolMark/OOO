######################################################
#                     sampler.py                     #
#        Implements the trajectory collector         #
#       for evaluations and online experiences       #
######################################################

from typing import Dict, Optional
import numpy as np
import jax.numpy as jnp
from replay_buffer import calc_return_to_go
from flax.training.train_state import TrainState
from rnd_net import calculate_intrinsic_reward


class TrajSampler(object):
    def __init__(
        self,
        env,
        use_goal=False,
        use_mc=False,
        gamma=0.99,
        reward_scale=1.0,
        reward_bias=0.0,
        use_intrinsic_mc=False,
    ):
        self._env = env
        self.use_goal = use_goal
        self.use_mc = use_mc
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.max_traj_length = env.spec.max_episode_steps
        self.use_intrinsic_mc = use_intrinsic_mc

    def sample(
        self,
        policy,
        n_trajs,
        deterministic=False,
        replay_buffer=None,
        rnd_predictor: Optional[TrainState] = None,
        rnd_target: Optional[TrainState] = None,
        obs_mean_std: Optional[Dict[str, jnp.ndarray]] = None,
        rewards_running_mean_std: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            if self.use_goal:
                goal_achieved_list = []

            observation = self.env.reset()

            for _ in range(self.max_traj_length):
                action = policy(
                    observation.reshape(1, -1), deterministic=deterministic
                ).reshape(-1)
                next_observation, reward, done, env_infos = self.env.step(action)
                if self.use_goal:
                    goal_achieved = 1 if env_infos["goal_achieved"] else 0
                    goal_achieved_list.append(goal_achieved)
                    if env_infos["goal_achieved"]:
                        done = True  # terminate the episode when goal is achieved in Adroit envs

                observations.append(observation)
                actions.append(action)
                rewards.append(reward * self.reward_scale + self.reward_bias)

                dones.append(done)
                next_observations.append(next_observation)
                observation = next_observation

                if done:
                    break

            if self.use_mc:
                """
                Calculate Monte Carlo returns for Cal-QL
                """
                if self.use_intrinsic_mc:
                    intrinsic_rewards = calculate_intrinsic_reward(
                        predictor=rnd_predictor,
                        target=rnd_target,
                        next_observations=np.array(next_observations),
                        observations_running_mean_std=obs_mean_std,
                        rewards_running_mean_std=rewards_running_mean_std,
                    )
                    intrinsic_mc_returns = calc_return_to_go(
                        self.env.spec.name,
                        intrinsic_rewards,
                        dones,
                        self.gamma,
                        self.reward_scale,
                        self.reward_bias,
                        is_sparse_reward=False,
                        infinite_horizon=False,
                    )
                if "antmaze" in self.env.spec.name or "maze2d" in self.env.spec.name:
                    mc_returns = calc_return_to_go(
                        self.env.spec.name,
                        rewards,
                        dones,
                        self.gamma,
                        self.reward_scale,
                        self.reward_bias,
                        is_sparse_reward=True,
                        infinite_horizon=False,
                    )
                elif self.env.spec.name in [
                    "pen-binary-v0",
                    "door-binary-v0",
                    "relocate-binary-v0",
                    "pen-binary",
                    "door-binary",
                    "relocate-binary",
                ]:
                    mc_returns = calc_return_to_go(
                        self.env.spec.name,
                        rewards,
                        dones,
                        self.gamma,
                        self.reward_scale,
                        self.reward_bias,
                        is_sparse_reward=True,
                        infinite_horizon=False,
                    )
                elif "kitchen" in self.env.spec.name:
                    mc_returns = calc_return_to_go(
                        self.env.spec.name,
                        rewards,
                        dones,
                        self.gamma,
                        self.reward_scale,
                        self.reward_bias,
                        is_sparse_reward=False,
                        infinite_horizon=True,
                    )
                elif "hammer" in self.env.spec.name:
                    mc_returns = calc_return_to_go(
                        self.env.spec.name,
                        rewards,
                        dones,
                        self.gamma,
                        self.reward_scale,
                        self.reward_bias,
                        is_sparse_reward=False,
                        infinite_horizon=False,
                    )
                else:
                    # mc_returns = calc_return_to_go(self.env.spec.name, rewards_unscaled, dones, self.gamma, self.reward_scale, self.reward_bias, is_sparse_reward=False)
                    """
                    if your new env has dense rewards, uncomment the above line will be fine
                    if your new env has sparse rewards, please check calc_return_to_go() in replay_buffer.py
                    """
                    raise NotImplementedError

            if replay_buffer is not None:
                for i in range(len(rewards)):
                    if self.use_mc:
                        replay_buffer.add_sample(
                            observations[i],
                            actions[i],
                            rewards[i],
                            next_observations[i],
                            dones[i],
                            mc_returns[i],
                            intrinsic_mc_returns=None
                            if not self.use_intrinsic_mc
                            else intrinsic_mc_returns[i],
                        )
                    else:
                        replay_buffer.add_sample(
                            observations[i],
                            actions[i],
                            rewards[i],
                            next_observations[i],
                            dones[i],
                        )

            traj_dict = dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
            )

            if self.use_mc:
                traj_dict.update(
                    dict(
                        mc_returns=np.array(mc_returns, dtype=np.float32),
                    )
                )

            if self.use_goal:
                traj_dict.update(
                    dict(goal_achieved=np.array(goal_achieved_list, dtype=np.float32))
                )

            trajs.append(traj_dict)

        return trajs

    @property
    def env(self):
        return self._env
