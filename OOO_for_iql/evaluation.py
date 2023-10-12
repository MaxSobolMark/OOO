from typing import Dict
import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {
        "return": [],
        "length": [],
        "actual_return": [],
        "binary_success_rate": [],
        "max_reward": [],
    }

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        episode_return = 0.0
        binary_goal_achived = False
        max_reward = np.NINF
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, rew, done, info = env.step(action)
            episode_return += rew
            if "goal_achieved" in info and info["goal_achieved"]:
                binary_goal_achived = True
            max_reward = max(max_reward, rew)

        for k in ["return", "length"]:
            stats[k].append(info["episode"][k])
        stats["actual_return"].append(episode_return)
        stats["binary_success_rate"].append(float(binary_goal_achived))
        stats["max_reward"].append(max_reward)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
