from typing import Dict

import gym
import numpy as np

from rlpd.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    binary_success = []
    max_reward = []
    for i in range(num_episodes):
        max_reward.append(np.NINF)
        binary_goal_achived = False
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, reward, done, info = env.step(action)
            # assert reward.shape == ()
            if "goal_achieved" in info and info["goal_achieved"]:
                binary_goal_achived = True
            max_reward[-1] = max(max_reward[-1], reward)
        binary_success.append(binary_goal_achived)
    return {
        "return": np.mean(env.return_queue),
        "length": np.mean(env.length_queue),
        "binary_success": np.mean(binary_success),
        "max_reward": np.mean(max_reward),
    }
