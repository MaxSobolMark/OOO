#####################################################  #
#                  point_mass_wall.py                  #
# Implements the environment from the didactic example #
########################################################


from typing import List, Tuple
import math
import numpy as np
from gym.spaces import Box
import unittest


def sparse_reward_function(
    current_position: np.ndarray,
    goal_position: np.ndarray,
    epsilon: float,
    goal_reaching_reward: float = 1.0,
) -> float:
    if np.linalg.norm(current_position - goal_position) < epsilon:
        return goal_reaching_reward
    return 0.0


def steps_to_goal_reward_function(
    current_position: np.ndarray,
    goal_position: np.ndarray,
    epsilon: float,
) -> float:
    if np.linalg.norm(current_position - goal_position) < epsilon:
        return 0.0
    return -1.0


# Methods to check segment intersection
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class PointMassEnv:
    def __init__(
        self,
        expose_velocity=False,
        expose_goal=False,
        max_dx_dy=math.sqrt(1 / 2) * 0.05,
        epsilon=0.01,
        goal_sequence: List[Tuple[float, float]] = [(1, 0.15)],
        add_noise_to_initial_position=True,
        initial_position=(0, 0.5),
        initial_position_noise_std=0.02,
        terminate_on_goal_reaching=True,
        reward_function=sparse_reward_function,
    ):
        self._initial_goal_sequence = goal_sequence
        self.goal_sequence = self._initial_goal_sequence.copy()
        self.goal = self.goal_sequence.pop(0)
        self._epsilon = epsilon

        self._expose_velocity = expose_velocity
        self._expose_goal = expose_goal
        self._max_dx_dy = max_dx_dy
        self._add_noise_to_initial_position = add_noise_to_initial_position
        self._initial_position = np.array(initial_position)
        self._initial_position_noise_std = initial_position_noise_std
        self._reward_function = reward_function
        self._terminate_on_goal_reaching = terminate_on_goal_reaching

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 + int(self._expose_velocity) * 2 + int(self._expose_goal) * 2,),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        self.position = None

    def step(self, action):
        dx = action[0] * self._max_dx_dy
        dy = action[1] * self._max_dx_dy
        qpos = self.sim.data.qpos.flat.copy()
        qpos[0] = np.clip(qpos[0] + dx, -0.25, 1.25)
        qpos[1] = np.clip(qpos[1] + dy, -0.25, 1.45)
        qvel = self.sim.data.qvel.flat.copy()
        self.set_state(qpos, qvel)

        ob = self._get_obs()
        """TODO: implement new reward functions:
                 - distance to goal
                 - distance to goal + increasing bonus when close to the goal (meta world style)
                 - sparse reward function"""
        # reward = -np.linalg.norm(self.sim.data.qpos.flat[:2] - self.goal)
        reward = self._reward_function(
            self.sim.data.qpos.flat[:2], self.goal, epsilon=self._epsilon
        )

        if np.linalg.norm(self.sim.data.qpos.flat[:2] - self.goal) < self._epsilon:
            if len(self.goal_sequence) == 0:
                done = True
                # print("reached last goal")
            else:
                # print("reached goal")
                # print(
                #     f"Current position: {self.sim.data.qpos.flat[:2]}. Goal: {self.goal}"
                # )
                if self.goal_sequence[0][0] == "DONE_FLAG":
                    done = True
                else:
                    self.goal = self.goal_sequence.pop(0)
                    done = False
        else:
            done = False
        if not self._terminate_on_goal_reaching:
            done = False

        return (
            ob,
            reward,
            done,
            {},
        )

    def _get_obs(self):
        new_obs = [self.position]
        if self._expose_velocity:
            new_obs += [self.sim.data.qvel.flat]
        if self._expose_goal and self.goal is not None:
            new_obs += [self.goal]
        return np.concatenate(new_obs)

    def reset(self):
        init_qpos = self._initial_position
        if len(self.goal_sequence) > 0:
            self.goal = self.goal_sequence.pop(0)
        else:
            self.goal_sequence = self._initial_goal_sequence.copy()
            self.goal = self.goal_sequence.pop(0)
        self.position = init_qpos + (
            self.np_random.uniform(
                low=-self._initial_position_noise_std,
                high=self._initial_position_noise_std,
                size=2,
            )
            if self._add_noise_to_initial_position
            else np.zeros(2),
        )

        return self._get_obs()


class PointMassWallEnv(PointMassEnv):
    def __init__(
        self,
        wall_coordinates=[[0.5, 0.1, 0.5, 1.2]],
        **kwargs,
    ):
        self.wall_coordinates = wall_coordinates
        super().__init__(**kwargs)

    def check_collision(self, action: np.ndarray) -> bool:
        # Check if the action makes the agent go through the wall.
        # Return True if the agent goes through the wall, False otherwise.

        qpos = self.sim.data.qpos.flat.copy()[0:2]
        dx = action[0] * self._max_dx_dy
        dy = action[1] * self._max_dx_dy
        new_qpos = qpos + np.array([dx, dy])

        for wall in self.wall_coordinates:
            A = wall[:2]
            B = wall[2:]
            if intersect(A, B, qpos, new_qpos):
                return True
        return False

    def step(self, action):
        assert not self._use_simulator
        assert self._use_dx_dy_physics
        if self.check_collision(action):
            return super().step(np.array([0, 0]))
        return super().step(action)

    def reset_model(self, reset_goals=False):
        return super().reset_model(reset_goals=True)


class TestPointMassWallEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = PointMassWallEnv(
            wall_coordinates=((0, 1, 1, 0)),
            # wall_coordinates={
            #     "first_coordinate": {"x": 0, "y": 1},
            #     "second_coordinate": {"x": 1, "y": 0},
            # },
            model_path="point_dx_dy.xml",
            goal_sequence_generation_fn=relabeled_stitching_goal_sequence_fn,
            goal_sequence_generation_fn_kwargs={"epsilon": 0.05},
            reward_function_kwargs={"epsilon": 0.05},
            add_noise_to_initial_position=False,
            use_simulator=False,
            use_dx_dy_physics=True,
            max_dx_dy=1,
        )

    def test_check_collision(self):
        self.env.reset()
        self.assertFalse(self.env.check_collision(np.array([0, 0])))
        self.assertTrue(self.env.check_collision(np.array([1, 1])))
        self.assertFalse(self.env.check_collision(np.array([0, 1])))
        self.assertFalse(self.env.check_collision(np.array([1, 0])))
        self.assertFalse(self.env.check_collision(np.array([0.5, 0.5])))
        self.assertTrue(self.env.check_collision(np.array([0.5, 0.6])))
        self.assertTrue(self.env.check_collision(np.array([0.6, 0.5])))
        self.env.reset()
        self.env.wall_coordinates = ((0, 10), (10, 0))
        # self.env.wall_coordinates = {
        #     "first_coordinate": {"x": 0, "y": 10},
        #     "second_coordinate": {"x": 10, "y": 0},
        # }
        self.assertFalse(self.env.check_collision(np.array([0, 0])))
        self.assertFalse(self.env.check_collision(np.array([1, 1])))
        # Check it works with max_dx_dy
        self.assertTrue(self.env.check_collision(np.array([100, 100])))
        self.env.wall_coordinates = ((0, 1), (1, 0))
        # self.env.wall_coordinates = {
        #     "first_coordinate": {"x": 0, "y": 1},
        #     "second_coordinate": {"x": 1, "y": 0},
        # }

    def test_step(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(np.array([0, 0]))
        self.assertTrue(np.allclose(obs[:2], np.array([0, 0])))
        obs, reward, done, info = self.env.step(np.array([10, 10]))
        self.assertTrue(np.allclose(obs[:2], np.array([0, 0])))
        obs, reward, done, info = self.env.step(np.array([0, 1]))
        self.assertTrue(np.allclose(obs[:2], np.array([0, 1])))
        self.env.reset()
        obs, reward, done, info = self.env.step(np.array([0.5, 0.5]))
        self.assertTrue(np.allclose(obs[:2], np.array([0.5, 0.5])))
        obs, reward, done, info = self.env.step(np.array([0.5, 0.5]))
        self.assertTrue(np.allclose(obs[:2], np.array([0.5, 0.5])))


if __name__ == "__main__":
    unittest.main()
