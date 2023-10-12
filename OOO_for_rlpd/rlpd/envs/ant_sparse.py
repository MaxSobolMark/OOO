import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class SparseAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        sparse_val=10.0,
        include_xpos_in_obs=True,
    ):
        # sparse_val is the threshold for sparse reward, i.e. the ant will get reward 1 if it
        # crosses this threshold.
        self.sparse_val = sparse_val
        self.include_xpos_in_obs = include_xpos_in_obs
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        ob = self._get_obs()

        if xposafter - self.init_qpos[0] > self.sparse_val:
            reward = 1.0
        else:
            reward = 0.0
        done = False

        return ob, reward, done, {}

    def _get_obs(self):
        qpos_obs = self.sim.data.qpos.flat[2:]
        if self.include_xpos_in_obs:
            qpos_obs = np.concatenate([[self.sim.data.qpos.flat[0]], qpos_obs])
        return np.concatenate(
            [
                qpos_obs,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5
