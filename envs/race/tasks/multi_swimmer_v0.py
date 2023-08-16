__credits__ = ["Rushiv Arora"]

import numpy as np
import math

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

from typing import Optional, Union

from envs.utils.ma_xml import create_multiagent_xml
from config.config import cfg

class MultiSwimmerEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description

    This environment is based on the mujoco environment from gymnasium swimmer_v0.py 

    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        agent_num             = cfg.EMAT.AGENT_NUM,
        xml_file              = "swimmer.xml",
        forward_reward_weight = cfg.ENV.FORWARD_REWARD_WEIGHT,
        ctrl_cost_weight      = cfg.ENV.CTRL_COST_WEIGHT,
        reset_noise_scale     = cfg.ENV.RESET_NOISE_SCALE,
        exclude_current_positions_from_observation=False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.agent_num = agent_num

        #----------------------- single agent observation dim -----------------------#
        # 5: sa_qpos
        # 5: sa_qvel
        # 2 : relative x pos and relative x vel
        self.sa_obs_dim = (5 + 5 + 2 * self.agent_num)
        sa_obs_space = Box(
            low=-np.inf, high=np.inf, shape=(1, self.sa_obs_dim), dtype=np.float64
        )
        ma_obs_space = Box(
            low=-np.inf, high=np.inf, shape=(self.agent_num, self.sa_obs_dim), dtype=np.float64
        )
        observation_space = ma_obs_space

        #----------------------- single agent action dim -----------------------#
        self.sa_action_dim = 2

        # extract agent from base xml
        local_position = "envs/race/assets/xml/" + xml_file
        self.fullxml = create_multiagent_xml(local_position)

        MujocoEnv.__init__(
            self, "swimmer.xml", 4, observation_space=observation_space, **kwargs
        )

    def _initialize_simulation(self):
        # import model from xml string
        self.model = mujoco.MjModel.from_xml_string(self.fullxml)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    def get_ma_control_cost(self, action):
        # break action into sub actions
        ma_action = self.get_ma_action(action)
        ma_control_cost = np.array([
            self._ctrl_cost_weight * np.sum(np.square(ma_action[idx])) for idx in range(self.agent_num)
            ])[:, np.newaxis]
        return ma_control_cost
    
    def get_ma_action(self, action):
        """ Break action into sub actions corresponding to each agent. """
        assert len(action) % self.agent_num == 0, "Action cannot be aligned!"
        agent_action_length = int(len(action) / self.agent_num)
        ma_action = np.array([action[(idx-1)*agent_action_length:idx*agent_action_length] 
                              for idx in range(1, self.agent_num+1)])
        return ma_action
    
    def _get_ma_obs(self):
        return [self._get_sa_obs(idx) for idx in range(self.agent_num)]
    
    def _get_sa_obs(self, idx):
        """ Return single agent observation. """
        #--------------- Proprioceptive observations -----------------------#
        sa_qpos = self.ma_qpos[idx]
        sa_qvel = self.ma_qvel[idx]

        #------------------ External observations --------------------------#
        sa_relative_obs = np.hstack(self.get_sa_relative_obs(idx))
        
        #----------------- Concatenate observations ------------------------#
        sa_obs = np.concatenate((sa_qpos, sa_qvel))
        sa_obs = np.concatenate((sa_obs, sa_relative_obs))
        return sa_obs
    
    def get_sa_relative_obs(self, idx):
        """ Return a list including relative observation infos. """
        # single agent position and velocity
        sa_position = self.ma_position[idx]
        sa_velocity = self.ma_velocity[idx]

        sa_relative_obs = []
        for j in range(self.agent_num):
            # neighbours position and velocity
            j_position = self.ma_position[j]
            j_velocity = self.ma_velocity[j]
            # relative position and velocity
            r_position = j_position - sa_position
            r_velocity = j_velocity - sa_velocity
            # only take x direction relative pos and vel
            j_relative_obs = np.concatenate((r_position[:1], r_velocity[:1]))
            if not cfg.ENV.USE_RELATIVE_OBS:
                j_relative_obs = np.zeros((2, ))
            sa_relative_obs.append(j_relative_obs)
        
        return sa_relative_obs
    
    @property
    def ma_qpos(self):
        """ Return an array that includes the pos of each agent. """
        assert len(self.data.qpos.ravel().copy()) % self.agent_num == 0, "State qpos cannot be aligned!"
        agent_qpos_length = int(len(self.data.qpos.ravel().copy()) / self.agent_num)
        ma_qpos = np.array([self.data.qpos.ravel().copy()[idx*agent_qpos_length:(idx+1)*agent_qpos_length] 
                              for idx in range(self.agent_num)])
        return ma_qpos
    
    @property
    def ma_qvel(self):
        """ Return an array that includes the vel of each agent. """
        assert len(self.data.qvel.ravel().copy()) % self.agent_num == 0, "State qvel cannot be aligned!"
        agent_qvel_length = int(len(self.data.qvel.ravel().copy()) / self.agent_num)
        ma_qvel = np.array([self.data.qvel.ravel().copy()[idx*agent_qvel_length:(idx+1)*agent_qvel_length] 
                              for idx in range(self.agent_num)])
        return ma_qvel

    @property
    def ma_position(self):
        """ Return an array to get multi agent x y positions. """
        return self.ma_qpos[:, :2]
    
    @property
    def ma_velocity(self):
        """ Return an array to get multi agent x y velocities. """
        return self.ma_qvel[:, :2]
    
    def get_ma_forward_reward(self, ma_xy_velocity):
        """ Return multi agent forward reward. """
        # 0 denote x, 1 denote y
        ma_forward_reward = np.array([ma_xy_velocity[idx, 0] for idx in range(self.agent_num)])[:, np.newaxis]
        return ma_forward_reward


    def step(self, action):
        # input action is a list
        action = np.hstack(action)

        ma_xy_position_before = self.ma_position[:, :2]
        self.do_simulation(action, self.frame_skip)
        ma_xy_position_after = self.ma_position[:, :2]
        ma_xy_velocity = (ma_xy_position_after - ma_xy_position_before) / self.dt

        ma_forward_reward = self.get_ma_forward_reward(ma_xy_velocity) * self._forward_reward_weight

        ma_costs = ma_ctrl_cost = self.get_ma_control_cost(action)

        observations = self._get_ma_obs()
        ma_rewards = ma_forward_reward - ma_ctrl_cost
        info = {
            "ma_reward_fwd": ma_forward_reward,
            "ma_reward_ctrl": -ma_ctrl_cost,
            "ma_x_position": ma_xy_position_after[:, 0],
            "ma_y_position": ma_xy_position_after[:, 1],
            "ma_distance_from_origin": np.linalg.norm(ma_xy_position_after[:, :2]-self.initial_ma_position[:, :2], ord=2),
            "ma_x_velocity": ma_xy_velocity[:, 0],
            "ma_y_velocity": ma_xy_velocity[:, 1],
            "ma_forward_reward": ma_forward_reward,
        }

        rewards = [ma_rewards[idx] for idx in range(self.agent_num)]

        if self.render_mode == "human":
            self.render()

        return observations, rewards, [False] * self.agent_num, False, info


    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        #----------- set initial positions for multi agent ---------------#
        init_ma_qpos = self.ma_qpos
        init_ma_qvel = self.ma_qvel
        for idx in range(self.agent_num):
            if cfg.EMAT.INIT_FORMATION == "star":
                r = cfg.EMAT.SPACE * math.log(self.agent_num)
                init_ma_qpos[idx][0] += r * math.cos(2 * math.pi * idx / cfg.EMAT.AGENT_NUM)
                init_ma_qpos[idx][1] += r * math.sin(2 * math.pi * idx / cfg.EMAT.AGENT_NUM)
            elif cfg.EMAT.INIT_FORMATION == "line":
                init_ma_qpos[idx][1] += cfg.EMAT.SPACE * 2 * (idx - (self.agent_num-1) / 2)

        # record agents initial xy position
        self.initial_ma_position = self.ma_position
        # flatten
        init_ma_qpos = init_ma_qpos.ravel()
        init_ma_qvel = init_ma_qvel.ravel()
        # add noise
        init_ma_qpos = init_ma_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        init_ma_qvel = (
            init_ma_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        # set state
        self.set_state(init_ma_qpos, init_ma_qvel)

        observation = self._get_ma_obs()
        return observation
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,):
        super().reset(seed=seed)

        self._reset_simulation()

        ob = self.reset_model()
        if self.render_mode == "human":
            self.render()
        return ob, {}