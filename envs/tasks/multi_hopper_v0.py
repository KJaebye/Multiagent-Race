import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from envs.utils.ma_xml import create_multiagent_xml
from config.config import cfg
from typing import Optional, Union

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None



DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class MultiHopperEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description

    This environment is based on the single agent mujoco environment from gymnasium hopper_v4.py

    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125, # 125
    }

    def __init__(
        self,
        agent_num                = cfg.EMAT.AGENT_NUM,
        forward_reward_weight    = cfg.ENV.FORWARD_REWARD_WEIGHT,
        ctrl_cost_weight         = cfg.ENV.CTRL_COST_WEIGHT,
        healthy_reward           = cfg.ENV.HEALTHY_REWARD,
        terminate_when_unhealthy = cfg.ENV.TERMINATE_WHEN_UNHEALTHY,
        healthy_state_range      = cfg.ENV.HEALTHY_STATE_RANGE,
        healthy_z_range          = cfg.ENV.HEALTHY_Z_RANGE,
        healthy_angle_range      = cfg.ENV.HEALTHY_ANGLE_RANGE,
        reset_noise_scale        = cfg.ENV.RESET_NOISE_SCALE,
        exclude_current_positions_from_observation=False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.agent_num = agent_num

        #----------------------- single agent observation dim -----------------------#
        # 6: sa_qpos
        # 6: sa_qvel
        # 2 : relative x pos and relative x vel
        self.sa_obs_dim = (6 + 6 + 2 * self.agent_num)
        sa_obs_space = Box(
            low=-np.inf, high=np.inf, shape=(1, self.sa_obs_dim), dtype=np.float64
        )
        ma_obs_space = Box(
            low=-np.inf, high=np.inf, shape=(self.agent_num, self.sa_obs_dim), dtype=np.float64
        )
        observation_space = ma_obs_space

        #----------------------- single agent action dim -----------------------#
        self.sa_action_dim = 3

        # extract agent from base xml
        local_position = "envs/assets/xml/hopper.xml"
        self.fullxml = create_multiagent_xml(local_position)

        MujocoEnv.__init__(
            self,
            "hopper.xml",
            4,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _initialize_simulation(self):
        # import model from xml string
        self.model = mujoco.MjModel.from_xml_string(self.fullxml)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

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
    
    def ma_state_vector(self):
        """Return the position and velocity joint states for each agent."""
        return np.hstack([self.ma_qpos, self.ma_qvel])

    @property
    def ma_is_healthy(self):
        ma_z = self.ma_qpos[:, 1]
        ma_angle = self.ma_qpos[:, 2]
        ma_state = self.ma_state_vector()[:, 2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        ma_healthy_state = np.array([
            np.all(np.logical_and(min_state < ma_state[idx], ma_state[idx] < max_state)) for idx in range(self.agent_num)
            ])
        
        ma_healthy_z = np.array([min_z < ma_z[idx] < max_z for idx in range(self.agent_num)])

        ma_healthy_angle = np.array([min_angle < ma_angle[idx] < max_angle for idx in range(self.agent_num)])

        ma_is_healthy = np.array([all((ma_healthy_state[idx], ma_healthy_z[idx], ma_healthy_angle[idx])) for idx in range(self.agent_num)])

        return ma_is_healthy

    @property
    def ma_healthy_reward(self):
        return (
            np.array([
                (self.ma_is_healthy[idx] or self._terminate_when_unhealthy)
            * self._healthy_reward for idx in range(self.agent_num)
            ])[:, np.newaxis]
        )
    
    def get_ma_action(self, action):
        """ Break action into sub actions corresponding to each agent. """
        assert len(action) % self.agent_num == 0, "Action cannot be aligned!"
        agent_action_length = int(len(action) / self.agent_num)
        ma_action = np.array([action[(idx-1)*agent_action_length:idx*agent_action_length] 
                              for idx in range(1, self.agent_num+1)])
        return ma_action
    
    def get_ma_control_cost(self, action):
        # break action into sub actions
        ma_action = self.get_ma_action(action)
        ma_control_cost = np.array([
            self._ctrl_cost_weight * np.sum(np.square(ma_action[idx])) for idx in range(self.agent_num)
            ])[:, np.newaxis]
        return ma_control_cost

    @property
    def ma_position(self):
        """ Return an array to get multi agent x positions. """
        return self.ma_qpos[:, :1]
    
    @property
    def ma_x_position(self):
        """ Return an array to get multi agent x positions. """
        return self.ma_qpos[:, :1]
    
    @property
    def ma_velocity(self):
        """ Return an array to get multi agent x velocities. """
        return self.ma_qvel[:, :1]
    
    @property
    def terminateds(self):
        terminated = [not is_healthy for is_healthy in self.ma_is_healthy]
        return terminated

    def _get_ma_obs(self):
        return [self._get_sa_obs(idx) for idx in range(self.agent_num)]
    
    def _get_sa_obs(self, idx):
        """ Return single agent observation. """
        #--------------- Proprioceptive observations -----------------------#
        sa_position = self.ma_qpos[idx].copy()
        sa_velocity = np.clip(self.ma_qvel[idx].copy(), -10, 10)

        #------------------ External observations --------------------------#
        sa_relative_obs = np.hstack(self.get_sa_relative_obs(idx))
        
        #----------------- Concatenate observations ------------------------#
        sa_obs = np.concatenate((sa_position, sa_velocity))
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

            j_relative_obs = np.concatenate((r_position, r_velocity))
            if not cfg.ENV.USE_RELATIVE_OBS:
                j_relative_obs = np.zeros((2, ))
            if cfg.ENV.USE_NOISE:
                j_relative_obs = np.random.random((2, ))
            sa_relative_obs.append(j_relative_obs)

        return sa_relative_obs
    
    
    def step(self, action):
        # input action is a list
        action = np.hstack(action)
        
        ma_x_position_before = self.ma_x_position[:, :1]
        self.do_simulation(action, self.frame_skip)
        ma_x_position_after = self.ma_x_position[:, :1]

        ma_x_velocity = (ma_x_position_after - ma_x_position_before) / self.dt

        ma_ctrl_cost = self.get_ma_control_cost(action)

        ma_forward_reward = self._forward_reward_weight * ma_x_velocity
        ma_healthy_reward = self.ma_healthy_reward

        ma_rewards = ma_forward_reward + ma_healthy_reward
        ma_costs = ma_ctrl_cost

        observations = self._get_ma_obs()
        terminateds = self.terminateds

        rewards = [ma_rewards[idx] - ma_costs[idx] for idx in range(self.agent_num)]

        info = {
            "ma_x_position": ma_x_position_after,
            "ma_x_velocity": ma_x_velocity,
        }

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminateds, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observations = self._get_ma_obs()
        return observations
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,):
        super().reset(seed=seed)

        self._reset_simulation()

        ob = self.reset_model()
        if self.render_mode == "human":
            self.render()
        return ob, {}

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)