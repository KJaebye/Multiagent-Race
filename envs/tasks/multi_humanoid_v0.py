import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from envs.utils.ma_xml import create_multiagent_xml
from config.config import cfg

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

import math
from typing import Optional, Union

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model_body_mass, data_xipos):
    mass = np.expand_dims(model_body_mass, axis=1)
    xpos = data_xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class MultiHumanoidEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(
        self,
        agent_num=cfg.EMAT.AGENT_NUM,
        xml_file="humanoid.xml",
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        
        self.agent_num = agent_num

        #----------------------- single agent observation dim -----------------------#
        # 24: sa_qpos
        # 23: sa_qvel
        # 2 : relative x pos and relative x vel
        # 130: cinert // (14-1)*10 
        # 78: cvel // (14-1)*6
        # 23: qfrc_actuator
        # 78: cfrc_ext // (14-1)*6

        if cfg.ENV.USE_ONE_HOT_ENCODING:
            self.onehot_num = 1
            self.sa_obs_dim = (356 + self.onehot_num * self.agent_num)
        else:
            self.sa_obs_dim = (356 + 2 * self.agent_num)

        # observation space definition
        sa_obs_space = Box(
            low=-np.inf, high=np.inf, shape=(1, self.sa_obs_dim), dtype=np.float64
        )
        ma_obs_space = Box(
            low=-np.inf, high=np.inf, shape=(self.agent_num, self.sa_obs_dim), dtype=np.float64
        )
        observation_space = ma_obs_space

        #----------------------- single agent action dim -----------------------#
        self.sa_action_dim = 17

        # extract agent from base xml
        local_position = "envs/assets/xml/" + xml_file
        self.fullxml = create_multiagent_xml(local_position)

        MujocoEnv.__init__(
            self,
            "humanoid.xml",
            5,
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
    
    @property
    def ma_cinert(self):
        """ Return an array that includes the cinert of each agent. """
        start = 1*10
        cinert = self.data.cinert.ravel().copy()[start:]
        assert len(cinert) % self.agent_num == 0, "State cinert cannot be aligned!"
        agent_cinert_length = int(len(cinert) / self.agent_num)
        ma_cinert = np.array([cinert[idx*agent_cinert_length:(idx+1)*agent_cinert_length] 
                              for idx in range(self.agent_num)])
        return ma_cinert
    
    @property
    def ma_cvel(self):
        """ Return an array that includes the cvel of each agent. """
        start = 1*6
        cvel = self.data.cvel.ravel().copy()[start:]
        assert len(cvel) % self.agent_num == 0, "State cvel cannot be aligned!"
        agent_cvel_length = int(len(cvel) / self.agent_num)
        ma_cvel = np.array([cvel[idx*agent_cvel_length:(idx+1)*agent_cvel_length] 
                              for idx in range(self.agent_num)])
        return ma_cvel
    
    @property
    def ma_qfrc_actuator(self):
        """ Return an array that includes the qfrc_actuator of each agent. """
        assert len(self.data.qfrc_actuator.ravel().copy()) % self.agent_num == 0, "State qfrc_actuator cannot be aligned!"
        agent_qfrc_actuator_length = int(len(self.data.qfrc_actuator.ravel().copy()) / self.agent_num)
        ma_qfrc_actuator = np.array([self.data.qfrc_actuator.ravel().copy()[idx*agent_qfrc_actuator_length:(idx+1)*agent_qfrc_actuator_length] 
                              for idx in range(self.agent_num)])
        return ma_qfrc_actuator
    
    @property
    def ma_cfrc_ext(self):
        """ Return an array that includes the cfrc_ext of each agent. """
        start = 1*6
        cfrc_ext = self.data.cfrc_ext.ravel().copy()[start:]
        assert len(cfrc_ext) % self.agent_num == 0, "State cfrc_ext cannot be aligned!"
        agent_cfrc_ext_length = int(len(cfrc_ext) / self.agent_num)
        ma_cfrc_ext = np.array([cfrc_ext[idx*agent_cfrc_ext_length:(idx+1)*agent_cfrc_ext_length] 
                              for idx in range(self.agent_num)])
        return ma_cfrc_ext
    
    @property
    def ma_body_mass(self):
        """ Return an array that includes the body_mass of each agent. """
        start = 1*1
        body_mass = self.model.body_mass.ravel().copy()[start:]
        assert len(body_mass) % self.agent_num == 0, "State body_mass cannot be aligned!"
        agent_body_mass_length = int(len(body_mass) / self.agent_num)
        ma_body_mass = np.array([body_mass.copy()[idx*agent_body_mass_length:(idx+1)*agent_body_mass_length] 
                              for idx in range(self.agent_num)])
        return ma_body_mass
    
    @property
    def ma_xipos(self):
        """ Return an array that includes the xipos of each agent. """
        start = 1*3
        xipos = self.data.xipos.ravel().copy()[start:]
        assert len(xipos) % self.agent_num == 0, "State xipos cannot be aligned!"
        agent_xipos_length = int(len(xipos) / self.agent_num)
        ma_xipos = np.array([xipos[idx*agent_xipos_length:(idx+1)*agent_xipos_length] 
                              for idx in range(self.agent_num)])
        return ma_xipos
    
    @property
    def ma_mass_center(self):
        ma_mass_center = np.array([
            mass_center(self.ma_body_mass[idx], self.ma_xipos[idx]) for idx in range(self.agent_num)
            ])
        return ma_mass_center
    
    @property
    def ma_position(self):
        """ Return an array to get multi agent x y z positions. """
        return self.ma_qpos[:, :3]
    
    @property
    def ma_velocity(self):
        """ Return an array to get multi agent x y z velocities. """
        return self.ma_qvel[:, :3]
    
    def get_ma_forward_reward(self, ma_xy_velocity):
        """ Return multi agent forward reward. """
        # 0 denote x, 1 denote y
        ma_forward_reward = np.array([ma_xy_velocity[idx, 0] for idx in range(self.agent_num)])[:, np.newaxis]
        return ma_forward_reward
    
    def get_ma_action(self, action):
        """ Break action into sub actions corresponding to each agent. """
        assert len(action) % self.agent_num == 0, "Action cannot be aligned!"
        agent_action_length = int(len(action) / self.agent_num)
        ma_action = np.array([action[(idx-1)*agent_action_length:idx*agent_action_length] 
                              for idx in range(1, self.agent_num+1)])
        return ma_action
    
    @property
    def ma_healthy_reward(self):
        return (
            np.array([
                (self.ma_is_healthy[idx] or self._terminate_when_unhealthy)
            * self._healthy_reward for idx in range(self.agent_num)
            ])[:, np.newaxis]
        )
    
    @property
    def ma_is_healthy(self):
        min_z, max_z = self._healthy_z_range
        ma_is_healthy = np.array([
            np.isfinite(self.ma_qpos[idx]).all() and 
            np.isfinite(self.ma_qvel[idx]).all() and 
            min_z <= self.ma_qpos[idx][2] <= max_z for idx in range(self.agent_num)]
            )
        return ma_is_healthy
    
    def get_ma_control_cost(self, action):
        # break action into sub actions
        ma_action = self.get_ma_action(action)
        ma_control_cost = np.array([
            self._ctrl_cost_weight * np.sum(np.square(ma_action[idx])) for idx in range(self.agent_num)
            ])[:, np.newaxis]
        return ma_control_cost
    
    @property
    def terminateds(self):
        # terminateds = not self.ma_is_healthy.all() if self._terminate_when_unhealthy else False
        # return [terminateds] * self.agent_num
        terminated = [not is_healthy for is_healthy in self.ma_is_healthy]
        return terminated

    def step(self, action):
        # input action is a list
        action = np.hstack(action)

        ma_xy_position_before = self.ma_mass_center
        self.do_simulation(action, self.frame_skip)
        ma_xy_position_after = self.ma_mass_center

        ma_xy_velocity = (ma_xy_position_after - ma_xy_position_before) / self.dt

        ma_costs = ma_ctrl_cost = self.get_ma_control_cost(action)

        ma_forward_reward = self._forward_reward_weight * self.get_ma_forward_reward(ma_xy_velocity)
        ma_healthy_reward = self.ma_healthy_reward

        ma_rewards = ma_forward_reward + ma_healthy_reward

        terminateds = self.terminateds
        observations = self._get_ma_obs()
        info = {
            "ma_reward_linvel": ma_forward_reward,
            "ma_reward_quadctrl": -ma_ctrl_cost,
            "ma_reward_alive": ma_healthy_reward,
            "ma_x_position": ma_xy_position_after[:, 0],
            "ma_y_position": ma_xy_position_after[:, 1],
            "ma_distance_from_origin": np.linalg.norm(ma_xy_position_after[:, :2]-self.initial_ma_position[:, :2], ord=2),
            "ma_x_velocity": ma_xy_velocity[:, 0],
            "ma_y_velocity": ma_xy_velocity[:, 1],
            "ma_forward_reward": ma_forward_reward,
        }

        info["ma_reward_ctrl"] = -ma_ctrl_cost

        rewards = [(ma_rewards[idx] - ma_costs[idx]) for idx in range(self.agent_num)]

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminateds, False, info
    
    @property
    def ma_contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext[1:, :] # first line contact friction data is worldbody?
        assert len(raw_contact_forces) % self.agent_num == 0, "Contact forces cannot be aligned!"
        agent_contact_length = int(len(raw_contact_forces) / self.agent_num)
        ma_contact_forces = np.array([raw_contact_forces[(idx-1)*agent_contact_length:idx*agent_contact_length] 
                              for idx in range(1, self.agent_num+1)])
        
        min_value, max_value = self._contact_force_range
        ma_contact_forces = np.array([
            np.clip(ma_contact_forces[idx], min_value, max_value) for idx in range(self.agent_num)
            ])
        return ma_contact_forces
    
    def _get_ma_obs(self):
        return [self._get_sa_obs(idx) for idx in range(self.agent_num)]
    
    def _get_sa_obs(self, idx):
        """ Return single agent observation. """
        #--------------- Proprioceptive observations -----------------------#
        sa_qpos = self.ma_qpos[idx]
        sa_qvel = self.ma_qvel[idx]
        sa_cinert = self.ma_cinert[idx]
        sa_cvel = self.ma_cvel[idx]
        sa_qfrc_actuator = self.ma_qfrc_actuator[idx]
        sa_cfrc_ext = self.ma_cfrc_ext[idx]

        #------------------ External observations --------------------------#
        sa_relative_obs = np.hstack(self.get_sa_relative_obs(idx))
        
        #----------------- Concatenate observations ------------------------#
        sa_obs = np.concatenate(
            (
                sa_qpos,
                sa_qvel,
                sa_cinert,
                sa_cvel,
                sa_qfrc_actuator,
                sa_cfrc_ext,
            )
        )

        sa_obs = np.concatenate((sa_obs, sa_relative_obs))
        return sa_obs

    def get_sa_relative_obs(self, idx):
        """ Return a list including relative observation infos. """
        # single agent position and velocity
        sa_position = self.ma_position[idx]
        sa_velocity = self.ma_velocity[idx]
        # the max x vel in agents
        x_velocity_max = np.max(self.ma_velocity[:, 0])

        sa_relative_obs = []
        for j in range(self.agent_num):
            # neighbours position and velocity
            j_position = self.ma_position[j]
            j_velocity = self.ma_velocity[j]
            # relative position and velocity
            r_position = j_position - sa_position
            r_velocity = j_velocity - sa_velocity

            if cfg.ENV.USE_ONE_HOT_ENCODING:
                assert cfg.ENV.USE_NOISE is False
                if cfg.ENV.USE_RELATIVE_OBS:
                    if j_velocity[0] == x_velocity_max: # when agent is the fastest one at this step, use 1 pedding
                        j_relative_obs = np.ones((self.onehot_num, ))
                    else:
                        j_relative_obs = np.zeros((self.onehot_num, ))
                else:
                    j_relative_obs = np.zeros((self.onehot_num, ))
            else:
                if cfg.ENV.USE_RELATIVE_OBS:
                    j_relative_obs = np.concatenate((r_position[:1], r_velocity[:1]))
                else:
                    j_relative_obs = np.zeros((2, ))

            if cfg.ENV.USE_NOISE:
                j_relative_obs = np.random.random((2, ))
            
            sa_relative_obs.append(j_relative_obs)
        
        return sa_relative_obs

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