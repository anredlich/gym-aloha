import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask, TrossenAIStationaryTransferCubeTask
from gym_aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
    TransferCubeEETask
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose, sample_box_pose_trossen_ai_stationary 


class AlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self._env = self._make_env_task(self.task)

        #anr, added cam_list
        cam_list=["top"]
        if task == 'trossen_ai_stationary_transfer_cube' or task == 'trossen_ai_stationary_transfer_cube_ee':
            cam_list=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

        #anr moved here so it can be used for 'agent_pos'
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)

        #anr use actual action_space:
        if task == 'trossen_ai_stationary_transfer_cube':
            ctrl_range = self._env.physics.model.actuator_ctrlrange
            self.action_space = spaces.Box(
                low=ctrl_range[:, 0].astype(np.float32),   # Lower bounds
                high=ctrl_range[:, 1].astype(np.float32),  # Upper bounds
                shape=(ctrl_range.shape[0],),
                dtype=np.float32
            ) 
        elif task == 'trossen_ai_stationary_transfer_cube_ee':
            self.action_space = spaces.Box(low=-.1, high=.1, shape=(16,), dtype=np.float32)

        #self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)
        #anr added if trossen:
        #if task == 'trossen_ai_stationary_transfer_cube':
        #    self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(16,), dtype=np.float32)

        if self.obs_type == "state":
            raise NotImplementedError()
            self.observation_space = spaces.Box(
                low=np.array([0] * len(JOINTS)),  # ???
                high=np.array([255] * len(JOINTS)),  # ???
                dtype=np.float64,
            )
        # elif self.obs_type == "pixels":
        #     self.observation_space = spaces.Dict(
        #         {
        #             "top": spaces.Box(
        #                 low=0,
        #                 high=255,
        #                 shape=(self.observation_height, self.observation_width, 3),
        #                 dtype=np.uint8,
        #             )
        #         }
        #     )
        elif self.obs_type == "pixels": #anr added
            self.observation_space = spaces.Dict({
                cam: spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.observation_height, self.observation_width, 3),
                    dtype=np.uint8,
                )
                for cam in cam_list
            })
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    # "pixels": spaces.Dict(
                    #     {
                    #         "top": spaces.Box(
                    #             low=0,
                    #             high=255,
                    #             shape=(self.observation_height, self.observation_width, 3),
                    #             dtype=np.uint8,
                    #         )
                    #     }
                    # ),
                    "pixels": spaces.Dict({ #anr added
                        cam: spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.observation_height, self.observation_width, 3),
                            dtype=np.uint8,
                        )
                        for cam in cam_list
                    }),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(self.action_space.shape[0],), #anr was len(JOINTS)
                        dtype=np.float64,
                    ),
                    #"agent_pos": self.action_space,
                }
            )

        #self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)
        # #anr added if trossen:
        # if task == 'trossen_ai_stationary_transfer_cube':
        #     self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(16,), dtype=np.float32)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        if self.task == 'trossen_ai_stationary_transfer_cube': #anr use most top-like camera
            image = self._env.physics.render(height=height, width=width, camera_id="cam_high")
        elif self.task == 'trossen_ai_stationary_transfer_cube_ee': #anr use most top-like camera
            image = self._env.physics.render(height=height, width=width, camera_id="cam_high")
        else:
            image = self._env.physics.render(height=height, width=width, camera_id="top")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif task_name == "trossen_ai_stationary_transfer_cube":
            xml_path = ASSETS_DIR / "trossen_ai_scene_joint.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TrossenAIStationaryTransferCubeTask()
        elif task_name == "trossen_ai_stationary_transfer_cube_ee":
            xml_path = ASSETS_DIR / "trossen_ai_scene.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEETask()
        elif task_name == "insertion":
            xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "end_effector_transfer_cube":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif task_name == "end_effector_insertion":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            if 'top' in raw_obs["images"].keys():                                   #anr added
                obs = {"top": raw_obs["images"]["top"].copy()}
            else:                                                                   #anr added
                obs = {key: img.copy() for key, img in raw_obs["images"].items()}   #anr added
        elif self.obs_type == "pixels_agent_pos":
            if 'top' in raw_obs["images"].keys():                                   #anr added
                obs = {
                    "pixels": {"top": raw_obs["images"]["top"].copy()},
                    "agent_pos": raw_obs["qpos"],
                }
            else:                                                                   #anr added
                 obs = {
                     "pixels": {key: img.copy() for key, img in raw_obs["images"].items()},
                     "agent_pos": raw_obs["qpos"],
                 }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)
            self._env.task.seed=seed #anr save the actual seed for use in reset()

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif self.task == "trossen_ai_stationary_transfer_cube":  #anr added
            BOX_POSE[0] = sample_box_pose_trossen_ai_stationary(seed) #anr added
        elif self.task == "trossen_ai_stationary_transfer_cube_ee":  #anr added
            BOX_POSE[0] = sample_box_pose_trossen_ai_stationary(seed) #anr added
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        info['raw_obs']=raw_obs.observation #anr added 8/2/25

        return observation, info

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        _, reward, _, raw_obs = self._env.step(action)

        # TODO(rcadene): add an enum
        terminated = is_success = reward == 4

        info = {"is_success": is_success}
        info['raw_obs']=raw_obs #anr added 8/2/25

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
