import mujoco
import mujoco.viewer
import gymnasium as gym
import numpy as np
import gym_aloha
from gym_aloha.tasks.sim import BOX_POSE
from gym_aloha.utils import plot_observation_images
import matplotlib.pyplot as plt 

obs_type='pixels_agent_pos'
env = gym.make("gym_aloha/TrossenAIStationaryTransferCube-v0", #TrossenAIStationaryTransferCube-v0", #TrossenAIStationaryTransferCubeEE-v0
             obs_type=obs_type,
             box_size=[0.02,0.02,0.02],
             box_pos=[0.0,0.0,-0.02], #[0.0,0.0,-0.02],
             box_color=[0,1,0,1],
             tabletop='my_desktop', #'my_desktop', #'wood',
             #backdrop='my_backdrop', #'none' 'my_backdrop'
             #lighting=[[0.1,0.1,0.1],[-0.5,0.5,0.5]], #[[0.3,0.3,0.3],[0.3,0.3,0.3]], #table lighting, ambient lighting
             #arms_pos=[-0.4575, 0.0, 0.02, 0.4575, 0.0, 0.02], #base position, left, right; default=[+-4575 -0.019 0.02]
             #arms_ref=[0,-0.015,0.015,0,0,0,0,-0.025,0.025,0,0,0], #left joints 0-5 ref, right joints 0-5 ref; default=[all zeros]
            )

BOX_POSE[0]=env.unwrapped._env.physics.named.data.qpos[-7:]
observation, info = env.reset() #seed=None,options='do_not_reset_BOX_POSE')
print(f"home_pose={observation['agent_pos']}")

model = env.unwrapped._env.physics.model
data = env.unwrapped._env.physics.data

# Update the simulation state
mujoco.mj_forward(model.ptr, data.ptr)

#show camera images
observation=env.unwrapped._env.task.get_observation(env.unwrapped._env.physics)
cam_list = ["top"]
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube' or env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
plt_imgs = plot_observation_images(observation['images'], cam_list)
plt.pause(0.2)

# Launch viewer
mujoco.viewer.launch(model.ptr, data.ptr)


