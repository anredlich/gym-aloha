import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
from gym_aloha.utils import plot_observation_images
import matplotlib.pyplot as plt

obs_type='pixels_agent_pos' #'pixels' is default
env = gym.make("gym_aloha/TrossenAIStationaryTransferCube-v0",
             obs_type=obs_type,
             #comment options for default
             box_size=[0.02,0.02,0.02], #40mm box, default is 0.0125
             #box_pos=[0.0,0.0,-0.02], #careful! will override reset() if z pos is >0
             box_color=[0,1,0,1], #default is [1,0,0,1]
             tabletop='wood', #'my_desktop' default is black
             backdrop='my_backdrop', #default is none
             #lighting=[[0.3,0.3,0.3],[0.3,0.3,0.3]],
             #arms_pos=[-0.4575, 0.0, 0.02, 0.4575, 0.0, 0.02], #for sim to real calibration
             #arms_ref=[0,-0.015,0.015,0,0,0,0,-0.025,0.025,0,0,0], #for sim to real calibration
             )
#env = gym.make("gym_aloha/TrossenAIStationaryTransferCubeEE-v0",
#             obs_type=obs_type,box_size=[0.02,0.02,0.02],box_color=[0,1,0,1])
#env = gym.make("gym_aloha/AlohaTransferCube-v0",
#               obs_type=obs_type)

observation, info = env.reset()
frames = []

cam_list = ["top"]
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube' or env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

# setup plotting
if obs_type=='pixels_agent_pos':
    im_observation=observation['pixels']
plt_imgs = plot_observation_images(im_observation, cam_list)
plt.pause(0.02)

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render() #uses top or cam_high
    frames.append(image)

    if obs_type=='pixels_agent_pos':
        observation=observation['pixels']
    for i in range(len(cam_list)): #anr added
        plt_imgs[i].set_data(observation[cam_list[i]])

    plt.pause(0.02)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
