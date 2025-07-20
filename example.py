import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
from gym_aloha.utils import plot_observation_images     #anr
import matplotlib.pyplot as plt                         #anr

obs_type='pixels_agent_pos' #'pixels_agent_pos' #'pixels' is default
env = gym.make("gym_aloha/TrossenAIStationaryTransferCube-v0",
             obs_type=obs_type)
#env = gym.make("gym_aloha/AlohaTransferCube-v0",
#               obs_type=obs_type)
observation, info = env.reset()
frames = []

cam_list = ["top"]
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

# setup plotting #anr added
if obs_type=='pixels_agent_pos':
    observation=observation['pixels']
plt_imgs = plot_observation_images(observation, cam_list)

for _ in range(100): #1000
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
