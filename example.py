import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
from gym_aloha.utils import plot_observation_images     #anr
import matplotlib.pyplot as plt                         #anr
from gym_aloha.tasks.sim import BOX_POSE

obs_type='pixels_agent_pos' #'pixels_agent_pos' #'pixels' is default
env = gym.make("gym_aloha/TrossenAIStationaryTransferCube-v0",
             obs_type=obs_type,
             box_size=[0.02,0.02,0.02],
             box_pos=[0.0,0.0,0.02],
             box_color=[0,1,0,1],
             #tabletop='wood',
             arms_pos=[-0.4575, 0.0, 0.02, 0.4575, 0.0, 0.02], #best match to real?
             arms_ref=[0,-0.015,0.015,0,0,0,0,-0.025,0.025,0,0,0], #best match to real?
             )
#env = gym.make("gym_aloha/TrossenAIStationaryTransferCubeEE-v0",
#             obs_type=obs_type,box_size=[0.05,0.05,0.05],box_color=[0,1,0,1])
#env = gym.make("gym_aloha/AlohaTransferCube-v0",
#               obs_type=obs_type)

random_box_position=False
if random_box_position:
    observation, info = env.reset()
else:
    #to keep the box at its gym.make() position:
    BOX_POSE[0]=env.unwrapped._env.physics.named.data.qpos[-7:]
    observation, info = env.reset(seed=None,options='do_not_reset_BOX_POSE')
frames = []

cam_list = ["top"]
if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube' or env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

###################################

#hack to set the initial arm poses:
# #self.home_pose = [0, np.pi/2, np.pi / 6, np.pi / 5, 0, 0, 0.05] #0] #anr
#action = np.array([0, np.pi/2, np.pi / 6, np.pi / 5, 0, 0, 0.05, 0, np.pi/2, np.pi / 2, np.pi / 5, 0, 0, 0.05])
#action = np.array([0, np.pi/2, np.pi / 6, np.pi / 5, 0, 0, 0.05, 0, np.pi/2, np.pi/6, np.pi / 5, 0, 0, 0.05])
#action = np.array([0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0, 0.05, 0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0, 0.05])
#for j in range(100):
#    observation, reward, terminated, truncated, info = env.step(action)

###################################

# setup plotting #anr added
if obs_type=='pixels_agent_pos':
    im_observation=observation['pixels']
plt_imgs = plot_observation_images(im_observation, cam_list)
plt.pause(0.02)

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
