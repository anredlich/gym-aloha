#to set arm poses and then display in the mujoco viewer

import mujoco
import mujoco.viewer
import gymnasium as gym
import numpy as np
import gym_aloha
from gym_aloha.tasks.sim import BOX_POSE
from gym_aloha.utils import plot_observation_images
import matplotlib.pyplot as plt 

#####################data########################

i=3 #0-4 tests 'calibration right # v#.png'
i=5 #5=sim 'cameras left right 40mm box.png' 6='sim cameras low high 40mm box.png'
i=-1 #default robot positions

box_size=[0.02,0.02,0.02]
box_pos=[0.0,0.0,0.02]
box_color=[1,0,0,1]

joint_angles_left=None
joint_angles_right=None

# Your specific joint angles from the real robot
if i>=0:
    joint_angles_left = {
        'left/joint_0': 0.0,
        'left/joint_1': 0.0,
        'left/joint_2': 0.0,
        'left/joint_3': 0.0,
        'left/joint_4': 0.0,
        'left/joint_5': 0.0,
        'left/left_carriage_joint': 0.0,    # left gripper
        'left/right_carriage_joint': 0.0,    # right gripper
    }

if i==0:
    box_size=[0.02,0.02,0.002]
    box_pos=[0.0,0.0,0.002]
    box_color=[0, 1, 0,1]
    joint_angles_right = {
        #calibration right 0 v2.png (0,0)
        'right/joint_0': 0.047494,     
        'right/joint_1': 1.8977,
        'right/joint_2': 0.93366,
        'right/joint_3': 0.19551,
        'right/joint_4': 0.078393,
        'right/joint_5': 0.033761,
        'right/left_carriage_joint': 0.0,
        'right/right_carriage_joint': 0.0,
    }
elif i==1:
    box_size=[0.02,0.02,0.002]
    box_pos=[-0.10,0.10,0.002]
    box_color=[0, 1, 0,1]
    joint_angles_right = {
        #calibration right 1 v1.png (-0.10,0.10)
        'right/joint_0': -0.19284,     
        'right/joint_1': 2.1956,
        'right/joint_2': 1.4132,
        'right/joint_3': 0.075341,
        'right/joint_4': -0.078775,
        'right/joint_5': 0.00019074,
        'right/left_carriage_joint': 0.0,
        'right/right_carriage_joint': 0.0,
    }
elif i==2:
    box_size=[0.02,0.02,0.002]
    box_pos=[0.10,0.10,0.002]
    box_color=[0, 1, 0,1]
    joint_angles_right = {
        #calibration right 2 v1.png (0.10,0.10)
        'right/joint_0': -0.29927,     
        'right/joint_1': 1.6119,
        'right/joint_2': 0.58919,
        'right/joint_3': 0.22488,
        'right/joint_4': -0.071145,
        'right/joint_5': -0.20848,
        'right/left_carriage_joint': 0.0,
        'right/right_carriage_joint': 0.0,
    }
elif i==3:
    box_size=[0.02,0.02,0.002]
    box_pos=[-0.10,-0.10,0.002]
    box_color=[0, 1, 0,1]
    joint_angles_right = {
        #calibration right 3 v1.png (-0.10,-0.10)
        'right/joint_0': 0.2554,     
        'right/joint_1': 2.2196,
        'right/joint_2': 1.459,
        'right/joint_3': 0.021172,
        'right/joint_4': 0.21992,
        'right/joint_5': 0.1276,
        'right/left_carriage_joint': 0.0,
        'right/right_carriage_joint': 0.0,
    }
elif i==4:
    box_size=[0.02,0.02,0.002]
    box_pos=[0.10,-0.10,0.002]
    box_color=[0, 1, 0,1]
    joint_angles_right = {
        #calibration right 4 v1.png (0.10,-0.10)
        'right/joint_0': 0.41943,     
        'right/joint_1': 1.6623,
        'right/joint_2': 0.59224,
        'right/joint_3': 0.27752,
        'right/joint_4': 0.2985,
        'right/joint_5': 0.25845,
        'right/left_carriage_joint': 0.0,
        'right/right_carriage_joint': 0.0,
    }
elif i==5:
    #sim cameras left right 40mm box.png
    #real cameras left right 40mm box.png
    #arms_pos=[-0.4575, 0.0, 0.02, 0.4575, 0.0, 0.02], #base position, left, right; default=[+-4575 -0.019 0.02]
    #arms_ref=[0,-0.015,0.015,0,0,0,0,-0.025,0.025,0,0,0], #left joints 0-5 ref, right joints 0-5 ref; default=[all zeros]
    joint_angles_left = {
        'left/joint_0': 0.0,
        'left/joint_1': np.pi/2,
        'left/joint_2': np.pi/6,
        'left/joint_3': np.pi/5,
        'left/joint_4': 0.0,
        'left/joint_5': 0.0,
        'left/left_carriage_joint': 0.0396,    # left gripper
        'left/right_carriage_joint': 0.0396,    # right gripper
    }
    joint_angles_right = {
        'right/joint_0': 0.0,     
        'right/joint_1': np.pi/2,
        'right/joint_2': np.pi/6,
        'right/joint_3': np.pi/5,
        'right/joint_4': 0.0,
        'right/joint_5': 0.0,
        'right/left_carriage_joint': 0.0396,
        'right/right_carriage_joint': 0.0396,
    }
elif i==6:
    #sim cameras low high 40mm box.png
    #real cameras low high 40mm box.png
    #arms_pos=[-0.4575, 0.0, 0.02, 0.4575, 0.0, 0.02], #base position, left, right; default=[+-4575 -0.019 0.02]
    #arms_ref=[0,-0.015,0.015,0,0,0,0,-0.025,0.025,0,0,0], #left joints 0-5 ref, right joints 0-5 ref; default=[all zeros]
    joint_angles_left = {
        'left/joint_0': 0.0,
        'left/joint_1': np.pi/3,
        'left/joint_2': np.pi/6,
        'left/joint_3': np.pi/5,
        'left/joint_4': 0.0,
        'left/joint_5': 0.0,
        'left/left_carriage_joint': 0.0396,    # left gripper
        'left/right_carriage_joint': 0.0396,    # right gripper
    }
    joint_angles_right = {
        'right/joint_0': 0.0,     
        'right/joint_1': np.pi/3,
        'right/joint_2': np.pi/6,
        'right/joint_3': np.pi/5,
        'right/joint_4': 0.0,
        'right/joint_5': 0.0,
        'right/left_carriage_joint': 0.0396,
        'right/right_carriage_joint': 0.0396,
    }

#################################################

#direct:
# model = mujoco.MjModel.from_xml_path('gym_aloha/assets/trossen_ai_scene_joint.xml')
# data = mujoco.MjData(model)

box_size=[0.02,0.02,0.02]
box_pos=[0.0,0.0,0.02] #[0.0,0.0,-0.02]
box_color=[1,0,0,1]

obs_type='pixels_agent_pos'
env = gym.make("gym_aloha/TrossenAIStationaryTransferCube-v0", #TrossenAIStationaryTransferCube-v0", #TrossenAIStationaryTransferCubeEE-v0
             obs_type=obs_type,
             box_size=box_size,
             box_pos=box_pos,
             box_color=[1,0,0,1], #[0.86, 0.18, 0.18,1], #box_color,
             arms_pos=[-0.4575, 0.0, 0.02, 0.4575, 0.0, 0.02], #base position, left, right; default=[+-4575 -0.019 0.02]
             arms_ref=[0,-0.015,0.015,0,0,0,0,-0.025,0.025,0,0,0], #left joints 0-5 ref, right joints 0-5 ref; default=[all zeros]
             tabletop='my_desktop', #'my_desktop', #'wood',
             #backdrop='my_backdrop', #'none' 'my_backdrop'
             #lighting=[[0.1,0.1,0.1],[-0.5,0.5,0.5]], #[[0.3,0.3,0.3],[0.3,0.3,0.3]], #[[0.1,0.1,0.1],[0.5,0.5,0.5]], #[[0.3,0.3,0.3],[0.3,0.3,0.3]], #[[0.1,0.1,0.1],[0.5,0.5,0.5]], #table lighting, ambient lighting
            )

BOX_POSE[0]=env.unwrapped._env.physics.named.data.qpos[-7:]
observation, info = env.reset() #seed=None,options='do_not_reset_BOX_POSE')
print(f"home_pose={observation['agent_pos']}")

# if joint_angles_left==None or joint_angles_right==None:
#     joint_angles_left = {
#         'left/joint_0': observation['agent_pos'][0],
#         'left/joint_1': observation['agent_pos'][1],
#         'left/joint_2': observation['agent_pos'][2],
#         'left/joint_3': observation['agent_pos'][3],
#         'left/joint_4': observation['agent_pos'][4],
#         'left/joint_5': observation['agent_pos'][5],
#         'left/left_carriage_joint': observation['agent_pos'][6],    # left gripper
#         'left/right_carriage_joint': observation['agent_pos'][6],    # right gripper
#     }
#     joint_angles_right = {
#         'right/joint_0': observation['agent_pos'][7],     
#         'right/joint_1': observation['agent_pos'][8],
#         'right/joint_2': observation['agent_pos'][9],
#         'right/joint_3': observation['agent_pos'][10],
#         'right/joint_4': observation['agent_pos'][11],
#         'right/joint_5': observation['agent_pos'][12],
#         'right/left_carriage_joint': observation['agent_pos'][13],
#         'right/right_carriage_joint': observation['agent_pos'][13],
#     }

# # Combine all joint angles
# all_joint_angles = {**joint_angles_left, **joint_angles_right}

# # Set initial joint positions and control targets
# for joint_name, angle in all_joint_angles.items():
#     # Set position
#     #joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
#     joint_id = env.unwrapped._env.physics.model.name2id(joint_name, 'joint')
#     if joint_id >= 0:
#         env.unwrapped._env.physics.data.qpos[joint_id] = angle
    
#     # Set control target
#     #actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
#     if env.unwrapped.task!='trossen_ai_stationary_transfer_cube_ee':
#         actuator_name=joint_name
#         if joint_name=='left/left_carriage_joint': 
#             actuator_name='left/joint_gl'
#         if joint_name=='left/right_carriage_joint': 
#             actuator_name='left/joint_gr'
#         if joint_name=='right/left_carriage_joint': 
#             actuator_name='right/joint_gl'
#         if joint_name=='right/right_carriage_joint': 
#             actuator_name='right/joint_gr'
#         actuator_id = env.unwrapped._env.physics.model.name2id(actuator_name, 'actuator')
#         if actuator_id >= 0:
#             env.unwrapped._env.physics.data.ctrl[actuator_id] = angle

# # Print joint positions for verification
# print("Left arm joint positions:")
# for joint_name, angle in joint_angles_left.items():
#     print(f"  {joint_name}: {angle:.4f} rad ({angle * 180/3.14159:.2f} deg)")

# print("\nRight arm joint positions:")
# for joint_name, angle in joint_angles_right.items():
#     print(f"  {joint_name}: {angle:.4f} rad ({angle * 180/3.14159:.2f} deg)")

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


