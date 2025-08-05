import numpy as np

from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
from dm_control.mujoco.engine import Physics
import collections

def sample_box_pose(seed=None):
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_box_pose_trossen_ai_stationary(seed=None) -> np.ndarray:
    """
    Generate a random pose for a cube within predefined position ranges.

    :return: A 7D array containing the sampled position ``[x, y, z, w, x, y, z]`` representing the
        cube's position and orientation as a quaternion.
    """
    x_range = [-0.1, 0.2]
    y_range = [-0.15, 0.15]
    z_range = [0.0125, 0.0125]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    #cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose(seed=None):
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

def plot_observation_images(observation: dict, cam_list: list[str]) -> list[AxesImage]:
    """
    Plot observation images from multiple camera viewpoints.

    :param observation: The observation data containing images.
    :param cam_list: List of camera names used for capturing images.
    :return: A list of AxesImage objects for dynamic updates.
    """
    #images = observation.get("images", {}) #anr, removed
    #if len(images)==0:      #anr in this case the observation is the top camera
    #    images=observation  #anr
    images = {key: observation[key] for key in cam_list if key in observation} #anr added

    # Define the layout based on the provided camera list
    num_cameras = len(cam_list)

    if num_cameras == 4:
        cols = 2
        rows = 2
    else:
        cols = min(3, num_cameras)  # Maximum of 3 columns
        rows = (num_cameras + cols - 1) // cols  # Compute rows dynamically
    _, axs = plt.subplots(rows, cols, figsize=(10, 10))
    axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else np.array(axs).flatten() #[axs] anr

    plt_imgs: list[AxesImage] = []
    titles = {
        "cam_high": "Camera High",
        "cam_low": "Camera Low",
        "cam_teleop": "Teleoperator POV",
        "cam_left_wrist": "Left Wrist Camera",
        "cam_right_wrist": "Right Wrist Camera",
        "top": "Top Camera",
    }

    for i, cam in enumerate(cam_list):
        if cam in images:
            plt_imgs.append(axs[i].imshow(images[cam]))
            axs[i].set_title(titles.get(cam, cam))

    for ax in axs.flat:
        ax.axis("off")

    plt.ion()
    return plt_imgs

def get_observation_base(
    physics: Physics,
    cam_list: list[str],
    on_screen_render: bool = True,
) -> collections.OrderedDict:
    """
    Capture image observations from multiple cameras in the simulation.

    :param physics: The simulation physics instance.
    :param cam_list: List of camera names to capture images from.
    :param on_screen_render: Whether to capture images from cameras, defaults to ``True``.
    :return: A dictionary containing image observations.
    """
    obs: collections.OrderedDict = collections.OrderedDict()
    if on_screen_render:
        obs["images"] = dict()
        for cam in cam_list:
            obs["images"][cam] = physics.render(height=480, width=640, camera_id=cam)
    return obs
