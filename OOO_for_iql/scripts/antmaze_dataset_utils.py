######################################################
#              antmaze_dataset_utils.py              #
#       Code to generate goal missing datasets       #
######################################################
import os
from typing import Dict, Optional, Tuple
import d4rl
import gym
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scripts.maze2d_utils import generate_downsampled_dataset


def generate_dataset_downsampling_goal_radius(
    goal_radius: float,
    dataset: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
    num_samples: int = None,
    additional_goal_radius_downsampling_rate: float = 0,  # I.e. remove all goal data.
    random_seed: int = 0,
    goal_grid_location: Tuple[int, int] = (7, 9),
) -> Dict[str, np.ndarray]:
    """
    Generate a downsampled dataset for a given goal radius.

    Args:
        goal_radius (float): Goal radius.
        dataset (Optional[Dict[str, np.ndarray]]): Dataset to downsample. If None, load the antmaze-large-diverse-v1 dataset.
        save_path (Optional[str]): Path to save the downsampled dataset.
        num_samples (int): Number of samples in the downsampled dataset.
        additional_goal_radius_downsampling_rate (float): Rate at which samples within the goal radius are downsampled. This is in addition to the downsampling rate of the entire dataset.
        random_seed (int): Random seed.
        goal_grid_location (Tuple[int, int]): Tuple of (row, col) representing the location of the goal in the grid.

    Returns:
        Dict[str, np.ndarray]: Downsampled dataset.
    """
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.random.seed(random_seed)

    env = gym.make("antmaze-large-diverse-v2")
    # Check that the goal location is valid.
    assert goal_grid_location[0] < len(env.unwrapped._maze_map) and goal_grid_location[
        1
    ] < len(env.unwrapped._maze_map[0])
    # Convert goal grid location to xy location.
    goal_location = env.unwrapped._rowcol_to_xy(goal_grid_location)
    print(f"Goal xy location: {goal_location}")

    if dataset is None:
        dataset = generate_downsampled_dataset(
            "antmaze-large-diverse-v2", num_samples, None, random_seed
        )
    samples_within_goal_radius_indices = np.where(
        np.linalg.norm(dataset["observations"][:, :2] - goal_location, axis=1)
        <= goal_radius
    )[0]
    samples_not_within_goal_radius_indices = np.where(
        np.linalg.norm(dataset["observations"][:, :2] - goal_location, axis=1)
        > goal_radius
    )[0]
    num_samples_within_goal_radius = int(
        len(samples_within_goal_radius_indices)
        * additional_goal_radius_downsampling_rate
    )
    sampled_goal_radius_indices = np.random.choice(
        samples_within_goal_radius_indices,
        num_samples_within_goal_radius,
        replace=False,
    )
    sampled_indices = np.concatenate(
        [samples_not_within_goal_radius_indices, sampled_goal_radius_indices]
    )
    sampled_dataset = {}
    for key in dataset.keys():
        sampled_dataset[key] = dataset[key][sampled_indices]

    if save_path is not None:
        np.savez_compressed(save_path, **sampled_dataset)
        # Save a plot of the dataset, with a circle indicating the goal radius, and a red dot for
        # the goal.
        plt.cla()
        x_positions = sampled_dataset["observations"][:, 0]
        y_positions = sampled_dataset["observations"][:, 1]
        plt.scatter(x_positions, y_positions, s=1)
        circle = plt.Circle(
            goal_location, goal_radius, color="r", fill=False, linewidth=1
        )
        plt.gca().add_patch(circle)
        plt.scatter(goal_location[0], goal_location[1], color="r", s=1)
        plt.savefig(save_path.replace(".npz", ".png"))

    return sampled_dataset
