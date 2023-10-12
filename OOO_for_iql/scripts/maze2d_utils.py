######################################################
#                   maze2d_utils.py                  #
#       Code to generate goal missing datasets       #
######################################################

import os
from typing import Dict, Optional, Tuple
import d4rl
import gym
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def generate_downsampled_dataset(
    env_name, num_samples: int, save_path: Optional[str], random_seed: int = 0
) -> Dict[str, np.ndarray]:
    """
    Generate a downsampled dataset for a given environment.

    Args:
        env_name (str): Name of the environment.
        num_samples (int): Number of samples in the downsampled dataset.
        save_path (Optional[str]): Path to save the downsampled dataset.
        random_seed (int): Random seed.

    Returns:
        Dict[str, np.ndarray]: Downsampled dataset.
    """
    np.random.seed(random_seed)
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    if num_samples is None:
        assert save_path is None
        return dataset
    sampled_indices = np.random.choice(
        len(dataset["rewards"]), num_samples, replace=False
    )
    sampled_dataset = {}
    for key in dataset.keys():
        sampled_dataset[key] = dataset[key][sampled_indices]

    if save_path is not None:
        np.savez_compressed(save_path, **sampled_dataset)
    return sampled_dataset


def plot_maze2d_dataset(dataset_path: str, save_path: str):
    """
    Plot a maze2d dataset.

    Args:
        dataset_path (str): Path to the dataset.
        save_path (str): Path to save the plot.
    """
    plt.cla()
    dataset = np.load(dataset_path)
    x_positions = dataset["observations"][:, 0]
    y_positions = dataset["observations"][:, 1]
    plt.scatter(x_positions, y_positions, s=1)
    plt.savefig(save_path)


def generate_dataset_downsampling_goal_radius(
    goal_radius: float,
    save_path: Optional[str],
    num_samples: int,
    additional_goal_radius_downsampling_rate: float,
    random_seed: int = 0,
    goal_location: Tuple[float, float] = (7, 9),
) -> Dict[str, np.ndarray]:
    """
    Generate a downsampled dataset for a given goal radius.

    Args:
        goal_radius (float): Goal radius.
        save_path (Optional[str]): Path to save the downsampled dataset.
        num_samples (int): Number of samples in the downsampled dataset.
        additional_goal_radius_downsampling_rate (float): Rate at which samples within the goal radius are downsampled. This is in addition to the downsampling rate of the entire dataset.
        random_seed (int): Random seed.
        goal_location (Tuple[float, float]): Tuple of (row, col) representing the location of the goal in the grid.

    Returns:
        Dict[str, np.ndarray]: Downsampled dataset.
    """
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.random.seed(random_seed)
    dataset = generate_downsampled_dataset(
        "maze2d-large-v1", num_samples, None, random_seed
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
