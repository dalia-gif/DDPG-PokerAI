import argparse
import datetime
import glob
import math
import os
import random
from collections import deque, namedtuple
from copy import deepcopy
from itertools import count
from pathlib import Path
from pprint import pprint

import colored as cl
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pokerenv.table import Table

from utils.agents import DDPG
from utils.model import DDPGActor, DDPGCritic
from utils.replaybuffer import ReplayBuffer

# Select a GPU when CUDA is available; otherwise, use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global training configuration.
SEED = 54321
HIGHT = 50
LOW = 10


def reset_seed():
    """Set random seeds to improve the reproducibility of training."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def str2bool(v):
    """
    Convert a string argument into a boolean value.

    Accepted true values: yes, true, t, y, 1.
    Accepted false values: no, false, f, n, 0.

    Args:
        v: Boolean value or string representation of a boolean.

    Returns:
        bool: The converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value is not recognized.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def range_limited_float_type(arg):
    """
    Parse a floating-point argument that must be non-positive.

    Args:
        arg: String representation of the floating-point value.

    Returns:
        float: The parsed value.

    Raises:
        argparse.ArgumentTypeError: If parsing fails or the value is positive.
    """
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f > 0:
        raise argparse.ArgumentTypeError(
            f"Must be a non-positive floating point number"
        )
    return f


if __name__ == "__main__":
    # Define the command-line interface for the training run.
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train DDPG agent to play poker",
    )

    parser.add_argument(
        "--players",
        "-p",
        type=int,
        default=2,
        help="Number of players in the game",
    )
    parser.add_argument(
        "--num_episodes",
        "-n",
        type=int,
        default=300,
        help="Number of episodes to train",
    )
    parser.add_argument(
        "--solve_criteria",
        "-s",
        type=float,
        default=100,
        help="Solve criteria",
    )
    parser.add_argument(
        "--ratio",
        "-r",
        type=float,
        default=1.0,
        help="Ratio of human experience",
    )
    parser.add_argument(
        "--replacable",
        "-re",
        type=str2bool,
        default=False,
        help="Use human experience",
    )
    parser.add_argument(
        "--punishment",
        "-pu",
        type=str2bool,
        default=True,
        help="Punishment for invalid action",
    )
    parser.add_argument(
        "--logfile",
        "-l",
        type=str,
        # default=f"{PLAYERS}_player.log",
        help="Log file",
    )

    # Read the training configuration from the command line.
    PLAYERS = parser.parse_args().players
    print(
        cl.Fore.red,
        parser.parse_args().punishment,
        cl.Style.reset,
    )

    # Configure the penalty applied to invalid actions.
    PENALTY = -HIGHT * PLAYERS * 10 if parser.parse_args().punishment else 0
    print(
        cl.Fore.red,
        f"Training DDPG agent to play poker with {PLAYERS} players and invalid action penalty {PENALTY}",
        cl.Style.reset,
    )

    # Create a timestamped directory for training outputs.
    LogDIR = Path(
        f"logs/{PLAYERS}_result_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    LogDIR.mkdir(exist_ok=False, parents=True)

    # Initialize the poker environment used during training.
    env = Table(
        n_players=PLAYERS,
        invalid_action_penalty=PENALTY,
        stack_high=HIGHT,
        stack_low=LOW,
        track_single_player=False,
        hand_history_location=None,
    )

    # Set learning rates for the actor and critic networks.
    actor_lr = 1e-4
    critic_lr = 1e-3

    # Initialize random seeds and the replay buffer.
    reset_seed()
    buffer = ReplayBuffer(
        10_000,
        ratio=parser.parse_args().ratio,
        replacable=parser.parse_args().replacable,
    )

    # Build the actor and critic networks from the environment dimensions.
    actor = DDPGActor(
        env.observation_space.shape[0], env.action_space[0].n + 1, 1, device=device
    )
    critic = DDPGCritic(
        env.observation_space.shape[0], env.action_space[0].n + 1, device=device
    )

    # Initialize the DDPG agent and configure its training parameters.
    agent = DDPG(
        buffer,
        actor,
        critic,
        actor_lr,
        critic_lr,
        # solve_criteria=100,
        solve_criteria=parser.parse_args().solve_criteria,
        # episode_limit=300_000,
        episode_limit=parser.parse_args().num_episodes,
        device=device,
        filter_invalid_actions=PENALTY == 0,
    )

    # Train the agent and collect the total reward for each episode.
    ddpg_total_rewards = agent.train(env)

    # Plot and save the training reward curve.
    plt.plot(ddpg_total_rewards)
    plt.title(f"DDPG Training for Poker Holdem {PLAYERS} Players")
    plt.xlabel("Episode")
    plt.ylabel("Player 0 Total Rewards")
    plt.savefig(
        f'{LogDIR}/{PLAYERS:02d}_ddpg_total_rewards{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png'
    )

    # Create a new environment configured to save hand histories.
    env = Table(
        n_players=PLAYERS,
        invalid_action_penalty=PENALTY,
        stack_high=HIGHT,
        stack_low=LOW,
        track_single_player=False,
        hand_history_location=f"{LogDIR}/",
    )

    # Evaluate the trained agent over ten games and print each result.
    for _ in range(10):
        pprint(agent.play(env))
        print("-" * 100)

    # Copy the specified log file into the training output directory.
    os.system(f"cp -r {parser.parse_args().logfile} {LogDIR}")