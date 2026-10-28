"""Load human poker hand histories and convert them into replay transitions.

The loader reconstructs each recorded episode in the PokerEnv table, replays
its actions, and stores transitions for the selected player in the format
expected by the DDPG replay buffer:
(state, action, reward, next_state, done).
"""

import glob
import json
import os
from pathlib import Path

import colored as cl
import numpy as np
import pokerenv.obs_indices as indices
import torch
import torch.nn as nn
from pokerenv.common import Action, PlayerAction, PlayerState, action_list
from pokerenv.player import Player
from pokerenv.table import Table
from tqdm import tqdm
from treys import Card, Evaluator


def load_episode(episode):
    """Reconstruct one recorded poker episode and extract player transitions.

    Args:
        episode: Parsed episode dictionary containing player stacks, known
            cards, actions, and optionally the community-card board.

    Returns:
        A list of replay-buffer transitions for the player whose name is taken
        from the first key in ``episode["known_cards"]``. Each transition is a
        tuple of ``(state, action, reward, next_state, done)``.

    Notes:
        The environment is initialized with a fixed stack range and without
        shuffling so the supplied cards and action history can be replayed.
        Only transitions belonging to the selected player are retained.
    """
    this_story = []

    # The first player listed in known_cards is the player for whom transitions
    # will be collected from this episode.
    IM = list(episode["known_cards"].keys())[0]
    myIndex = list(episode["stacks"].keys()).index(IM)

    # Preserve the input player order when mapping player names to environment
    # indices. The reverse mapping is retained as in the original code.
    names = list(episode["stacks"].keys())
    names_rev = {name: i for i, name in enumerate(names)}
    names = {i: name for i, name in enumerate(names)}

    # Create a table matching the players and starting stacks in the episode.
    env = Table(
        n_players=len(episode["stacks"]),
        player_names=names,
        stack_low=0,
        stack_high=1000,
        hand_history_location=None,
    )
    env.reset(shuffle=False)

    # Restore the recorded community cards, if they are provided.
    if "board" in episode:
        env.cards = [Card.new(card.replace("10", "T")) for card in episode["board"]]

    # Restore the known private cards for the selected player.
    for player in env.all_players:
        if player.name == IM:
            player.hole_cards = episode["known_cards"][IM]

    # Restore each player's recorded stack before replaying the actions.
    for player in env.all_players:
        for key, value in episode["stacks"].items():
            if player.name == key:
                player.stack = value

    # Replay the hand one action at a time to reconstruct the state observed
    # by the acting player and the resulting transition.
    for action in episode["actions"]:
        for i, player in enumerate(env.all_players):
            if player.name == action["player"]:
                acting_player = player
                acting_index = i
                break

        # Set the environment's current player to the player named in the log.
        env.current_player_i = env.next_player_i = acting_index
        player.state = PlayerState.ACTIVE
        state = env._get_observation(acting_player)
        valid_actions = env._get_valid_actions(acting_player)

        # Convert the action label in the human-readable log into the action
        # index used by PokerEnv, along with an optional bet amount.
        action_name = action["action"].lower()
        if "check" in action_name:
            a = 0
            v = 0
        elif "folds" in action_name:
            a = 1
            v = 0
        elif "raise" in action_name or "bet" in action_name:
            a = 2
            v = action["amount"]

            # Clamp the recorded amount to the legal betting interval reported
            # by the environment for this state.
            v = min(
                max(v, valid_actions["bet_range"][0]), valid_actions["bet_range"][1]
            )
        elif "call" in action_name:
            a = 3
            v = 0
        else:
            raise ValueError(f"Invalid action {action['action']}")

        # Encode the discrete action as a one-hot vector and append the amount.
        # The final element stores the bet amount (zero for non-betting actions).
        a_coombine = np.array([0, 0, 0, 0, v], dtype=np.float32).flatten()
        a_coombine[a] = 1
        table_action = Action(PlayerAction(action_list[int(a)]), v)

        # Advance the environment using the reconstructed action.
        new_state, reward, done, _ = env.step(table_action)

        # PokerEnv returns rewards for all players; select the reward associated
        # with the player who was acting in the observation just constructed.
        acting_player_index = int(state[indices.ACTING_PLAYER])
        reward = float(reward[acting_player_index])

        # Store only transitions for the selected player. Other players' actions
        # are replayed to keep the environment state synchronized.
        if action["player"] == IM:
            this_story.append((state, a_coombine, reward, new_state, done))

    return this_story


def load_moemory(file):
    """Load a JSON file of poker episodes and convert them to transitions.

    Args:
        file: Path to the JSON file containing parsed poker games.

    Returns:
        A list of transitions suitable for adding to the DDPG replay buffer.

    Episodes that raise an exception during reconstruction are skipped, as in
    the original implementation.
    """
    with open(file, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)

    # Accumulate transitions from successfully reconstructed episodes.
    all_story = []
    count = 0

    # Preserve the original behavior of skipping the first 21 records.
    for episode in tqdm(json_data[21:]):
        try:
            this_story = load_episode(episode)
            if len(this_story) > 0:
                all_story.extend(this_story)
            count += 1
        except Exception:
            # Keep processing the remaining dataset if an individual episode
            # cannot be reconstructed, matching the original loader behavior.
            continue

    print(f"Total episodes: {count}")
    return all_story


if __name__ == "__main__":
    # Convert the parsed human-game dataset to a serialized PyTorch file.
    file = "datasets/parsed_poker_games.json"
    all_story = load_moemory(file)
    torch.save(all_story, "datasets/parsed_poker_games.pt")
