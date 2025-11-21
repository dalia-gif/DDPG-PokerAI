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
    this_story = []
    IM = list(episode["known_cards"].keys())[0]
    myIndex = list(episode["stacks"].keys()).index(IM)
    money_state = episode["stacks"].values()
    names = list(episode["stacks"].keys())
    names_rev = {name: i for i, name in enumerate(names)}
    names = {i: name for i, name in enumerate(names)}
    # print(cl.Fore.yellow, names, cl.Style.reset)
    env = Table(
        n_players=len(episode["stacks"]),
        player_names=names,
        stack_low=0,
        stack_high=1000,
        hand_history_location=None,
    )
    env.reset(shuffle=False)

    # TODO: not verify yet
    #  load cards
    if "board" in episode:
        # print(episode["board"])
        env.cards = [Card.new(card.replace("10", "T")) for card in episode["board"]]

        # c = input("Enter the cards for the board: ")

    # load myplayer cards
    for player in env.all_players:
        if player.name == IM:
            player.hole_cards = episode["known_cards"][IM]

    # load money
    for player in env.all_players:
        for key, value in episode["stacks"].items():
            if player.name == key:
                player.stack = value

    for action in episode["actions"]:
        for i, player in enumerate(env.all_players):
            if player.name == action["player"]:
                acting_player = player
                acting_index = i
                # print(player.name, action["player"])
                break
        env.current_player_i = env.next_player_i = acting_index
        player.state = PlayerState.ACTIVE
        state = env._get_observation(acting_player)
        valid_actions = env._get_valid_actions(acting_player)

        if "check" in action["action"].lower():
            a = 0
            v = 0
        elif "folds" in action["action"].lower():
            a = 1
            v = 0
        elif "raise" in action["action"].lower() or "bet" in action["action"].lower():
            a = 2
            v = action["amount"]
            v = min(
                max(v, valid_actions["bet_range"][0]), valid_actions["bet_range"][1]
            )
        elif "call" in action["action"].lower():
            a = 3
            v = 0
        else:
            raise ValueError(f"Invalid action {action['action']}")
        a_coombine = np.array([0, 0, 0, 0, v], dtype=np.float32).flatten()
        a_coombine[a] = 1
        table_action = Action(PlayerAction(action_list[int(a)]), v)
        # print(
        #     cl.Fore.green,
        #     f"Player {player.name:20s}: {action['player']:20s} Action: {action_list[a].name:20s}, {v} {player.state.name}",
        #     cl.Style.reset,
        # )

        new_state, reward, done, _ = env.step(table_action)

        # find the action index
        acting_player = int(state[indices.ACTING_PLAYER])

        reward = float(reward[acting_player])
        if action["player"] == IM:
            this_story.append((state, a_coombine, reward, new_state, done))
        # state, action, reward, new_state, done
    return this_story


def load_moemory(file):
    json_file = open(file, "r")
    json_file = json.load(json_file)
    # split the file into episodes
    all_story = []
    count = 0
    for episode in tqdm(json_file[21:]):
        # print("-" * 200)
        try:
            this_story = load_episode(episode)
            if len(this_story) > 0:
                all_story.extend(this_story)
            count += 1
        except Exception as e:

            # print(f"{episode['game_id']} failed")
            # for action in episode["actions"]:
            #     if "amount" in action:
            #         print(
            #             cl.Fore.red,
            #             f'{action["player"]:20s}    {action["action"]:20s}   {action["amount"]}',
            #             cl.Style.reset,
            #         )
            #     else:
            #         print(
            #             cl.Fore.red,
            #             f'{action["player"]:20s}    {action["action"]:20s}   ',
            #             cl.Style.reset,
            #         )

            # raise e
            # print(e)
            # print("Error in episode")
            continue
    print(f"Total episodes: {count}")
    return all_story


if __name__ == "__main__":
    file = "datasets/parsed_poker_games.json"
    all_story = load_moemory(file)
    torch.save(all_story, "datasets/parsed_poker_games.pt")
