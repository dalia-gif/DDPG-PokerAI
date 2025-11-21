from copy import deepcopy

import colored as cl
import numpy as np
import pokerenv.obs_indices as indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pokerenv.common import Action, PlayerAction, PlayerState, action_list
from pokerenv.table import Table
from tqdm import tqdm

from .model import DDPGActor, DDPGCritic
from .replaybuffer import ReplayBuffer

SEED = 42


class DDPG:
    def __init__(
        self,
        buffer: ReplayBuffer,
        actor: DDPGActor,
        critic: DDPGCritic,
        actor_lr: float,
        critic_lr: float,
        solve_criteria: float,
        episode_limit: int,
        filter_invalid_actions: bool = True,
        device=None,
    ):
        if device is None:
            device = "cpu"
        self.device = device
        self.buffer = buffer

        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        self.solve_criteria = solve_criteria
        self.episode_limit = episode_limit
        self.filter_invalid_actions = filter_invalid_actions

    @staticmethod
    def soft_target_update(net: nn.Module, target_net: nn.Module, tau: float):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self):
        # Feel free to change these hyperparameters.
        BATCH_SIZE = 4096
        DISCOUNT_FACTOR = 0.99
        TAU = 0.005

        states, actions, rewards, next_states, dones = zip(
            *self.buffer.sample(BATCH_SIZE)
        )
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        rewards = (
            torch.from_numpy(np.array(rewards, dtype=np.float32))
            .unsqueeze(1)
            .to(self.device)
        )
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(
            self.device
        )
        dones = (
            torch.from_numpy(np.array(dones, dtype=np.float32))
            .unsqueeze(1)
            .to(self.device)
        )

        with torch.no_grad():
            next_actions = self.target_actor(next_states.to(self.device))
            target_q_values = rewards + DISCOUNT_FACTOR * (
                1 - dones
            ) * self.target_critic(next_states, next_actions)

        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, target_q_values)

        # With loss above, update the critic network.
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Calculate actor loss and update the actor network.
        actor_loss = -self.critic(
            states.to(self.device), self.actor(states.to(self.device))
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        self.soft_target_update(self.actor, self.target_actor, TAU)
        self.soft_target_update(self.critic, self.target_critic, TAU)

    def interaction_step(self, state, env, noise=True, global_step=10000):
        action = self.actor(torch.from_numpy(state).to(self.device))
        action_raw = action.cpu().detach().numpy().flatten()
        # print(cl.Fore.red, action_raw.shape, cl.Style.reset)
        a = action_raw[:-1]
        v = action_raw[-1]

        if noise:
            # noise = np.random.normal(0, np.exp(-global_step / 100), size=a.shape)
            # a = np.clip(a + noise, 0, 1)
            a = gumbel_softmax(a, value=np.exp(-global_step / 100))
            noise = np.random.normal(0, np.exp(-global_step / 100), size=v.shape)
            # noise = np.random.normal(0, 0.2, size=v.shape)
            v = np.clip(v + noise, 0, 1)
            # action_raw = np.concatenate((a, v.reshape(1)))
        else:
            a = F.softmax(torch.from_numpy(a), dim=-1).numpy()

        state = torch.from_numpy(state)
        valid_actions = torch.argwhere(state[indices.VALID_ACTIONS] == 1).flatten()
        invalid_actions = torch.argwhere(state[indices.VALID_ACTIONS] != 1).flatten()
        # make invalid actions impossible
        if self.filter_invalid_actions:
            a[invalid_actions] = 0

        valid_bet_low = state[indices.VALID_BET_LOW].flatten().cpu().numpy()
        valid_bet_high = state[indices.VALID_BET_HIGH].flatten().cpu().numpy()

        chosen_action = PlayerAction(action_list[a.argmax()])
        bet_size = v * (valid_bet_high - valid_bet_low) + valid_bet_low

        # action_raw = np.concatenate((a, v.reshape(1)))

        action = Action(chosen_action, bet_size)
        new_state, reward, done, _ = env.step(action)
        acting_player = int(state[indices.ACTING_PLAYER])
        # for player in env.all_players:
        #     print(
        #         f"Player {player.name} stack: {player.stack} reward: {player.get_reward()}"
        #     )
        for i, player in enumerate(env.all_players):
            if player.name == "player_1":
                break
        player1_rewards = reward[i]
        if player1_rewards is None:
            player1_rewards = 0
        reward = reward[acting_player]
        reward = float(reward)

        return action_raw, reward, new_state, done, player1_rewards

    def train(self, env):
        total_rewards = []
        pbar = tqdm(range(1, self.episode_limit + 1))
        solved = False
        for episode in pbar:
            state, done = env.reset(), False
            self.rewards = []
            while not done:
                action, reward, new_state, done, player1_rewards = (
                    self.interaction_step(state, env, global_step=episode)
                )
                self.buffer.push((state, action, reward, new_state, done))
                self.update()
                state = new_state
                self.rewards.append(player1_rewards)

            # if done:
            #     # find tthe winner
            #     all_players = env.all_players
            #     winners = []
            #     for player in all_players:
            #         if player.state == PlayerState.ACTIVE:
            #             winners.append(player.name)
            #     print(f"Winner: {winners}")
            #     for winner in winners:
            #         state = env._get_observation(winner)
            #         new_state = env._get_observation(winner)
            #         a, v = np.array(winner.history[-1]["action"]), np.array(
            #             winner.history[-1]["value"]
            #         )
            #         action = np.concatenate((a, v))

            #         state = torch.from_numpy(state).to(self.device)
            #         action = torch.from_numpy(action).to(self.device)
            #         new_state = torch.from_numpy(new_state).to(self.device)
            #         self.buffer.push((state, action, reward, new_state, done))
            if len(self.rewards) != 0:
                total_reward = sum(self.rewards)
                total_rewards.append(total_reward)

            avg_rewards = np.mean(total_rewards[-100:])
            pbar.set_description(
                f"reward avg: {avg_rewards:.2f} min: {min(total_rewards):.2f} max: {max(total_rewards):.2f}"
            )
            if avg_rewards >= self.solve_criteria:
                print(f"\nSOLVED in {episode} episodes!")
                solved = True
                break

        # if not solved:
        #     raise Exception("Not solved. Please check your implementation!")

        return total_rewards

    def play(self, env):
        state, done = env.reset(), False

        # for displaying the stack of each player
        for player in env.all_players[: env.n_players]:
            print(f"Player {player.name} stack: {player.stack}")

        while not done:
            action, reward, new_state, done, player1_reward = self.interaction_step(
                state, env, noise=False
            )

            # # for displaying the action taken by each player
            # a = action[:-1]
            # v = action[-1]
            # action = PlayerAction(action_list[a.argmax()])
            # bet_size = (
            #     v * (state[indices.VALID_BET_HIGH] - state[indices.VALID_BET_LOW])
            #     + state[indices.VALID_BET_LOW]
            # )
            # action = Action(action, bet_size)
            # actiing_player = int(state[indices.ACTING_PLAYER])
            # if action.action_type == PlayerAction.BET:
            #     print(
            #         cl.Fore.green,
            #         f"Player {actiing_player} reward: {reward} acted: {action.action_type.name} bet_size: {bet_size}",
            #         cl.Style.reset,
            #     )
            # else:
            #     print(
            #         cl.Fore.green,
            #         f"Player {actiing_player} reward: {reward} acted: {action.action_type.name}",
            #         cl.Style.reset,
            #     )
            # # for player in env.all_players[: env.n_players]:
            # #     print(f"Player {player.name} stack: {player.stack}")
            # ######################################################

            state = new_state

        # return env.players[0].winnings
        for player in env.all_players[: env.n_players]:
            print(f"Player {player.name} stack: {player.stack}")
        return env.hand_history


def gumbel_softmax(logits, temperature=1.0, hard=False, value=0):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, num_classes] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
        Sampled tensor of shape [batch_size, num_classes] from the Gumbel-Softmax distribution.
    """
    # gumbel_noise = torch.distributions.Gumbel(0, 1).sample(logits.shape)
    if np.exp(-value) > 0:
        gumbel_noise = torch.distributions.Gumbel(
            0,
            value,
        ).sample(logits.shape)
    else:
        gumbel_noise = torch.zeros_like(logits)
    logits = torch.from_numpy(logits)
    y = logits + gumbel_noise.to(logits.device)
    y = F.softmax(y / temperature, dim=-1)
    if hard:
        y_hard = torch.argmax(y, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, y_hard, 1.0)
        y = (y_hard - y).detach() + y
    return y.detach().numpy()
