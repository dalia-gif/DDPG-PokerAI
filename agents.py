"""DDPG agent and action-selection utilities for the poker environment.

The agent learns a continuous action representation with an Actor-Critic setup.
The Actor outputs scores for the available action types and a scalar used to
select a bet size. The Critic estimates the value of a state-action pair.
"""

from copy import deepcopy

import numpy as np
import pokerenv.obs_indices as indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pokerenv.common import Action, PlayerAction, action_list
from tqdm import tqdm

from .model import DDPGActor, DDPGCritic
from .replaybuffer import ReplayBuffer


SEED = 42


class DDPG:
    """Deep Deterministic Policy Gradient agent for poker.

    Args:
        buffer: Replay buffer containing state-action-reward transitions.
        actor: Online policy network that proposes actions.
        critic: Online network that estimates state-action values.
        actor_lr: Learning rate for the Actor optimizer.
        critic_lr: Learning rate for the Critic optimizer.
        solve_criteria: Moving-average reward threshold used to stop training.
        episode_limit: Maximum number of training episodes.
        filter_invalid_actions: If True, mask actions marked invalid by the
            environment before selecting an action.
        device: PyTorch device used for model computations (defaults to CPU).
    """

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

        # Target networks provide slowly changing targets during training.
        # They start with the same parameters as the online networks.
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        self.solve_criteria = solve_criteria
        self.episode_limit = episode_limit
        self.filter_invalid_actions = filter_invalid_actions

    @staticmethod
    def soft_target_update(net: nn.Module, target_net: nn.Module, tau: float):
        """Move target-network parameters a small step toward online parameters.

        A small ``tau`` makes the target networks change more gradually than
        the online networks, which helps stabilize the DDPG learning targets.
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self):
        """Update the Actor, Critic, and their target networks from replay data."""
        # Training hyperparameters for the DDPG update.
        BATCH_SIZE = 4096
        DISCOUNT_FACTOR = 0.99
        TAU = 0.005

        # Sample transitions and convert each field into a batch tensor.
        # A transition is (state, action, reward, next_state, done).
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

        # Bellman target: r + gamma * Q_target(s', pi_target(s')).
        # Do not track gradients through the target calculation. For terminal
        # transitions, (1 - done) removes the bootstrapped future value.
        with torch.no_grad():
            next_actions = self.target_actor(next_states.to(self.device))
            target_q_values = rewards + DISCOUNT_FACTOR * (
                1 - dones
            ) * self.target_critic(next_states, next_actions)

        # Train the Critic to regress toward the Bellman target (TD learning).
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train the Actor to maximize the Critic's estimated value of its actions.
        # The negative sign converts this maximization objective into a loss.
        actor_loss = -self.critic(
            states.to(self.device), self.actor(states.to(self.device))
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Slowly synchronize target networks with their online counterparts.
        self.soft_target_update(self.actor, self.target_actor, TAU)
        self.soft_target_update(self.critic, self.target_critic, TAU)

    def interaction_step(self, state, env, noise=True, global_step=10000):
        """Choose an action, apply it to the environment, and return the transition.

        During training, exploration noise is added to the Actor's outputs.
        During evaluation, the action scores are converted to probabilities
        without adding exploration noise.

        Returns:
            A tuple containing the raw Actor output, the acting player's reward,
            the next observation, the terminal flag, and player_1's reward.
        """
        action = self.actor(torch.from_numpy(state).to(self.device))
        action_raw = action.cpu().detach().numpy().flatten()

        # The Actor output is split into action-type scores and one normalized
        # scalar that controls the bet size within the environment's valid range.
        a = action_raw[:-1]
        v = action_raw[-1]

        if noise:
            # Anneal exploration as training progresses by reducing the scale
            # derived from global_step. Gumbel noise is used for action scores.
            a = gumbel_softmax(a, value=np.exp(-global_step / 100))
            noise = np.random.normal(0, np.exp(-global_step / 100), size=v.shape)
            v = np.clip(v + noise, 0, 1)
        else:
            # Convert action scores into a probability distribution at evaluation.
            a = F.softmax(torch.from_numpy(a), dim=-1).numpy()

        state = torch.from_numpy(state)
        invalid_actions = torch.argwhere(state[indices.VALID_ACTIONS] != 1).flatten()

        # Mask actions that the environment says are invalid so they cannot win
        # the argmax selection when invalid-action filtering is enabled.
        if self.filter_invalid_actions:
            a[invalid_actions] = 0

        valid_bet_low = state[indices.VALID_BET_LOW].flatten().cpu().numpy()
        valid_bet_high = state[indices.VALID_BET_HIGH].flatten().cpu().numpy()

        chosen_action = PlayerAction(action_list[a.argmax()])
        # Map the Actor's normalized scalar v in [0, 1] to the legal bet interval.
        bet_size = v * (valid_bet_high - valid_bet_low) + valid_bet_low

        action = Action(chosen_action, bet_size)
        new_state, reward, done, _ = env.step(action)

        # The environment returns rewards by player index. Keep the reward for
        # player_1 separately for episode-level reporting, while returning the
        # reward belonging to the player who acted in this step for DDPG updates.
        acting_player = int(state[indices.ACTING_PLAYER])
        for i, player in enumerate(env.all_players):
            if player.name == "player_1":
                break
        player1_rewards = reward[i]
        if player1_rewards is None:
            player1_rewards = 0
        reward = float(reward[acting_player])

        return action_raw, reward, new_state, done, player1_rewards

    def train(self, env):
        """Train the agent for up to ``episode_limit`` episodes.

        The reward history used for the progress display is based on player_1's
        reward accumulated during each episode. Training stops when the mean of
        the most recent 100 episode rewards reaches ``solve_criteria``.

        Returns:
            List of episode-level rewards collected during training.
        """
        total_rewards = []
        pbar = tqdm(range(1, self.episode_limit + 1))

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

            # Record the episode's accumulated player_1 reward for monitoring.
            if len(self.rewards) != 0:
                total_reward = sum(self.rewards)
                total_rewards.append(total_reward)

            avg_rewards = np.mean(total_rewards[-100:])
            pbar.set_description(
                f"reward avg: {avg_rewards:.2f} min: {min(total_rewards):.2f} max: {max(total_rewards):.2f}"
            )
            if avg_rewards >= self.solve_criteria:
                print(f"\nSOLVED in {episode} episodes!")
                break

        return total_rewards

    def play(self, env):
        """Run one episode without exploration noise and return the hand history.

        The method prints each player's starting and final stack, then lets the
        trained policy play until the environment signals that the hand is over.
        """
        state, done = env.reset(), False

        for player in env.all_players[: env.n_players]:
            print(f"Player {player.name} stack: {player.stack}")

        while not done:
            _, _, new_state, done, _ = self.interaction_step(
                state, env, noise=False
            )
            state = new_state

        for player in env.all_players[: env.n_players]:
            print(f"Player {player.name} stack: {player.stack}")
        return env.hand_history


def gumbel_softmax(logits, temperature=1.0, hard=False, value=0):
    """Apply a Gumbel-Softmax transformation to action logits.

    Args:
        logits: NumPy array of unnormalized action scores.
        temperature: Softmax temperature; lower values produce sharper outputs.
        hard: If True, return a one-hot forward value while retaining the
            straight-through gradient construction used by Gumbel-Softmax.
        value: Scale parameter passed to the Gumbel distribution in this
            implementation.

    Returns:
        NumPy array containing the transformed action scores.
    """
    # Add sampled Gumbel noise before applying softmax. The noise scale is
    # controlled by ``value`` by the existing exploration schedule.
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
        # Straight-through estimator: use a one-hot result in the forward pass
        # while preserving the soft sample's gradient path.
        y_hard = torch.argmax(y, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, y_hard, 1.0)
        y = (y_hard - y).detach() + y

    return y.detach().numpy()
