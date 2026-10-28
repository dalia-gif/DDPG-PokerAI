"""Neural network architectures used by the DDPG poker agent.

The actor maps an observation to a continuous action representation. Its output
contains one score per poker action (including BET) and one scalar used to
parameterize the bet size. The critic receives an observation-action pair and
estimates its value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pokerenv.common import Action, PlayerAction, action_list


class DDPGActor(nn.Module):
    """Actor network that proposes a poker action and a bet-size parameter.

    Args:
        in_features: Number of features in the state/observation vector.
        out_features: Size of the actor output, including the final bet-size
            scalar. The action-score head therefore has ``out_features - 1``
            outputs.
        action_range: Scale applied to the bet-size output. Since the scalar
            head ends in a sigmoid, its output is in ``[0, action_range]``.
        device: Device on which to place the network (defaults to CPU).

    The shared trunk learns a state representation. Two separate heads use
    that representation: ``action_net`` produces action scores, while
    ``value_net`` produces the scalar associated with bet sizing.
    """

    def __init__(
        self, in_features: int, out_features: int, action_range: float, device=None
    ):
        super(DDPGActor, self).__init__()

        if device is None:
            device = "cpu"
        self.device = device
        self.action_range = action_range

        # Shared feature extractor: progressively compress the input state
        # into a representation used by both output heads.
        base = 128
        self.net = nn.Sequential(
            nn.Linear(in_features, base),
            nn.ReLU(),
            nn.Linear(base, base // 2),
            nn.ReLU(),
            nn.Linear(base // 2, base // 2),
            nn.ReLU(),
        ).to(self.device)

        # Action head: returns one bounded score for each available action.
        # The caller uses these scores to select an action; they are not
        # converted to probabilities inside this network.
        self.action_net = nn.Sequential(
            nn.Linear(base // 2, base // 4),
            nn.ReLU(),
            nn.Linear(base // 4, base // 8),
            nn.ReLU(),
            nn.Linear(base // 8, out_features - 1),
            nn.Sigmoid(),
        ).to(self.device)

        # Bet-size head: predicts a scalar in [0, 1], then scales it to the
        # configured action range.
        self.value_net = nn.Sequential(
            nn.Linear(base // 2, base // 4),
            nn.ReLU(),
            nn.Linear(base // 4, base // 8),
            nn.ReLU(),
            nn.Linear(base // 8, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map an observation to action scores and a bet-size scalar.

        Args:
            x: Observation tensor, or an array-like observation. Non-tensor
                inputs are converted to float tensors and given a batch
                dimension.

        Returns:
            Tensor whose final dimension contains the action scores followed
            by the scaled bet-size scalar. When BET is the highest-scoring
            action, the scalar is set to zero by the current policy logic.
        """
        # Accept array-like observations as well as tensors. The reshape
        # follows the input convention used by the environment/training code.
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.squeeze(-1).unsqueeze(0)

        # Compute a shared state representation, then predict each part of
        # the action representation with its dedicated output head.
        x = self.net(x)
        a = self.action_net(x)
        v = self.value_net(x) * self.action_range

        # Apply the existing policy rule: if BET is the highest-scoring
        # action, zero the bet-size scalar. The environment interaction code
        # interprets the action scores and scalar to construct a PlayerAction.
        index = torch.argmax(a, dim=-1) == action_list.index(PlayerAction.BET)
        v[index] = 0

        # The action representation expected by the agent is the action-score
        # vector concatenated with the scalar bet-size parameter.
        x = torch.cat([a, v], dim=-1)
        return x


class DDPGCritic(nn.Module):
    """Critic network that estimates values from state-action pairs.

    Args:
        in_features_obs: Number of features in the observation vector.
        in_features_act: Number of features in the action representation.
        device: Device on which to place the network (defaults to CPU).

    The observation and action vectors are concatenated and passed through a
    multilayer perceptron. The final layer currently emits two sigmoid-bounded
    outputs, matching the architecture in the original implementation.
    """

    def __init__(self, in_features_obs, in_features_act, device=None):
        super(DDPGCritic, self).__init__()

        if device is None:
            device = "cpu"
        self.device = device

        # The critic jointly processes the state and action so it can estimate
        # how valuable the proposed action is in the given state.
        self.obs_net = nn.Sequential(
            nn.Linear(in_features_obs + in_features_act, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        ).to(self.device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Estimate the value output for a batch of state-action pairs.

        Args:
            state: Observation tensor, or an array-like observation.
            action: Action-representation tensor, or an array-like action.

        Returns:
            The critic network output for the concatenated state-action
            representation.
        """
        # Convert non-tensor inputs to float tensors and add the batch
        # dimension expected by the network.
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            state = state.squeeze(-1).unsqueeze(0)

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
            action = action.unsqueeze(0)

        # Concatenate features along the final dimension, then estimate the
        # value representation with the critic MLP.
        return self.obs_net(torch.cat([state, action], dim=-1))
