import torch
import torch.nn as nn
import torch.nn.functional as F
from pokerenv.common import Action, PlayerAction, action_list


class DDPGActor(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, action_range: float, device=None
    ):
        # in_features: the size of state representation
        # out_features: the size of action representation
        # action_range: e.g., action \in [-2, 2] if action_range is 2.0
        super(DDPGActor, self).__init__()
        if device is None:
            device = "cpu"
        self.device = device
        self.action_range = action_range
        base = 128
        self.net = nn.Sequential(
            nn.Linear(in_features, base),
            nn.ReLU(),
            nn.Linear(base, base // 2),
            nn.ReLU(),
            nn.Linear(base // 2, base // 2),
            nn.ReLU(),
        ).to(self.device)
        self.action_net = nn.Sequential(
            nn.Linear(base // 2, base // 4),
            nn.ReLU(),
            nn.Linear(base // 4, base // 8),
            nn.ReLU(),
            nn.Linear(base // 8, out_features - 1),
            # nn.Softmax(dim=-1),
            nn.Sigmoid(),
        ).to(self.device)
        self.value_net = nn.Sequential(
            nn.Linear(base // 2, base // 4),
            nn.ReLU(),
            nn.Linear(base // 4, base // 8),
            nn.ReLU(),
            nn.Linear(base // 8, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.squeeze(-1).unsqueeze(0)
        x = self.net(x)
        # a, v = x[:, :-1], x[:, -1:]
        a = self.action_net(x)
        v = self.value_net(x) * self.action_range

        index = torch.argmax(a, dim=-1) == action_list.index(PlayerAction.BET)

        v[index] = 0

        x = torch.cat([a, v], dim=-1)

        return x


class DDPGCritic(nn.Module):
    def __init__(self, in_features_obs, in_features_act, device=None):
        # in_features_obs: the size of state representation
        # in_features_act: the size of action representation
        super(DDPGCritic, self).__init__()
        if device is None:
            device = "cpu"
        self.device = device
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
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            state = state.squeeze(-1).unsqueeze(0)

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
            action = action.unsqueeze(0)

        return self.obs_net(torch.cat([state, action], dim=-1))


