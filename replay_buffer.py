import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity: int, ratio: float = 1.0, replacable=False):
        self.capacity = capacity
        self.replacable = replacable
        self.ratio = ratio
        if self.ratio != 1:
            if self.replacable:
                self.buffer = torch.load("datasets/parsed_poker_games.pt")
                self.human_experience = torch.load(
                    "datasets/parsed_poker_games.pt",
                    weights_only=False,
                )
            else:
                self.buffer = []
                self.human_experience = torch.load(
                    "datasets/parsed_poker_games.pt",
                    weights_only=False,
                )
        else:
            self.buffer = []

    def push(self, transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        target_size = int(batch_size * self.ratio)
        idxs_buffer = np.random.choice(
            len(self.buffer),
            size=target_size,
            replace=True if len(self.buffer) < target_size else False,
        )

        if self.ratio == 1:
            return [self.buffer[idx] for idx in idxs_buffer]
        if self.replacable:
            idxs_human = np.random.choice(
                len(self.buffer),
                size=batch_size - target_size,
                replace=(
                    True if len(self.buffer) < (batch_size - target_size) else False
                ),
            )
            return [self.buffer[idx] for idx in idxs_buffer] + [
                self.buffer[idx] for idx in idxs_human
            ]

        if len(self.human_experience) == 0:
            return [self.buffer[idx] for idx in idxs_buffer]

        idxs_human = np.random.choice(
            len(self.human_experience),
            size=batch_size - target_size,
            replace=(
                True
                if len(self.human_experience) < (batch_size - target_size)
                else False
            ),
        )
        return [self.buffer[idx] for idx in idxs_buffer] + [
            self.human_experience[idx] for idx in idxs_human
        ]

    def __len__(self):
        return len(self.buffer + self.human_experience)

    def load_from_memory(self, file):
        self.human_experience = torch.load(file)
