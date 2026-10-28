import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """
    Store and sample transitions for DDPG training.

    The buffer can combine transitions collected by the agent with
    preprocessed human poker experience loaded from a dataset.
    The `ratio` parameter controls the proportion of samples drawn
    from the agent's buffer when sampling a batch.
    """

    def __init__(self, capacity: int, ratio: float = 1.0, replacable=False):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of agent transitions to store.
            ratio (float): Proportion of each batch sampled from the agent's
                buffer. The remaining samples may come from human experience.
            replacable (bool): If True, load the dataset into the buffer and
                use the buffer for both parts of the sampled batch.
        """
        self.capacity = capacity
        self.replacable = replacable
        self.ratio = ratio

        if self.ratio != 1:
            if self.replacable:
                # Initialize the buffer with preprocessed poker game data.
                self.buffer = torch.load("datasets/parsed_poker_games.pt")
                self.human_experience = torch.load(
                    "datasets/parsed_poker_games.pt",
                    weights_only=False,
                )
            else:
                # Store agent-generated transitions separately from human data.
                self.buffer = []
                self.human_experience = torch.load(
                    "datasets/parsed_poker_games.pt",
                    weights_only=False,
                )
        else:
            # Use only agent-generated transitions when ratio equals 1.
            self.buffer = []

    def push(self, transition):
        """
        Add a transition to the buffer.

        If the buffer has reached its capacity, remove the oldest transition
        before appending the new one.

        Args:
            transition: The transition to add.
        """
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions.

        When `ratio` is 1, all samples are drawn from the agent's buffer.
        Otherwise, the batch combines samples from the agent's buffer and,
        when available, the human experience dataset.

        Sampling is performed without replacement when enough entries are
        available, and with replacement otherwise.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            list: A list of sampled transitions.
        """
        # Determine how many samples to draw from the agent's buffer.
        target_size = int(batch_size * self.ratio)
        idxs_buffer = np.random.choice(
            len(self.buffer),
            size=target_size,
            replace=True if len(self.buffer) < target_size else False,
        )

        if self.ratio == 1:
            return [self.buffer[idx] for idx in idxs_buffer]

        if self.replacable:
            # Draw the remaining samples from the same buffer.
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

        # If no human experience is available, return the agent samples only.
        if len(self.human_experience) == 0:
            return [self.buffer[idx] for idx in idxs_buffer]

        # Sample the remaining transitions from the human experience dataset.
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
        """
        Return the combined number of stored transitions.

        Returns:
            int: Number of entries in the agent buffer and human dataset.
        """
        return len(self.buffer + self.human_experience)

    def load_from_memory(self, file):
        """
        Load human experience from a serialized file.

        Args:
            file (str): Path to the file containing saved transitions.
        """
        self.human_experience = torch.load(file)