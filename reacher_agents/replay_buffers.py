## ~~! From Deep Q Network Exercise from Udacity's Deep Reinforement Learning Course
import random
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Buffer(ABC):
    @abstractmethod
    def add(self, state, action, reward, next_state, done) -> None:
        """
        Add parameters, state, action, reward, next_state, and done into replay
        queue as tuple
        """
        pass

    @abstractmethod
    def sample(self) -> Tuple["states", "actions", "rewards", "next_states", "dones"]:
        """
        Retrieve random tuple contining state, action, reward, next_state, and
        done information
        """
        pass


class ReplayBuffer(Buffer):
    """
    Fixed-size buffer to store experience tuples

    Parameters
    ----------
    action_size : int
        dimension of each action
    buffer_size : int
        maximum size of buffer
    batch_size  : int
        size of each training batch
    seed : int
        random seed
    device : str
        'cpu' | 'cuda:0'
    """

    def __init__(
        self,
        action_size: int,
        buffer_size: int,
        batch_size: int,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(
                np.vstack([e.state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack(
                    [e.done for e in experiences if e is not None]
                ).astype(np.uint8)
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
