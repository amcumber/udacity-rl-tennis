## ~~! From Deep Q Network Exercise from Udacity's Deep Reinforement Learning
#      Course
## DDPG CITATION: T Lillicrap, et al. Continuous Control with Deep Reinforement
#                 Learning. arXiv, 5 Jul 20019, 1509.02971v6
#                 (https://arxiv.org/pdf/1509.02971.pdf)
## CITATION: From DDPG Bipedal exersize from Udacity's Deep Reinforement
#            Learning Course
from abc import ABC, abstractmethod
import copy
import random
from typing import Tuple

import numpy as np


class Noise(ABC):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def sample(self):
        ...


class OUNoise(Noise):
    """Ornstein-Uhlenbeck process."""

    def __init__(
        self,
        size: int,
        seed: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for i in range(len(x))]
        )
        self.state = x + dx
        return self.state

    def __call__(self):
        return self.sample()
