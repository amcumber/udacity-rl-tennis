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


class ActionNoise(ABC):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def sample(self):
        ...


class ParamNoise(ABC):
    @abstractmethod
    def adapt(self):
        ...

    @abstractmethod
    def get_stats(self):
        ...

    @abstractmethod
    def reset(self):
        ...


class OUActionNoise(ActionNoise):
    """Ornstein-Uhlenbeck process applied to action space"""

    def __init__(
        self,
        action_size: int,
        seed: int,
        mu: float = 0.0,
        sigma: float = 0.05,
        theta: float = 0.25,
    ):
        """Initialize parameters and noise process."""
        self.mu = mu * np.zeros(action_size)
        self.dt = action_size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.x_prev = np.zeros_like(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        # CITATION corrected by : https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
        x = self.x_prev
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * np.random.normal(size=self.mu.shape)
        self.x_prev = x + dx
        return self.x_prev

    def __call__(self):
        return self.sample()

class OUActionNoiseV2(ActionNoise):
    """
    Ornstein-Uhlenbeck process applied to action space
    CITATION : https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
    """

    def __init__(
        self,
        action_size: int,
        seed: int,
        mu: float = 0.0,
        sigma: float = 0.05,
        theta: float = 0.25,
    ):
        """Initialize parameters and noise process."""
        self.mu = mu * np.zeros(action_size)
        self.dt = action_size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.x_prev = np.zeros_like(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        # CITATION corrected by : https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
        x = self.x_prev
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * np.random.normal(size=self.mu.shape)
        self.x_prev = x + dx
        return self.x_prev

    def __call__(self):
        return self.sample()


# CITATION: from openAI baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# Related article:
# https://openai.com/blog/better-exploration-with-parameter-noise/
# and related Paper: https://arxiv.org/abs/1706.01905
# Additional help from disccusion found here:
# https://pythonrepo.com/repo/ikostrikov-pytorch-naf-python-deep-learning
# and here
# https://github.com/l5shi/Multi-DDPG-with-parameter-noise/blob/master/Multi_DDPG_with_parameter_noise.ipynb
class AdaptiveParameterNoise:
    """
    Adaptive Parameter Noise Class from OpenAI baselines

    Parameters
    ----------
    initial_std : float (0.1)
        Initial standard deviation parameter
    desired_action_std: float (0.1)
        Desired Action standard deviation parameter
    adoption_coef: float (1.01)
        Adoption Coeffcient, used to adjust the current std-deviation - must be
        > 1.00
    """

    def __init__(
        self,
        initial_std: float = 0.1,
        desired_action_std: float = 0.1,
        adoption_coef: float = 1.01,
    ):
        self.initial_std = initial_std
        self.desired_action_std = desired_action_std
        self.adoption_coef = adoption_coef

        self.reset()

        if self.adoption_coef < 1.0:
            msg = (
                f"adoption_coef must be >1.0, adoption_coef set to: "
                + f"{self.adoption_coef}"
            )
            raise ValueError(msg)

    def adapt(self, distance):
        """Adapt noise given distance"""
        if distance > self.desired_action_std:
            # Decrease
            self.current_std /= self.adoption_coef
        else:
            # Increase
            self.current_std *= self.adoption_coef

    def get_stats(self):
        """Get statistics from parameter noise standard deviation"""
        stats = {"param_noise_stddev": self.current_std}
        return stats

    def reset(self):
        self.current_std = self.initial_std
