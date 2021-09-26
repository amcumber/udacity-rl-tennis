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
from dataclasses import dataclass

import numpy as np


class Noise(ABC):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def sample(self):
        ...


class OUActionNoise(Noise):
    """Ornstein-Uhlenbeck process applied to action space"""

    def __init__(
        self,
        action_size: int,
        seed: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(action_size)
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


# CITATION: from openAI baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# Related article:
# https://openai.com/blog/better-exploration-with-parameter-noise/
# and related Paper: https://arxiv.org/abs/1706.01905
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
    def __init__(self,
        initial_std: float = 0.1,
        desired_action_std: float = 0.1,
        adoption_coef: float = 1.01,
    ):
        self.initial_std = initial_std
        self.desired_action_std = desired_action_std
        self.adoption_coef = adoption_coef


    def __post_init__(self):
        self.reset()

        if self.adoption_coef < 1.0:
            msg = (f"adoption_coef must be >1.0, adoption_coef set to: " + 
                   f"{self.adoption_coef}")
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
        stats = {
            'param_noise_stddev': self.current_std
        }
        return stats

    def reset(self):
        self.current_std = self.initial_std

# def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
#     # CITATION: https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg_learner.py
#     assert len(actor.vars) == len(perturbed_actor.vars)
#     assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

#     updates = []
#     for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
#         if var in actor.perturbable_vars:
#             logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
#             updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
#         else:
#             logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
#             updates.append(tf.assign(perturbed_var, var))
#     assert len(updates) == len(actor.vars)
#     return tf.group(*updates)


