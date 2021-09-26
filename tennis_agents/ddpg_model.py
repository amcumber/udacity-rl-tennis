# CITATION: Udacity's Deep Reinforcement Learning Course - DDPG Bipedal Exercise
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class DDPGActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 42,
        hidden_units: Tuple[int] = (128, 64),
        upper_bound: float = 1.0,
        act_func: callable = F.relu,
        batch_norm: bool = False
    ):
        """Initialize parameters and build model.
        Parameters
        ----------
        state_size : int
            Dimension of each state
        action_size : int
            Dimension of each action
        seed : int
            Random seed
        hidden_units : tuple[int]
            Number of nodes in first, second, and third hidden layer
        upper_bound : float
            upper bound of action space to clip to - equal and opposite to
            lower bound
        act_func : callable
            activation function to use between FC layers
        batch_norm : bool
            enable batch normalization between FC layers

        CITATION: the algorithm for implementing the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforcement Learning Course
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.batch_norm = batch_norm
        self.upper_bound = upper_bound
        self.act_func = act_func
        self.n_layers = 3

        self.fc1 = nn.Linear(state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], action_size)

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(state_size)
            self.bn2 = nn.BatchNorm1d(hidden_units[0])
            self.bn3 = nn.BatchNorm1d(hidden_units[1])

        self.reset_parameters()


    def reset_parameters(self):
        for i in range(self.n_layers):
            fc = getattr(self, f'fc{i+1}')
            if i == self.n_layers-1:
                fc.weight.data.uniform_(-3e-3, 3e-3)
            else:
                fc.weight.data.uniform_(*hidden_init(fc))

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        if self.batch_norm:
            x = self.bn1(x)
            x = self.fc1(x)
            x = self.act_func(x)
            x = self.bn2(x)
            x = self.fc2(x)
            x = self.act_func(x)
            x = self.bn3(x)
            x = self.fc3(x)
            x = F.tanh(x)
            return x
        x = self.act_func(self.fc1(x))
        x = self.act_func(self.fc2(x))
        x = torch.tanh(self.fc3(x))


class DDPGCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        hidden_units: tuple = (128, 64),
        act_func: callable = F.leaky_relu,
        batch_norm : bool = False,
    ):
        """Initialize parameters and build model.
        Params
        ======
        state_size : int
            Dimension of each state
        action_size : int
            Dimension of each action
        seed : int
            Random seed
        hidden_units : Tuple[int]
            Number of nodes in the first, second, and third hidden layers
        batch_norm : bool
            enable batch normalization between FC layers

        CITATION: the algorithm for implementing the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforcement Learning Course
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.batch_norm = batch_norm
        self.act_func = act_func
        self.n_layers = 3

        self.fc1 = nn.Linear(state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + action_size,
                             hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(state_size)
            self.bn2 = nn.BatchNorm1d(hidden_units[0] + action_size)
            self.bn3 = nn.BatchNorm1d(hidden_units[1])

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_layers):
            fc = getattr(self, f'fc{i+1}')
            if i == self.n_layers-1:
                fc.weight.data.uniform_(-3e-3, 3e-3)
            else:
                fc.weight.data.uniform_(*hidden_init(fc))

    def forward(self, state, action):
        """
        Build a critic (value) network that maps (state, action)
        pairs -> Q-values.
        """
        x = state
        if self.batch_norm:
            x = self.bn1(x)
            x = self.fc1(x)
            x = self.act_func(x)
            x = torch.cat((x, action), dim=1)
            x = self.bn2(x)
            x = self.fc2(x)
            x = self.act_func(x)
            x = self.bn3(x)
            x = self.fc3(x)
            return x
        x = self.act_func(self.fc1(x))
        x = torch.cat((x, action), dim=1)
        x = self.act_func(self.fc2(x))
        x = self.fc3(x)
        return x