## ~~! From Deep Q Network Exercise from Udacity's Deep Reinforement Learning
#      Course
## DDPG CITATION: T Lillicrap, et al. Continuous Control with Deep Reinforement
#                 Learning. arXiv, 5 Jul 20019, 1509.02971v6
#                 (https://arxiv.org/pdf/1509.02971.pdf)
## CITATION: From DDPG Bipedal exersize from Udacity's Deep Reinforement
#            Learning Course
import copy
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .agents import Agent, MultiAgent
from .ddpg_model import DDPGActor, DDPGCritic
from .noise_model import Noise, OUNoise
from .replay_buffers import ReplayBuffer


class DDPGAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        buffer_size: int,
        batch_size: int,
        gamma: float,
        tau: float,
        lr_actor: float,
        lr_critic: float,
        learn_f: int,
        weight_decay: float,
        device: str = "cpu",
        random_seed: int = 42,
        actor: nn.Module = DDPGActor,
        actor_hidden: Tuple[int] = (256, 128),
        critic: nn.Module = DDPGCritic,
        critic_hidden: Tuple[int] = (256, 128),
        noise: Noise = OUNoise,
        upper_bound: int = 1,
        add_noise = True,
    ):
        """Initialize an Agent object.

        Parameters
        ----------
        state_size : int
            dimension of each state
        action_size : int
            dimension of each action
        random_seed : int
            random seed

        Optional Parameters
        -------------------
        buffer_size : int (int(1e6))
            replay buffer size
        batch_size : int (128)
            minibatch size
        gamma : float (0.99)
            discount factor
        tau : float (1e-3)
            for soft update of target parameters
        lr_actor : float (1e-4)
            learning rate of the actor
        lr_critic : float (3e-4)
            learning rate of the critic
        target_update_f : int (10)
            update target networks at specified iteration
        learn_f : int (2)
            update local networks at specified iteration
        weight_decay : float (0.0001)
            L2 weight decay
        actor : torch.nn.Module (DDPGActor)
            Actor Network to use for DDPG
        critic : torch.nn.Module (DDPGCritic)
            Critic Network to use for DDPG
        noise : Noise (OUNoise)
            Noise Model to use for normalizing the A/C Network
        upper_bound : int (1)
            bounding box for action upper_bound is set to value and lower_bound
            is set to value
        """

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.learn_f = learn_f

        self.weight_decay = weight_decay

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.device = torch.device(device)

        self.upper_bound = upper_bound

        # Actor Network (w/ Target Network)
        self.actor_local = actor(
            state_size,
            action_size,
            random_seed,
            hidden_units=actor_hidden,
            upper_bound=upper_bound,
        ).to(device)
        self.actor_target = actor(
            state_size,
            action_size,
            random_seed,
            hidden_units=actor_hidden,
            upper_bound=upper_bound,
            act_func=F.relu,
        ).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(),
            lr=lr_actor,
        )

        # Critic Network (w/ Target Network)
        self.critic_local = critic(
            state_size,
            action_size,
            random_seed,
            hidden_units=critic_hidden,
        ).to(device)
        self.critic_target = critic(
            state_size,
            action_size,
            random_seed,
            hidden_units=critic_hidden,
            act_func=F.leaky_relu,
        ).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=lr_critic,
            weight_decay=weight_decay,
        )

        # Noise process
        self.noise = noise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(
            action_size,
            buffer_size,
            batch_size,
            random_seed,
            device=device,
        )

        # init Step Counter
        self.i_step = 0
        self.add_noise = add_noise

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=None):
        """Returns actions for given state as per current policy."""
        if add_noise is None:
            add_noise = self.add_noise
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -self.upper_bound, self.upper_bound)

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to
        learn.
        """
        self.i_step += 1
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if (len(self.memory) > self.batch_size):
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences: Tuple[torch.tensor]):
        """
        Update policy and value parameters using given batch of experience
        tuples.

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters
        ----------
        experiences : Tuple[torch.Tensor]
            tuple of (s, a, r, s', done) tuples

        CITATION: the alogrithm for implemeting the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforement Learning Course
        """
        states, actions, rewards, next_states, dones = experiences

        # ------------------------ update critic ------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.forward(next_states)
        q_targets_next = self.critic_target.forward(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local.forward(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ------------------------ update actor -------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local.forward(states)
        actor_loss = -self.critic_local.forward(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- update target networks --------------------- #
        if (self.i_step % self.learn_f == 0):
            self._soft_update(self.critic_local, self.critic_target)
            self._soft_update(self.actor_local, self.actor_target)

    def _soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                (self.tau * local_param.data) + (1.0 - self.tau) * target_param.data
            )

    def save(self, root_filename: str = "checkpoint") -> None:
        """
        Save Actor|Critic Models

        Parameters
        ----------
        root_filename : str
            root of file to save. For example, "checkpoint" will save the actor
            file as "checkpoint_actor.pth" and the critic file as
            "checkpoint_critic.pth
        """
        actor_file = root_filename + "_actor.pth"
        critic_file = root_filename + "_critic.pth"
        torch.save(self.actor_local.state_dict(), actor_file)
        torch.save(self.critic_local.state_dict(), critic_file)

    def load(self, actor_file, critic_file) -> None:
        """Load Actor and Critic Files"""
        self.actor_local.load_state_dict(torch.load(actor_file))
        self.actor_target.load_state_dict(torch.load(actor_file))

        self.critic_local.load_state_dict(torch.load(critic_file))
        self.critic_target.load_state_dict(torch.load(critic_file))

    def save_hyperparameters(self, file: str) -> None:
        if file is None:
            raise ValueError("File must be specified")
        data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'learn_f': self.learn_f,
            'weight_decay': self.weight_decay,
            'device': self.device,
            'random_seed': self.random_seed,
            'upper_bound': self.upper_bound,
        }
        with Path(file).open("w") as fh:
            toml.dump(data, fh)
