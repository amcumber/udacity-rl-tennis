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
from typing import Tuple, Type

import numpy as np
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .agents import Agent
from .ddpg_model import DDPGActor, DDPGCritic
from .noise_models import Noise
from .replay_buffers import ReplayBuffer


class DDPGAgent(Agent):
    """
    DDPG Multi agent implementation - Recieves memories from trainer
    Interacts with and learns from the environment.
    """
    def __init__(
        self,
        state_size: int,
        action_size: int,
        memory: ReplayBuffer,
        noise: Noise,
        gamma: float,
        tau: float,
        lr_actor: float,
        lr_critic: float,
        learn_f: int,
        weight_decay: float,
        device: str = "cpu",
        random_seed: int = 42,
        actor: Type[nn.Module] = DDPGActor,
        actor_hidden: Tuple[int] = (256, 128),
        actor_act: callable = F.relu,
        critic: Type[nn.Module] = DDPGCritic,
        critic_hidden: Tuple[int] = (256, 128),
        critic_act: callable = F.leaky_relu,
        upper_bound: int = 1,
        add_noise: bool = True,
        batch_norm: bool = True,
    ):
        """Initialize a DDPG Agent object.

        Parameters
        ----------
        state_size : int
            dimension of each state
        action_size : int
            dimension of each action
        gamma : float
            discount factor
        tau : float
            for soft update of target parameters
        lr_actor : float (1e-4)
            learning rate of the actor
        lr_critic : float (3e-4)
            learning rate of the critic
        learn_f : int (2)
            update local networks at specified iteration
        weight_decay : float (0.0001)
            L2 weight decay

        Optional Parameters
        -------------------
        random_seed : int (42)
            random seed
        actor : torch.nn.Module (DDPGActor)
            Actor Network to use for DDPG
        actor_hidden : tuple[int, ...] (256,128)
            Actor hidden architecture
        critic : torch.nn.Module (DDPGCritic)
            Critic Network to use for DDPG
        critic_hidden : tuple[int, ...] (256,128)
            Actor hidden architecture
        noise : Noise (OUActionNoise)
            Noise Model to use for normalizing the A/C Network
        upper_bound : int (1)
            bounding box for action upper_bound is set to value and lower_bound
            is set to value
        add_noise : bool (True)
            add noise to system
        """

        self.state_size = state_size
        self.action_size = action_size
        self.memory = memory
        self.noise = noise
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.learn_f = learn_f
        self.weight_decay = weight_decay
        self.seed = random.seed(random_seed)
        self.device = torch.device(device)
        self.upper_bound = upper_bound
        self.batch_norm = batch_norm

        # Noise process
        self.add_noise = add_noise

        # init Step Counter
        self.i_step = 0

        # Actor Network (w/ Target Network)
        self.actor_local = actor(
            state_size,
            action_size,
            random_seed,
            hidden_units=actor_hidden,
            upper_bound=upper_bound,
            act_func=actor_act,
            batch_norm=batch_norm
        ).to(device)
        self.actor_target = actor(
            state_size,
            action_size,
            random_seed,
            hidden_units=actor_hidden,
            upper_bound=upper_bound,
            act_func=actor_act,
            batch_norm=batch_norm
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
            act_func=critic_act,
            batch_norm=batch_norm
        ).to(device)
        self.critic_target = critic(
            state_size,
            action_size,
            random_seed,
            hidden_units=critic_hidden,
            act_func=critic_act,
            batch_norm=batch_norm
        ).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=lr_critic,
            weight_decay=weight_decay,
        )

        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

    def reset(self):
        if self.add_noise:
            self.noise.reset()

    def act(self, state, add_noise=None, noise_dampening=1.0):
        """Returns actions for given state as per current policy."""
        if add_noise is None:
            add_noise = self.add_noise
        state = torch.from_numpy(np.expand_dims(state, 0)).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * noise_dampening
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
        if (len(self.memory) > self.memory.batch_size):
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences: Tuple[torch.tensor, ...]):
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

        CITATION: the alogorithm for implementing the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforcement Learning Course
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
            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
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
        actor_file = f"{root_filename}_actor.pth"
        critic_file = f"{root_filename}_critic.pth"
        torch.save(self.actor_local.state_dict(), actor_file)
        torch.save(self.critic_local.state_dict(), critic_file)

    def load(self, actor_file, critic_file) -> None:
        """Load Actor and Critic Files"""
        self.actor_local.load_state_dict(torch.load(actor_file))
        self.actor_target.load_state_dict(torch.load(actor_file))

        self.critic_local.load_state_dict(torch.load(critic_file))
        self.critic_target.load_state_dict(torch.load(critic_file))

    def copy_from(self, other: Agent) -> None:
        """Copy agent weights from other"""
        # CITATION: https://github.com/salvioli/deep-rl-tennis/blob/master/ddpg_agent.py

        self.actor_local.load_state_dict(other.actor_local.state_dict())
        self.actor_target.load_state_dict(other.actor_target.state_dict())

        self.critic_local.load_state_dict(other.critic_local.state_dict())
        self.critic_target.load_state_dict(other.critic_target.state_dict())

    def copy(self) -> Agent:
        other = type(self).__new__(self.__class__)
        other.__dict__.update(self.__dict__)

        other.copy_from(self)

        return other

    def save_hyperparameters(self, file: str) -> None:
        if file is None:
            raise ValueError("File must be specified")
        with Path(file).open("w") as fh:
            toml.dump(self.__dict__, fh)
