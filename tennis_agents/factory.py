from os import stat
import random
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple, Type

import numpy as np
import toml

from .agents import Agent
from .environments import EnvironmentMgr
from .trainers import Trainer

class TrainerFactory:
    """
    Factory that can create an EnvironmentMgr, Agent, or Trainer Class
    
    When called will generate a Trainer instance populated with correct Agent 
    and EnvironmentMgr (see __call__).
    """
    def __call__(
        self,
        trainer_class: Type[Trainer],
        trainer_file: str,
        envh_class: Type[EnvironmentMgr],
        envh_param: str,
        agent_class: Type[Agent],
        agent_file: str,
        actor_file: str,
        critic_file: str,
    ) -> Trainer:
        """
        Generate a Trainer with populated parameters

        Parameters
        ----------
        trainer_class : Type[Trainer]
            Class of trainer to populate(MultiAgentTrainer, SingleAgentTrainer)
        agent_class : Type[Agent]
            Class of agnet (DDPGAgent)
        envh_class : Type[EnvironmentMgr]
            Class of EnvironmentMgr (UnityEnvMgr, GymContinuousEnvMgr)
        envh_param : str
            Environment handler input parameter (either filepath or scenario
            name)
        trainer_data : str
            File path to toml file containing trainer parameters
        agent_data : str
            File path to toml file containing agent parameters
        actor_data : str
            File path to Actor weight file (pth type)
        critic_data : str
            File path to Critic weight file (pth type)
        """
        agent = self.agent_factory(
            agent_class,
            agent_file,
            actor_file,
            critic_file,
        )
        envh = self.envh_factory(envh_class, envh_param)
        return self.trainer_factory(trainer_class, trainer_file, agent, envh)

    @staticmethod
    def envh_factory(
        envh_class: Type[EnvironmentMgr],
        envh_param: str,
    ):
        return envh_class(envh_class, envh_param)

    @classmethod
    def agent_factory(
        cls,
        agent_class: Type[Agent],
        agent_data: str = None,
        actor_file: str = None,
        critic_file: str = None,
    ):
        """Generate an Agent instance"""
        # NOTE: takes actor, critic, and noise as defult
        hyperparams = cls.read_toml(agent_data)

        agent = agent_class(**hyperparams)
        agent.load(actor_file, critic_file)
        return agent

    @classmethod
    def trainer_factory(
        cls,
        trainer_class: Type[Trainer],
        trainer_data: str,
        agent: Agent,
        envh: EnvironmentMgr,
    ):
        hyperparameters = cls.load_toml(trainer_data)
        return trainer_class(agent = agent, env=envh, **hyperparameters)

    @staticmethod
    def read_toml(file: str):
        if file is None:
            raise ValueError("Data File Must be Specified")
        with Path(file).open('r') as fh:
            return toml.load(fh, file)
