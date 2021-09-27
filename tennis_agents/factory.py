"""Factories for core classes"""
from abc import ABC, abstractmethod
from pathlib import Path
from tennis_agents.environments import EnvironmentMgr
from typing import Any, Dict, Tuple
import logging

import toml
import torch
import torch.nn.functional as F

from .ddpg_agent import DDPGAgent
from .maddpg import MADDPGAgent
from .trainers import TennisTrainer, Trainer
from .unity_environments import UnityEnvMgr
from .replay_buffers import ReplayBuffer
from .noise_models import AdaptiveParameterNoise, OUActionNoise


# logger = logging.getLogger("TennisTrainer")

class TrainerFactory(ABC):
    @staticmethod
    def read_toml(file: str) -> Dict[str, Any]:
        assert Path(file).exists()
        with Path(file).open('r') as fh:
            data = toml.load(fh)
        return data

    @abstractmethod
    def get_trainer(*args, **kwargs):
        """Returns a trainer object complete with environment manager and agents"""


class TennisFactory(TrainerFactory):
    @classmethod
    def get_environment_manager(cls, data: Dict[str, Any]) -> EnvironmentMgr:
        ENV_FILE: str = data['environment']['ENV_FILE']
        return UnityEnvMgr(ENV_FILE)

    @classmethod
    def get_agent(cls, data: Dict[str, Any]) -> MADDPGAgent:
        # Unpack config
        # # Environment
        # ENV_FILE: str                  = data['environment']['ENV_FILE']
        STATE_SIZE: int                = data['environment']['STATE_SIZE']
        ACTION_SIZE: int               = data['environment']['ACTION_SIZE']
        UPPER_BOUND: float             = data['environment']['UPPER_BOUND']
        # SOLVED: float                  = data['environment']['SOLVED']
        # ROOT_NAME: str                 = data['environment']['ROOT_NAME']
        SEED: int                      = data['environment']['SEED']

        # # Memory
        BATCH_SIZE: int                = data['memory']['BATCH_SIZE']
        BUFFER_SIZE: int               = data['memory']['BUFFER_SIZE']

        # # Agent
        N_AGENTS: int                  = data['agent']['N_AGENTS']
        LEARN_F: int                   = data['agent']['LEARN_F']
        GAMMA: float                   = data['agent']['GAMMA']
        TAU: float                     = data['agent']['TAU']
        LR_ACTOR: float                = data['agent']['LR_ACTOR']
        LR_CRITIC: float               = data['agent']['LR_CRITIC']
        WEIGHT_DECAY: float            = data['agent']['WEIGHT_DECAY']
        ACTOR_HIDDEN: Tuple[int, int]  = data['agent']['ACTOR_HIDDEN']
        CRITIC_HIDDEN: Tuple[int, int] = data['agent']['CRITIC_HIDDEN']
        ACTOR_ACT: str                 = data['agent']['ACTOR_ACT']
        CRITIC_ACT: str                = data['agent']['CRITIC_ACT']
        ADD_NOISE: Tuple[bool, bool]   = data['agent']['ADD_NOISE']
        BATCH_NORM: float              = data['agent']['BATCH_NORM']
        NOISE_DECAY: float             = data['agent']['NOISE_DECAY']

        # # Noise
        NOISE_TYPE: str                = data['noise'].pop('TYPE').upper()

        action_noise = None
        param_noise = None
        if NOISE_TYPE == 'OU' or NOISE_TYPE == 'BOTH':
            action_noise_data = {
                key.lower(): val for key, val in data['OU'].items()
            }
            action_noise_data['action_size'] = ACTION_SIZE
            action_noise_data['seed'] = SEED
            action_noise = OUActionNoise(**action_noise_data)
        if NOISE_TYPE == 'AP' or NOISE_TYPE == 'BOTH':
            param_noise_data = {
                key.lower(): val for key, val in data['AP'].items()
            }
            param_noise = AdaptiveParameterNoise(**param_noise_data)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        memory = ReplayBuffer(
            buffer_size = BUFFER_SIZE,
            batch_size = BATCH_SIZE,
            seed = SEED,
            device = device,
        )
        agents = [DDPGAgent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            memory=memory,
            gamma=GAMMA,
            tau=TAU,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            learn_f=LEARN_F,
            weight_decay=WEIGHT_DECAY,
            device=device,
            random_seed=SEED,
            upper_bound=UPPER_BOUND,
            actor_hidden=ACTOR_HIDDEN,
            critic_hidden=CRITIC_HIDDEN,
            actor_act=cls.get_function(ACTOR_ACT),
            critic_act=cls.get_function(CRITIC_ACT),
            batch_norm=BATCH_NORM,
            add_noise=ADD_NOISE[idx],
            action_noise=action_noise,
            noise_decay=NOISE_DECAY,
            param_noise=param_noise,
        ) for idx in range(N_AGENTS)]
        return MADDPGAgent(agents)


    @classmethod
    def get_trainer(
        cls, data: Dict[str, Any], envh: UnityEnvMgr, magent: MADDPGAgent
    ) -> TennisTrainer:
        """Configure trainer with environment and agents given config file"""
        # Unpack config
        # # Environment
        # ENV_FILE: str                  = data['environment']['ENV_FILE']
        # STATE_SIZE: int                = data['environment']['STATE_SIZE']
        # ACTION_SIZE: int               = data['environment']['ACTION_SIZE']
        # UPPER_BOUND: float             = data['environment']['UPPER_BOUND']
        SOLVED: float                  = data['environment']['SOLVED']
        ROOT_NAME: str                 = data['environment']['ROOT_NAME']

        # # Trainer
        Trainer                        = TennisTrainer
        N_EPISODES: int                = data['trainer']['N_EPISODES']
        MAX_T: int                     = data['trainer']['MAX_T']
        WINDOW_LEN: int                = data['trainer']['WINDOW_LEN']

        # logger.debug(f'Device Info:{device}')
        # logger.debug(f'Config Data: {data}')
        # torch.device("cpu")
        trainer = Trainer(
            magent=magent,
            env=envh,
            n_episodes=N_EPISODES,
            max_t=MAX_T,
            window_len=WINDOW_LEN,
            solved=SOLVED,
            save_root=ROOT_NAME,
        )
        return trainer

    @staticmethod
    def get_function(func_name: str):
        """Get function from torch.nn.funtional module"""
        return getattr(F, func_name)

    def __call__(self, config: str) -> Trainer:
        """Reuturn Complete Trainer given configuration"""
        data = self.read_toml(config)

        envh = self.get_environment_manager(data)
        maddpg = self.get_agent(data)
        return self.get_trainer(data, envh, maddpg)
