from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
import torch
import torch.nn.functional as F

from tennis_agents.ddpg_agent import DDPGMultiAgent
from tennis_agents.trainers import TennisTrainer
from tennis_agents.unity_environments import UnityEnvMgr
from tennis_agents.replay_buffers import ReplayBuffer


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


class ConfigFileFactory(TrainerFactory):
    @classmethod
    def get_trainer(cls, data: Dict[str, Any]):
        """Configure trainer with environment and agents given config file"""
        # Unpack config
        # # Environment
        ENV_FILE: str                  = data['environment']['ENV_FILE']
        STATE_SIZE: int                = data['environment']['STATE_SIZE']
        ACTION_SIZE: int               = data['environment']['ACTION_SIZE']
        UPPER_BOUND: float             = data['environment']['UPPER_BOUND']
        SOLVED: float                  = data['environment']['SOLVED']
        ROOT_NAME: str                 = data['environment']['ROOT_NAME']

        # # Trainer
        Trainer                        = TennisTrainer
        BATCH_SIZE: int                = data['trainer']['BATCH_SIZE']
        N_EPISODES: int                = data['trainer']['N_EPISODES']
        MAX_T: int                     = data['trainer']['MAX_T']
        WINDOW_LEN: int                = data['trainer']['WINDOW_LEN']

        # # Agent
        N_AGENTS: int                  = data['agent']['N_AGENTS']
        BUFFER_SIZE: int               = data['agent']['BUFFER_SIZE']
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
        SEED: int                      = data['agent']['SEED']
        BATCH_NORM: float              = data['agent']['BATCH_NORM']

        envh = UnityEnvMgr(ENV_FILE)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # torch.device("cpu")
        memory = ReplayBuffer(
            buffer_size = BUFFER_SIZE,
            batch_size = BATCH_SIZE,
            seed = SEED,
            device = device,
        )
        agents = [DDPGMultiAgent(
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
            add_noise=ADD_NOISE[idx],
            batch_norm=BATCH_NORM,
        ) for idx in range(N_AGENTS)]
        trainer = Trainer(
            agents=agents,
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

def plot_scores(scores, i_map=0):
    sns.set_style('darkgrid')
    sns.set_context('talk')
    sns.set_palette('Paired')
    cmap = sns.color_palette('Paired')

    scores = np.array(scores).squeeze()
    score_df = pd.DataFrame({'scores': scores})
    score_df = score_df.assign(mean=lambda df: df.rolling(10).mean()['scores'])

    _ ,ax = plt.subplots(1,1, figsize=(10,8))

    ax = score_df.plot(ax=ax, color=cmap[2*(i_map%4):])
    ax.set_title('DDPG Scores vs Time for Tennis')
    ax.set_xlabel('Episode #')
    ax.set_ylabel('Score')
    plt.show()


def config_main(file:str) -> None:
    """
    Main Function that will load configuration from file, train, and plot the
    resulting scores
    """
    factory = ConfigFileFactory()
    data = factory.read_toml(file)
    trainer = factory.get_trainer(data)
    scores = trainer.train()
    plot_scores(scores)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Run Tennis from terminal')
    p.add_argument('config', help='Configuration File - toml')
    args = p.parse_args()

    config_main(args.config)

