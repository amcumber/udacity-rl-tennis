from typing import Tuple
import gym
import numpy as np

from .environments import EnvironmentMgr, EnvironmentNotLoadedError, EnvEnum


# Classes
class GymEnvironmentMgr(EnvironmentMgr):
    def __init__(self, scenario, seed=42):
        self.scenario = scenario
        self.seed = seed
        
        self.env = None
        self.action_size = None
        self.state_size = None

        self.state = EnvEnum.idle
        
    def __enter__(self):
        return self.start()
    
    def __exit__(self, e_type, e_value, e_traceback):
        self.state = EnvEnum.idle
        self.env.close()
    
    def step(self, action)-> Tuple['next_state', 'reward', 'done', 'env_info']:
        return self.env.step(action)
    
    def reset(self) -> 'state':
        if self.env is None:
            msg = 'Environment Not Initialized, run start method'
            raise EnvironmentNotLoadedError(msg)
        return self.env.reset()
    
    def start(self):
        if self.env is None:
            self.env = self.get_env(self.scenario)
            self.env.seed(self.seed)
            self.state_size = self.env.observation_space.shape[0]
            self.action_size = self.env.action_space.n
        return self.env
    
    def get_env(self, scenario):
        if self.state == EnvEnum.idle:
            return gym.make(scenario)
        self.state = EnvEnum.active
        return self.env

    def render(self):
        self.env.render()
    

class GymContinuousEnvMgr(GymEnvironmentMgr):
    """
    Manager Continuous Gym Environments such as Pendulum-v0
    """
    def reset(self) -> "state":
        """Reset the state of the environment"""
        if self.env is None:
            raise EnvironmentNotLoadedError(
                "Environment Not Initialized, run start method"
            )
        return self.env.reset()

    def start(self):
        """Start the loaded environment"""
        if self.env is None:
            self.env = self.get_env(self.scenario)
            self.env.seed(self.seed)
            self.state_size = self.env.observation_space.shape[0]
            self.action_size = self.env.action_space.shape[0]
        return self.env

    @staticmethod
    def get_env(scenario):
        return gym.make(scenario)

    def close(self):
        self.env.close()
