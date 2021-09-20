import time
from typing import Tuple

from unityagents import UnityEnvironment

from .environments import (
    EnvEnum,
    EnvironmentMgr,
    EnvironmentNotLoadedError,
)


def validate_unity_env(func):
    """Validate the environment is not dead - otherwise raises an error."""

    def wrapper(cls, *args, **kwargs):
        if cls.state == EnvEnum.dead:
            msg = "Must Reset Kernel - due to bug in UnityAgents"
            print(msg)
        return func(cls, *args, **kwargs)

    return wrapper


class UnityEnvMgr(EnvironmentMgr):
    """Unity Manager class for the UnityEnvironment."""

    def __init__(self, file):
        """Initialize the Unity environment with the given file."""
        self.file = file

        self.env = None
        self.brain_name = None
        self.action_size = None
        self.state_size = None

        self.n_agents = None
        self.state = EnvEnum.idle

    def __enter__(self):
        return self.start()

    def __exit__(self, e_type, e_value, e_traceback):
        self.close()

    @validate_unity_env
    def reset(self, train_mode=True) -> "states":
        """Reset the Unity environment."""
        if self.state != EnvEnum.active:
            raise EnvironmentNotLoadedError(
                "Environment Not Initialized, run start method"
            )
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        states = env_info.vector_observations
        return states

    @validate_unity_env
    def step(
        self,
        actions,
    ) -> Tuple["next_state", "reward", "done", "env_info"]:
        """Advance the state of the environment given actions."""
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return (next_states, rewards, dones, env_info)

    @validate_unity_env
    def start(self) -> UnityEnvironment:
        """Start the environment with the given train_mode."""
        self.env = self.get_env(self.file)
        time.sleep(2)
        self.brain_name = self.env.brain_names[0]

        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        self.n_agents, self.state_size = states.shape
        return self.env

    @validate_unity_env
    def get_env(self, file) -> UnityEnvironment:
        """Get the environment from UnityEnvironment"""
        if self.state == EnvEnum.active:
            return self.env
        self.state = EnvEnum.active
        return UnityEnvironment(file_name=file)

    def close(self) -> None:
        """Close the environment"""
        self.env.close()
        msg = "Must Reset Kernel - due to bug in UnityAgents"
        print(msg)

    def render(self):
        self.env.render()


# class UnityEnvSingleAgentMgr(UnityEnvMgr):
#     """Unity Manager for a single agent."""

#     def reset(self) -> "state":
#         states = super().reset()
#         return states[0]

#     def step(self, action) -> Tuple["next_state", "reward", "done", "env_info"]:
#         next_states, rewards, dones, env_info = super().step([action])
#         next_state = next_states[0]
#         reward = rewards[0]
#         done = dones[0]
#         return (next_state, reward, done, env_info)

#     def start(self) -> UnityEnvironment:
#         states = super().start()
#         self.n_agents = 1
#         self.state_size = len(states[0])
#         return self.env
