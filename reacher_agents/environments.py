from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Tuple


# Errors
class EnvironmentNotLoadedError(Exception):
    pass


class EnvironmentResetError(Exception):
    pass


# Enums
class EnvEnum(Enum):
    idle = auto()
    active = auto()
    dead = auto()


# Abstract Classes
class EnvironmentMgr(ABC):
    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, e_type, e_value, e_traceback):
        pass

    @abstractmethod
    def step(self, action) -> Tuple["next_state", "reward", "done", "env_info"]:
        pass

    @abstractmethod
    def reset(self) -> "state":
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def get_env(self, stream):
        pass

    @abstractmethod
    def close(self):
        pass
