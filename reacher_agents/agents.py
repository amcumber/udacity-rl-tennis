from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def act(self, state, eps=0.0):
        pass

    @abstractmethod
    def save(self, file):
        pass

    @abstractmethod
    def load(self, file):
        pass


class MultiAgent(Agent):
    @abstractmethod
    def step(self, states, actions, rewards, next_states, dones):
        pass
