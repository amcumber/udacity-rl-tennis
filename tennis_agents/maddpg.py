from typing import List

from .agents import MultiAgent, Agent
from .ddpg_agent import DDPGAgent
from .replay_buffers import Buffer


class MADDPGAgent(MultiAgent):
    """Multi-agent example of DDPGAgent"""
    def __init__(self, agents: List[DDPGAgent]):
        self.agents = agents

    @property
    def n_agents(self):
        return len(self.agents)

    def __len__(self):
        return self.n_agents

    def step(self, states, actions, rewards, next_states, dones) -> None:
        """Step the multi agent"""
        for agent, s, a, r, ns, d in zip(
            self.agents, states, actions, rewards, next_states, dones
        ):
            agent.step(s, a, r, ns, d)

    def act(self, states):
        """Perform inference on the multi agent"""
        return [agent.act(state) for state, agent in zip(states, self.agents)]

    def save(self, root_filename: str = "checkpoint") -> None:
        """Save the Multi Agent"""
        for i, agent in enumerate(self.agents):
            agent.save(root_filename=f"{root_filename}-{i}")

    def load( self, actor_files: List[str], critic_files: List[str]) -> None:
        """Load agents"""
        for agent, actor_f, critic_f in zip(self.agents, actor_files, critic_files):
            agent.load(actor_f, critic_f)

    def reset(self):
        [agent.reset() for agent in self.agents]

    def cleanup(self):
        [agent.cleanup() for agent in self.agents]
