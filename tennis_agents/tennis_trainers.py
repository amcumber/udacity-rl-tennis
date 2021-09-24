from os import stat
import random
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple, Type

import numpy as np
import toml
import pickle

from .agents import Agent
from .environments import EnvironmentMgr
from .workspace_utils import keep_awake


class Trainer(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass


class TennisTrainer(Trainer):
    def __init__(
        self,
        agents: tuple[Agent, Agent],
        env: EnvironmentMgr,
        n_episodes: int,
        window_len: int,
        solved: float,
        save_root: str = "checkpoint",
        max_t: int = 2000,
    ):
        """
        Tennis - Repurposed from Continuous Control Submission

        Parameters
        ----------
        agent : Agent
            agent to act upon
        env : UnityEnvironmentMgr
            environment manager containing enter and exit methods to call
            UnityEnvironment
            - DO NOT CLOSE in v0.4.0 - this will cause you to be locked
            out of your environment... NOTE TO UDACITY STAFF - fix this issue
            by upgrading UnityEnvironment requirements. See
            https://github.com/Unity-Technologies/ml-agents/issues/1167
        n_episodes: int
            maximum number of training episodes
        window_len : int (100)
            update terminal with information for every specified iteration,
        solved : float
            score to be considered solved
        save_root: str
            file to save network weights
        max_t : int (2000)
            timeout for an episode

        CITATION: the algorithm for implementing the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforcement Learning Course
        """
        self.agents = agents
        self.env = env
        self.n_episodes = n_episodes

        self.solved = solved
        self.window_len = window_len

        self.scores_ = None
        self.save_root = save_root

        self.SAVE_EVERY = 10

    def _report_score(self, i_episode, scores_window, scores, end="") -> None:
        """
        Report the score
        Parameters
        ----------
        i_episode  : int
            current episode number
        scores_window : deque
            latest scores
        end : str
            how to end the print function ('' will repeat the line)
        n : int
            report last the mean of the last n scores for 'latest score'
        """
        print(
            f"\rEpisode {i_episode+1:d}"
            f"\tAverage Score (episode): {np.mean(scores):.2f}",
            f"\tMax Score (episode): {np.max(scores):.2f}",
            f"\tAverage Score (deque): {np.mean(scores_window):.2f}",
            end=end,
        )

    def _check_solved(self, i_episode, scores_window) -> None:
        if np.mean(scores_window) >= self.solved:
            print(
                f"\nEnvironment solved in {i_episode+1:d} episodes!"
                f"\tAverage Score: {np.mean(scores_window):.2f}"
            )
            return True
        return False

    def _get_save_file(self, root):
        now = datetime.now()
        return f'{root}-{now.strftime("%Y%m%dT%H%M%S")}'

    def train(self, save_all=False, is_cloud=False):
        if save_all:
            save_root = self._get_save_file(self.save_root)
            trainer_file = f'trainer-{save_root}.toml'
            agent_file = f'agent-{save_root}.toml'
            self.save_hyperparameters(trainer_file)
            self.agent.save_hyperparameters(agent_file)
        self.env.start()
        scores_episode = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.window_len)
        rng = range(self.n_episodes)
        if is_cloud:
            rng = keep_awake(rng)
        for i_episode in rng:
            (scores_episode, scores_window, scores) = self._run_episode(
                scores_episode, scores_window
            )
            self.scores_ = scores_episode
            self._report_score(i_episode, scores_window, scores)
            if (i_episode + 1) % self.SAVE_EVERY == 0:
                self._report_score(i_episode, scores_window, scores, end="\n")
                self.agent.save(f"{self.save_root}-agent-checkpoint")
                self.save_scores(f'{self.save_root}-scores-checkpoint.pkl')
            if self._check_solved(i_episode, scores_window):
                self.agent.save(self._get_save_file(f"{self.save_root}-solved"))
                break
        return scores_episode

    def _run_episode(
        self, scores_episode, scores_window, render=False
    ) -> Tuple[list, deque, float]:
        """Run an episode of the training sequence"""
        states = self.env.reset()
        for agent in self.agents:
            agent.reset()
        scores = np.zeros(self.n_workers)
        for _ in range(self.max_t):
            if render:
                self.env.render()
            actions = [agent.act(states) for agent in self.agents]
            next_states, rewards, dones, _ = self.env.step(actions)
            self._step_agents(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        score = np.mean(scores)
        scores_window.append(score)  # save most recent score
        scores_episode.append(score)  # save most recent score
        return (scores_episode, scores_window, scores)

    def _step_agents(self, states, actions, rewards, next_states, dones):
        """
        Step Agents depending on number of workers

        CITATION: the algorithm for implementing the learn_every // update_every
                  was derived from recommendations for the continuous control
                  project as well as reviewing recommendations on the Mentor
                  help forums - Udacity's Deep Reinforcement Learning Course
        """
        for idx in range(self.n_agents):
            self.agent[idx].step(
                states[idx],
                actions[idx],
                rewards[idx],
                next_states[idx],
                dones[idx],
            )

    def eval(self, n_episodes=3, render=False):
        ## scores_window
        scores_episode = []
        scores_window = deque(maxlen=self.window_len)
        for i in range(n_episodes):
            (scores_episode, scores_window) = self._run_episode(
                scores_episode, scores_window, render=render
            )
            self.scores_ = scores_episode
            print(f"\rEpisode {i+1}\tFinal Score: {np.mean(scores_episode):.2f}", end="")
        return scores_episode

    def save_hyperparameters(self, file: str) -> None:
        """
        Save trainer meta data into a toml file
        """
        if file is None:
            raise ValueError("File must be specified")
        with Path(file).open("w") as fh:
            toml.dump(self.__dict__, fh)

    def save_scores(self, file: str) -> None:
        scores = self.scores_
        with Path(file).open("wb") as fh:
            pickle.dump(scores, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def read_scores(self, file: str) -> list:
        """Read Scores from pickle file
        
        NOTE: DO NOT LOAD pickle files generated from a source you do not trust!
        """
        with Path(file).open("rb") as fh:
            scores = pickle.load(fh)
        return scores
