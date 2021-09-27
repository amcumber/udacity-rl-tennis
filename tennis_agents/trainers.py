import pickle
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple

# import logging

import numpy as np
import toml

from .agents import MultiAgent
from .environments import EnvironmentMgr
from .workspace_utils import keep_awake

# logger = logging.getLogger('TennisTrainer')


class Trainer(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass


class TennisTrainer(Trainer):
    SAVE_EVERY = 100

    def __init__(
        self,
        magent: MultiAgent,
        env: EnvironmentMgr,
        n_episodes: int,
        window_len: int,
        solved: float,
        save_root: str = "checkpoint",
        save_dir: str = "runs",
        max_t: int = 2000,
    ):
        """
        Tennis - Repurposed from Continuous Control Submission

        Parameters
        ----------
        magent : MultiAgent
            multi-agent to act upon
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
        self.magent = magent
        self.env = env
        self.n_episodes = n_episodes
        self.solved = solved
        self.window_len = window_len
        self.scores_ = None
        self.save_root = save_root
        self.save_dir = Path(save_dir)
        self.max_t = max_t
        self.n_workers = len(self.magent)

    def _report_score(self, i, s_window, best_score, end="") -> None:
        """
        Report the score
        Parameters
        ----------
        i : int
            current episode number
        s_window : deque
            latest scores
        end : str
            how to end the print function ('' will repeat the line)
        n : int
            report last the mean of the last n scores for 'latest score'
        """
        msg = (
            f"\rEp {i+1:d}"
            + f"\tBest (all): {best_score:.4f}"
            f"\tMean (window): {np.mean(s_window):.4f}"
        )
        # logger.info(msg)
        print(msg, end=end)

    def _check_solved(self, i_episode, scores_window) -> None:
        if np.mean(scores_window) >= self.solved:
            print(
                f"\nEnvironment solved in {i_episode+1:d} episodes!"
                f"\tAverage Score: {np.mean(scores_window):.4f}"
            )
            return True
        return False

    def _get_save_file(self, root):
        now = datetime.now()
        return f'{root}-{now.strftime("%Y%m%dT%H%M%S")}'

    def save(self):
        save_root = self._get_save_file(self.save_root)
        trainer_file = self.save_dir / f"trainer-{save_root}.toml"
        agent_file = self.save_dir / f"agent-{save_root}.toml"
        self.save_hyperparameters(trainer_file)

    def train(self):
        self.env.start()
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.window_len)
        best_score = -np.inf
        for i_episode in range(self.n_episodes):
            (scores, scores_window, indiv_score) = self._run_episode(scores, scores_window)
            if indiv_score > best_score:
                best_score = indiv_score
            self.scores_ = scores
            self._report_score(i_episode, scores_window, best_score)
            if (i_episode + 1) % self.SAVE_EVERY == 0:
                self._report_score(
                    i_episode, scores_window, best_score, end="\n"
                )
                self.magent.save(self.save_dir / f"{self.save_root}-agent-checkpoint")
                self.save_scores(
                    self.save_dir / f"{self.save_root}-scores-checkpoint.pkl"
                )
            if self._check_solved(i_episode, scores_window):
                self.magent.save(
                    self.save_dir / self._get_save_file(f"{self.save_root}-solved")
                )
                break
        return scores

    def _run_episode(
        self, scores, scores_window, render=False
    ) -> Tuple[list, deque, float]:
        """Run an episode of the training sequence"""
        states = self.env.reset()
        self.magent.reset()
        score = np.zeros(len(self.magent))
        for _ in range(self.max_t):
            if render:
                self.env.render()
            actions = self.magent.act(states)
            next_states, rewards, dones, _ = self.env.step(actions)
            self.magent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if np.any(dones):
                break
        score = np.max(score)  # Specific to Tennis!
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        idx = np.array(score).argmax()
        best_agent = self.magent.agents[idx]
        for i, agent in enumerate(self.magent.agents):
            if i == idx:
                pass
            agent.copy_from(best_agent)
        self.magent.cleanup()
        return (scores, scores_window, score)

    def eval(self, n_episodes=3, render=False):
        ## scores_window
        scores = []
        scores_window = deque(maxlen=self.window_len)
        for agent in self.magent.agents:
            agent.add_noise = False
        for i in range(n_episodes):
            (scores, scores_window) = self._run_episode(
                scores, scores_window, render=render
            )
            self.scores_ = scores
            print(
                f"\rEpisode {i+1}\tFinal Score: {np.mean(scores):.2f}", end=""
            )
        return scores

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
