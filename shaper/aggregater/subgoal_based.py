import os
import logging

import numpy as np
from shaper.achiever import BaseAchiever

from shaper.aggregater.interface import AbstractAggregater


logger = logging.getLogger(__name__)


class DynamicTrajectoryAggregation(AbstractAggregater[int]):
    def __init__(self, achiever: BaseAchiever):
        self.achiever = achiever
        self.current_state = 0
        self.n_states = len(self.achiever.subgoals) + 1

    def __call__(self, obs: np.ndarray) -> int:
        if self.achiever.eval(obs, self.current_state):
            self.current_state += 1
            logger.debug(
                "Subgoal is achieved! Transit to {} at {}".format(
                    self.current_state, os.getpid()
                )
            )
            return self.current_state
        else:
            return self.current_state

    def reset(self) -> None:
        self.current_state = 0

    def get_n_states(self) -> int:
        return self.n_states

    def get_current_state(self) -> int:
        return self.current_state


class Checker(AbstractAggregater):
    def __init__(self, achiever):
        self.achiever = achiever
        # id2achiever[env_id](n_obs=n_obs, _range=_range,
        #                                     **params)
        self.current_state = 0
        self.n_states = len(self.achiever.subgoals) + 1

    def __call__(self, obs):
        if self.achiever.eval(obs, self.current_state):
            self.current_state += 1
            logger.debug(
                "Subgoal is achieved! Transit to {}".format(self.current_state)
            )
            return True
        else:
            return False

    def reset(self):
        self.current_state = 0

    def get_n_states(self):
        return self.n_states
