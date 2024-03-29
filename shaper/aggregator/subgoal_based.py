import os
import logging
from typing import Any, Dict, Optional, Callable

import numpy as np
from shaper.achiever import AbstractAchiever

from shaper.aggregator.interface import AbstractAggregator
from shaper.value import AbstractValue, TableValue


logger = logging.getLogger(__name__)


class DynamicTrajectoryAggregation(AbstractAggregator[int]):
    def __init__(self, achiever: AbstractAchiever, is_success: Callable[[bool, Dict[str, Any]], bool]):
        self.achiever = achiever
        self.is_success = is_success
        self.current_state = 0
        # +2 consists of abstract state before achieving and at end state.
        self.n_states = len(self.achiever.subgoals) + 2

    def __call__(self, obs: np.ndarray, done: bool, info: Dict[str, Any]) -> int:
        # if self.is_success(done, info):
        #     # ゴール直前の抽象状態からの遷移のみを受け付ける
        #     self.current_state = self.n_states - 1
        #     logger.debug(
        #         "The episode is succeeded! Transit to {} at {}".format(
        #             self.current_state, os.getpid()
        #         )
        #     )
        #     return self.current_state

        if self.achiever.eval(obs, self.current_state):
            self.current_state += 1
            logger.debug(
                "Subgoal is achieved! Transit to {} at {}".format(
                    self.current_state, os.getpid()
                )
            )
            return self.current_state

        return self.current_state

    def reset(self) -> None:
        self.current_state = 0

    def get_n_states(self) -> int:
        return self.n_states

    def get_current_state(self) -> int:
        return self.current_state

    def create_vfunc(self, values: Optional[Dict[int, float]] = None) -> AbstractValue:
        return TableValue(self.n_states, values)


class Checker(AbstractAggregator):
    def __init__(self, achiever: AbstractAchiever):
        self.achiever = achiever
        self.current_state = 0
        self.n_states = len(self.achiever.subgoals) + 1

    def __call__(self, obs, done, info):
        if self.achiever.eval(obs, self.current_state, done, info):
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

    def create_vfunc(self, values: Optional[Dict[int, float]] = None) -> AbstractValue:
        return TableValue(self.n_states, values)
