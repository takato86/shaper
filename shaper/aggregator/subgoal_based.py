import os
import logging
from typing import Any, Dict, Optional, Callable, Tuple

import numpy as np
from shaper.achiever import AbstractAchiever

from shaper.aggregator.interface import AbstractAggregator
from shaper.transiter import AbstractTransiter
from shaper.value import AbstractValue, TableValue


logger = logging.getLogger(__name__)


class DynamicTrajectoryAggregation(AbstractAggregator[int]):
    def __init__(self, achiever: AbstractAchiever, is_success: Callable[[bool, Dict[str, Any]], bool]):
        self.achiever = achiever
        self.is_success = is_success
        self.__current_state = 0
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

        if self.achiever.eval(obs, self.__current_state):
            self.__current_state += 1
            logger.debug(
                "Subgoal is achieved! Transit to {} at {}".format(
                    self.__current_state, os.getpid()
                )
            )
            return self.__current_state

        return self.__current_state

    @property
    def current_state(self) -> int:
        return self.__current_state

    def reset(self) -> None:
        self.__current_state = 0

    def get_n_states(self) -> int:
        return self.n_states

    def create_vfunc(self, values: Optional[Dict[int, float]] = None) -> AbstractValue:
        return TableValue(self.n_states, values)


class DynamicStateAggregation(AbstractAggregator[int]):
    def __init__(self, transiter: AbstractTransiter, is_success: Callable[[bool, Dict[str, Any]], bool]):
        self.transiter = transiter
        self.is_success = is_success
        self.__current_state = self.transiter.reset()
        # +2 consists of abstract state before achieving and at end state.

    def __call__(self, obs: np.ndarray, done: bool, info: Dict[str, Any]) -> int:
        self.__current_state = self.transiter.transit(obs, self.__current_state)
        return self.__current_state

    @property
    def current_state(self) -> int:
        return self.__current_state

    def reset(self) -> None:
        self.__current_state = self.transiter.reset()

    def get_n_states(self) -> Tuple[int]:
        return self.transiter.n_states

    def create_vfunc(self, values: Optional[np.ndarray] = None) -> AbstractValue:
        return TableValue(self.transiter.n_states, values)


class Checker(AbstractAggregator[int]):
    def __init__(self, achiever: AbstractAchiever):
        self.achiever = achiever
        self.__current_state = 0
        self.n_states = len(self.achiever.subgoals) + 1

    def __call__(self, obs: Any, done: bool, info: Dict[str, Any]) -> int:
        if self.achiever.eval(obs, self.__current_state):
            self.__current_state += 1
            logger.debug(
                "Subgoal is achieved! Transit to {}".format(self.__current_state)
            )
        return self.__current_state

    @property
    def current_state(self) -> int:
        return self.__current_state

    def reset(self):
        self.__current_state = 0

    def get_n_states(self):
        return self.n_states

    def create_vfunc(self, values: Optional[Dict[int, float]] = None) -> AbstractValue:
        return TableValue(self.n_states, values)
