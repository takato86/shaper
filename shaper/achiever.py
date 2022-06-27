import abc
from typing import List

import numpy as np


class BaseAchiever(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eval(self, obs: np.ndarray, subgoal_idx: int) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def subgoals(self) -> List[any]:
        raise NotImplementedError
