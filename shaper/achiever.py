import abc
from typing import Any, List

import numpy as np


class AbstractAchiever(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eval(self, obs: np.ndarray, subgoal_idx: int) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def subgoals(self) -> List[Any]:
        raise NotImplementedError
