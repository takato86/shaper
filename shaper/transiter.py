import abc
from typing import Any, Generic, List, Tuple, TypeVar
import numpy as np

A = TypeVar('A')
S = TypeVar('S')

class AbstractTransiter(Generic[A, S], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self) -> A:
        raise NotImplementedError

    @abc.abstractmethod
    def transit(self, obs: S, abstract_state: A) -> A:
        """decide the next abstract state.

        Args:
            obs (np.ndarray): _description_
            subgoal_idx (int): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            int: next_state_index
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_states(self) -> Tuple[int]:
        raise NotImplementedError
