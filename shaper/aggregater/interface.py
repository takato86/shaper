import abc
import numpy as np
from typing import Generic, TypeVar


AggregatedState = TypeVar("AggregatedState")


class AbstractAggregater(Generic[AggregatedState], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, obs: np.ndarray) -> AggregatedState:
        """observation into internal state."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_n_states(self) -> int:
        """return the number of internal states."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """reset internal properties in case of necessity."""
        raise NotImplementedError
