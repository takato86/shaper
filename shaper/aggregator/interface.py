import abc
import numpy as np
from typing import Any, Dict, Generic, Optional, TypeVar

from shaper.value import AbstractValue


AggregatedState = TypeVar("AggregatedState")


class AbstractAggregator(Generic[AggregatedState], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, obs: np.ndarray, done: bool, info: dict[str, Any]) -> AggregatedState:
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

    @abc.abstractmethod
    def create_vfunc(self, values: Optional[Dict[int, float]] = None) -> AbstractValue:
        """return Value object. The size of states is dependent to aggregation."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def current_state(self) -> AggregatedState:
        """return current state."""
        raise NotImplementedError