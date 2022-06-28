import abc
import numpy as np
from typing import Any, Dict, Optional


class AbstractShaping(metaclass=abc.ABCMeta):
    @classmethod
    @property
    @abc.abstractmethod
    def is_learn(cls) -> bool:
        """flag if the shaping learn."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def current_state(self) -> Optional[np.ndarray]:
        """Return None if the shaping has no internal states."""
        raise NotImplementedError

    @abc.abstractmethod
    def shape(self, aobs: np.ndarray, aobs2: np.ndarray) -> float:
        """Return a shaping reward without internal updates."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, pre_obs: np.ndarray, action: np.ndarray, reward: float,
             obs: np.ndarray, done: bool, info: Dict[str, Any]) -> float:
        """Return a shaping reward with internal updates."""
        raise NotImplementedError
