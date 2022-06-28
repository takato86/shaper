import abc
from typing import Any, Dict, Optional


class AbstractValue(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, state: Any) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state: Any, reward: float) -> None:
        raise NotImplementedError


class TableValue(AbstractValue):
    def __init__(self, n_states: int, values: Optional[Dict[int, float]] = None):
        self.n_states = n_states
        self.value = self.__init_value(values=values)

    def __call__(self, state: Any):
        return self.value[state]

    def __init_value(self, values: Optional[Dict[int, float]]) -> Dict[int, float]:
        if values is None:
            return {
                i: 0
                for i in range(self.n_states)
            }
        else:
            if len(values) != self.n_states:
                raise Exception(
                    "Not match the size of values: {} with the number of states: {}".format(
                        len(values), self.n_states
                    )
                )
            values = {
                int(key): value for key, value in values.items()
            }
            return values

    def update(self, state: Any, v: float) -> None:
        self.value[state] = v

    def get_min_value(self) -> float:
        return min(self.value.values())
