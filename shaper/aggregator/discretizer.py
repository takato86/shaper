
from typing import Any, Dict, Optional

import numpy as np
from shaper.aggregator.interface import AbstractAggregator
from shaper.splitter import Splitter
from shaper.value import AbstractValue, TableValue


class Discretizer(AbstractAggregator):
    def __init__(self, splitter: Splitter):
        self.splitter = splitter

    def __call__(self, obs: np.ndarray, done: bool, info: dict[str, Any]):
        return self.splitter.eval(obs)

    def reset(self):
        pass

    def get_n_states(self):
        return self.splitter.n_states

    def create_vfunc(self, values: Optional[Dict[int, float]] = None) -> AbstractValue:
        return TableValue(self.splitter.n_states, values)
