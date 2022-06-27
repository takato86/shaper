import logging
import numpy as np
from shaper.utils import n_ary2decimal

logger = logging.getLogger(__name__)


class Splitter:
    def __init__(self, low, high, k):
        self.k = k
        self._ranges = self.__make(low, high)
        self.__n_states = k**len(low)

    @property
    def n_states(self) -> int:
        return self.__n_states

    def __make(self, low, high):
        quantile = (high - low) / self.k
        splitter = {"lower": [], "higher": []}
        for i in range(self.k):
            n_i = i + 1
            lower = low + quantile * i
            higher = low + quantile * (n_i)
            if n_i == self.k:
                higher = higher if all(higher == high) else high
            splitter["lower"].append(lower)
            splitter["higher"].append(higher)
        assert len(splitter["lower"]) == len(splitter["higher"])
        return splitter

    def eval(self, obs):
        logger.debug(obs)

        if type(obs) != np.ndarray:
            obs = np.array(obs)

        res = np.full(obs.shape, -1)

        for j, _range in enumerate(zip(self._ranges["lower"],
                                       self._ranges["higher"])):
            lower, higher = _range
            lower_bool = lower <= obs
            higher_bool = obs <= higher
            target_bool = lower_bool & higher_bool
            idxs = np.argwhere(target_bool)
            res[idxs] = j

        assert len(np.where(res < 0)[0]) == 0
        return n_ary2decimal(res, self.k)


class NSplitter(Splitter):
    def __init__(self, low, high, k):
        # Create k abstract states. (k-1) splitted states and the other state.
        super().__init__(low, high, k-1)

    def eval(self, obs):
        for i in range(self.k):
            lower = self._ranges["lower"][i]
            higher = self._ranges["higher"][i]
            if all(lower <= obs) and all(obs <= higher):
                return i
        return self.k
