
from shaper.aggregater.interface import AbstractAggregater
from shaper.splitter import Splitter


class Discretizer(AbstractAggregater):
    def __init__(self, splitter: Splitter):
        self.splitter = splitter

    def __call__(self, obs):
        return self.splitter.eval(obs)

    def reset(self):
        pass

    def get_n_states(self):
        return self.splitter.n_states
