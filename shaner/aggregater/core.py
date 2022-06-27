import numpy as np
from shaner.utils import get_box
from shaner.aggregater.entity.splitter import NSplitter


class AbstractAggregater:
    def __call__(self, obs):
        raise NotImplementedError

    def get_n_states(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class Discretizer(AbstractAggregater):
    def __init__(self, splitter):
        self.splitter = splitter

    def __call__(self, obs):
        return self.splitter.eval(obs)

    def reset(self):
        pass

    def get_n_states(self):
        # TODO ここは要修正。
        return self.splitter.k


class NDiscretizer(Discretizer):
    def __init__(self, env, n, clip_range=None):
        super().__init__(env, n, clip_range)
        obs_box = get_box(env.observation_space)
        self.n_states = n
        self.splitter = self.__create_splitter(obs_box, clip_range)

    def __create_splitter(self, box, clip_range):
        if clip_range is not None:
            low = np.array([-clip_range] * self.shape[0])
            high = np.array([clip_range] * self.shape[0])
        else:
            low = box.low
            high = box.high
        return NSplitter(low, high, self.n)


class ExampleTransformer(AbstractAggregater):
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        # Plese write your transfromation function from raw observation to
        # abstract state.
        pass
