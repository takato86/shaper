import numpy as np
from shaner.utils import get_box
from shaner.aggregater.entity.splitter import Splitter, NSplitter


class AbstractAggregater:
    def __call__(self, obs):
        raise NotImplementedError
    
    def get_n_states(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class Discretizer(AbstractAggregater):
    def __init__(self, env, n, clip_range=None):
        self.env = env
        self.n = n
        obs_box = get_box(env.observation_space)
        self.shape = obs_box.shape
        self.splitter = self.__create_splitter(obs_box, clip_range)
        self.n_states = n ** self.shape[0]

    def __call__(self, obs):
        return self.splitter.eval(obs)

    def __create_splitter(self, box, clip_range):
        if clip_range is not None:
            low = np.array([-clip_range] * self.shape[0])
            high = np.array([clip_range] * self.shape[0])
        else:
            low = box.low
            high = box.high
        return Splitter(low, high, self.n)

    def reset(self):
        pass

    def get_n_states(self):
        return self.n_states


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

