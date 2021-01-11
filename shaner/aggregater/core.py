import numpy as np
from gym.spaces import Box, Dict
from shaner.utils import get_box, n_ary2decimal
from shaner.aggregater.entity.splitter import Splitter


class AbstractTransformer:
    def __call__(self, obs, done):
        raise NotImplementedError
    
    def get_n_states(self):
        raise NotImplementedError


class Discretizer(AbstractTransformer):
    def __init__(self, env, n, clip_range=None):
        self.env = env
        self.n = n
        obs_box = get_box(env.observation_space)
        self.shape = obs_box.shape
        self.splitter = self.__create_splitter(obs_box, clip_range)
        self.n_states = n ** self.shape[0]

    def __call__(self, obs, done):
        three_ary = self.splitter.eval(obs)
        return n_ary2decimal(three_ary, self.n)

    def __create_splitter(self, box, clip_range):
        if clip_range is not None:
            low = np.array([-clip_range] * self.shape[0])
            high = np.array([clip_range] * self.shape[0])
        else:
            low = box.low
            high = box.high
        return Splitter(low, high, self.n)

    def get_n_states(self):
        return self.n_states


class ExampleTransformer(AbstractTransformer):
    def __init__(self, env):
        self.env = env

    def __call__(self, obs, done):
        # Plese write your transfromation function from raw observation to
        # abstract state.
        pass
