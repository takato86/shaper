import numpy as np

from gym.spaces import Box, Dict
from shaner.utils import get_box, n_ary2decimal
from shaner.transformer.splitter import Splitter


class AbstractTransformer:
    def __call__(self, obs):
        raise NotImplementedError


class TwoDiscretizer(AbstractTransformer):
    def __init__(self, env, clip_range=None):
        self.env = env
        obs_box = get_box(env.observation_space)
        self.center = self.__cal_center(obs_box, clip_range)
        self.shape = obs_box.shape

    def __call__(self, obs):
        """Transform by binary to decimal number.

        Args:
            obs ([type]): [description]

        Returns:
            [type]: [description]
        """
        ind = 0
        for i, i_obs in enumerate(obs):
            if i_obs > self.center[0]:
                ind += 2**i
        return ind

    def __cal_center(self, box, clip_range):
        if clip_range is not None:
            return [0 for _ in range(box.shape[0])]
        else:
            return (box.high + box.low) / 2


class ThreeDiscretizer(AbstractTransformer):
    def __init__(self, env, clip_range=None):
        self.env = env
        obs_box = get_box(env.observation_space)
        self.shape = obs_box.shape
        self.splitter = self.__create_splitter(obs_box, clip_range)

    def __call__(self, obs):
        three_ary = self.splitter.eval(obs)
        return n_ary2decimal(three_ary, 3)

    def __create_splitter(self, box, clip_range):
        if clip_range is not None:
            low = np.array([-clip_range] * self.shape[0])
            high = np.array([clip_range] * self.shape[0])
        else:
            low = box.low
            high = box.high
        return Splitter(low, high, 3)


class ExampleTransformer(AbstractTransformer):
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        # Plese write your transfromation function from raw observation to
        # abstract state.
        pass
