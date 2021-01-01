from gym.spaces import Box, Dict
from shaner.utils import get_box


class AbstractTransformer:
    def __call__(self, obs):
        raise NotImplementedError


class TwoDiscretizer(AbstractTransformer):
    def __init__(self, env, clip_range=None):
        self.env = env
        obs_box = get_box(env.observation_space)
        self.center = self.__cal_center(obs_box, clip_range)
        self.shape = obs_box.shape
        self.idxs = list(range(2**self.shape[0]))

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


class ExampleTransformer(AbstractTransformer):
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        # Plese write your transfromation function from raw observation to
        # abstract state.
        pass
