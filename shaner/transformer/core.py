

class AbstractTransformer:
    def __call__(self, obs):
        raise NotImplementedError


class TwoDiscretizer(AbstractTransformer):
    def __init__(self, env):
        self.env = env
        self.center = (env.observation_space.high + env.observation_space.low) / 2
        self.shape = env.observation_space.shape
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


class ExampleTransformer(AbstractTransformer):
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        # Plese write your transfromation function from raw observation to
        # abstract state.
        pass
