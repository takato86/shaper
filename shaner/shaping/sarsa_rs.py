from shaner.factory import ValueFactory
from shaner.reward import HighReward
from shaner.transformer import TwoDiscretizer


class SarsaRS:
    def __init__(self, gamma, lr, env, params):
        self.gamma = gamma
        self.lr = lr
        # TODO factory method pattern
        self.transformer = TwoDiscretizer(env=env,
                                          clip_range=params['clip_range'])
        self.vfunc = ValueFactory().create(params['vid'], env=env,
                                           transformer=self.transformer)
        # TODO factory method pattern
        self.high_reward = HighReward(gamma=gamma)
        self.t = 0  # timesteps during abstract states.
        self.pz = None  # previous abstract state.

    def start(self, obs):
        self.pz = self.transformer(obs)

    def train(self, pre_obs, obs, reward):
        if self.pz is None:
            self.pz = self.transformer(pre_obs)
        z = self.transformer(obs)
        self.t += 1
        self.high_reward.update(reward)
        if self.pz != z or self.high_reward() > 0:
            target = reward + self.gamma ** self.t * self.vfunc(obs)
            td_error = target - self.vfunc(pre_obs)
            self.vfunc.update(pre_obs,
                              self.vfunc(pre_obs) + self.lr * td_error)
            self.t = 0
        self.pz = z

    def perform(self, pre_obs, obs, reward, done):
        self.train(pre_obs, obs, reward)
        v = self.gamma * self.potential(obs) - self.potential(pre_obs)
        if done:
            self.reset()
        return v

    def potential(self, obs):
        return self.vfunc(obs)

    def reset(self):
        self.t = 0
        self.pz = None
        self.high_reward.reset()
