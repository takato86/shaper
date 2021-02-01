from shaner.factory import ValueFactory, AggregaterFactory
from shaner.reward import HighReward


class SarsaRS:
    def __init__(self, gamma, lr, env, params):
        self.gamma = gamma
        self.lr = lr
        self.aggregater = AggregaterFactory().create(params['aggr_id'],
                                                     params['params'])
        self.vfunc = ValueFactory().create(params['vid'],
                                           n_states=self.aggregater.n_states,
                                           env=env)
        # TODO factory method pattern
        self.high_reward = HighReward(gamma=gamma)
        self.t = 0  # timesteps during abstract states.
        self.pz = None  # previous abstract state.

    def start(self, obs):
        self.pz = self.aggregater(obs, False)

    def __train(self, pz, z, reward, done):
        self.t += 1
        self.high_reward.update(reward)
        if self.pz != z or self.high_reward() > 0:
            assert self.t > 0
            target = reward + self.gamma ** self.t * self.vfunc(z)
            td_error = target - self.vfunc(pz)
            self.vfunc.update(pz,
                              self.vfunc(pz) + self.lr * td_error)
            self.t = 0

    def perform(self, pre_obs, obs, reward, done):
        # import pdb; pdb.set_trace()
        if self.pz is None:
            self.pz = self.aggregater(pre_obs, False)
        z = self.aggregater(obs, done)
        v = self.gamma * self.potential(z) - self.potential(self.pz)
        self.__train(self.pz, z, reward, done)
        # trainでself.pzの価値関数が更新されたときの挙動は？
        # Dynamic PBRSに従えば、更新前の値を使うべき
        self.pz = z
        if done:
            self.reset()
        return v

    def potential(self, z):
        return self.vfunc(z)

    def reset(self):
        self.t = 0
        self.pz = None
        self.high_reward.reset()
