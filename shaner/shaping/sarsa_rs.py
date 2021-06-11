from shaner.factory import ValueFactory, AggregaterFactory
from shaner.reward import HighReward
from shaner.utils import decimal_calc


class SarsaRS:
    def __init__(self, gamma, lr, env, params, is_success=None):
        self.gamma = gamma
        self.lr = lr
        self.aggregater = AggregaterFactory().create(params['aggr_id'],
                                                     params['params'])
        self.vfunc = ValueFactory().create(params['vid'],
                                           n_states=self.aggregater.n_states,
                                           env=env,
                                           values=params.get('values'))
        # TODO factory method pattern
        self.high_reward = HighReward(gamma=gamma)
        self.t = 0  # timesteps during abstract states.
        self.pz = None  # previous abstract state.
        # for analysis varibles
        self.counter_transit = 0
        self.is_success = is_success

    def start(self, obs):
        self.counter_transit = 0
        self.pz = self.aggregater(obs)

    def __train(self, z, reward, done, info):
        self.t += 1
        if self.pz != z or self.is_success(done, info):
            assert self.t > 0
            if self.is_success(done, info):
                # 成功したときだけこの式。失敗した場合は更新を行わない。
                target = self.high_reward() + self.gamma ** self.t * reward
            else:
                target = self.high_reward() + self.gamma ** self.t * self.vfunc(z)
            td_error = target - self.vfunc(self.pz)
            self.vfunc.update(self.pz,
                              self.vfunc(self.pz) + self.lr * td_error)
            self.t = 0
            self.high_reward.reset()
        self.high_reward.update(reward, self.t)

    def perform(self, pre_obs, obs, reward, done, info=None):
        # import pdb; pdb.set_trace()
        if self.pz is None:
            self.pz = self.aggregater(pre_obs)
        z = self.aggregater(obs)
        if self.pz != z:
            self.counter_transit += 1
        v = decimal_calc(
            self.gamma * self.potential(z),
            self.potential(self.pz),
            "-"
        )
        self.__train(z, reward, done, info)
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
        self.aggregater.reset()

    def get_counter_transit(self):
        return self.counter_transit
