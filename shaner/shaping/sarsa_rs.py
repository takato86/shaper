from shaner.factory import ValueFactory, AggregaterFactory
from shaner.reward import HighReward
from shaner.utils import decimal_calc


class SarsaRS:
    def __init__(self, gamma, lr, env, params, is_success):
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
        # 成功軌跡でゴール報酬が0の場合、ターゲットとすると不都合
        # ゴール: 0、ステップ: -1の場合抽象状態空間のゴールを含む状態の価値関数が小さくなる。
        is_end_update = self.is_success(done, info) and self.high_reward() > 0
        if self.pz != z or is_end_update:
            assert self.t > 0
            # 成功した時はrewardをValueとする。
            value = reward if self.is_success(done, info) else self.vfunc(z)
            target = self.high_reward() + self.gamma ** self.t * value
            td_error = target - self.vfunc(self.pz)
            self.vfunc.update(self.pz,
                              self.vfunc(self.pz) + self.lr * td_error)
            self.t = 0
            self.high_reward.reset()
        self.high_reward.update(reward, self.t)

    def perform(self, pre_obs, obs, reward, done, info):
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
        # trainの前に価値関数を計算しておく。
        self.__train(z, reward, done, info)
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
