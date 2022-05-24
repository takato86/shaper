from shaner.factory import ValueFactory, AggregaterFactory
from shaner.reward import HighReward
from shaner.utils import decimal_calc
from shaner.shaping.interface import AbstractShaping
import warnings


class SarsaRS(AbstractShaping):
    is_learn = True

    def __init__(self, gamma, lr, aggr_id, abstractor, vid,
                 is_success, values=None):
        self.gamma = gamma
        self.lr = lr
        self.aggregater = AggregaterFactory.create(aggr_id,
                                                   abstractor)
        self.vfunc = ValueFactory.create(vid,
                                         n_states=self.aggregater.n_states,
                                         values=values)
        self.high_reward = HighReward(gamma=gamma)
        self.t = -1  # timesteps during abstract states.
        self.pz = None  # previous abstract state.
        # for analysis varibles
        self.counter_transit = 0
        self.is_success = is_success
        # Pick and Place domainのように成功しても続く環境などのために必要。
        self.is_pre_success = False

    def start(self, obs):
        self.counter_transit = 0
        self.pz = self.aggregater(obs)

    def __train(self, z, reward, done, info):
        self.t += 1
        # 成功軌跡でゴール報酬が0の場合、ターゲットとすると不都合
        # ゴール: 0、ステップ: -1の場合抽象状態空間のゴールを含む状態の価値関数が小さくなる。
        is_end_update = self.is_success(done, info)
        self.high_reward.update(reward, self.t)

        if self.pz != z or (is_end_update and not self.is_pre_success):
            assert self.t >= 0

            # 成功した時はrewardをValueとする。
            if is_end_update:
                target = self.high_reward()
            else:
                target = self.high_reward() + \
                    self.gamma ** (self.t + 1) * self.vfunc(z)

            td_error = target - self.vfunc(self.pz)
            self.vfunc.update(self.pz,
                              self.vfunc(self.pz) + self.lr * td_error)
            self.t = -1
            self.high_reward.reset()
            self.is_pre_success = is_end_update

    def perform(self, pre_obs, reward, obs, done, info):
        """[DEPRECATED] return the shaping reward and train the potentials

        Args:
            pre_obs (_type_): _description_
            reward (_type_): _description_
            obs (_type_): _description_
            done (function): _description_
            info (_type_): _description_

        Returns:
            _type_: _description_
        """
        warnings.warn("deprecated", DeprecationWarning)
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

    def step(self, pre_obs, pre_action, reward, obs, done, info):
        """return the shaping reward and train the potentials

        Args:
            pre_obs (_type_): _description_
            pre_action (_type_): _description_
            reward (_type_): _description_
            obs (_type_): _description_
            done (function): _description_
            info (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.pz is None:
            self.pz = self.aggregater(pre_obs)
        z = self.aggregater(obs)
        if self.pz != z:
            self.counter_transit += 1
        v = self.shape(self.pz, z)
        # trainの前に価値関数を計算しておく。
        self.__train(z, reward, done, info)
        self.pz = z
        if done:
            self.reset()
        return v

    def shape(self, pz, z):
        r = decimal_calc(
            self.gamma * self.potential(z),
            self.potential(pz),
            "-"
        )
        return r

    def potential(self, z):
        return self.vfunc(z)

    def reset(self):
        self.t = 0
        self.pz = None
        self.high_reward.reset()
        self.aggregater.reset()

    def get_counter_transit(self):
        return self.counter_transit

    def get_current_state(self):
        return self.aggregater.get_current_state()


class OffsetSarsaRS(SarsaRS):
    def __init__(self, gamma, lr, env, aggr_id, abstractor, vid,
                 is_success, values=None):
        super().__init__(gamma, lr, env, aggr_id, abstractor, vid,
                         is_success, values)

    def potential(self, z):
        """必ず正のポテンシャルが生成されるようにオフセット."""
        potential = super().potential(z)
        min_potential = self.vfunc.get_min_value()
        return potential - min_potential
