import logging
from shaner.factory import AggregaterFactory
from shaner.shaping.interface import AbstractShaping
from shaner.utils import decimal_calc


logger = logging.getLogger(__name__)


class SubgoalRS(AbstractShaping):
    def __init__(self, gamma, lr, eta, rho, aggr_id, abstractor):
        self.gamma = gamma
        self.lr = lr
        self.eta = eta
        self.rho = rho
        self.aggregater = AggregaterFactory.create(aggr_id,
                                                   abstractor)
        self.reset()
        self.counter_transit = 0

    def reset(self):
        self.t = 0
        self.pz = None
        self.p_potential = 0
        self.aggregater.reset()

    def start(self, obs):
        self.counter_transit = 0
        self.pz = self.aggregater(obs)

    def perform(self, pre_obs, obs, reward, done, info=None):
        # import pdb; pdb.set_trace()
        if self.pz is None:
            self.pz = self.aggregater(pre_obs)
        z = self.aggregater(obs)
        if self.pz == z:
            self.t += 1
        else:
            self.counter_transit += 1
            self.t = 0
        c_potential = self.potential(z)
        # ※浮動小数点flaotは2進数による10進数の近似を行っている
        v = decimal_calc(
            self.gamma * c_potential,
            self.p_potential,
            "-"
        )
        self.pz = z
        self.p_potential = c_potential
        if done:
            self.reset()
        return v

    def step(self, pre_obs, pre_action, reward, obs, done, info):
        if self.pz is None:
            self.pz = self.aggregater(pre_obs)
        z = self.aggregater(obs)
        if self.pz == z:
            self.t += 1
        else:
            self.counter_transit += 1
            self.t = 0
        c_potential = self.potential(z)
        # ※浮動小数点flaotは2進数による10進数の近似を行っている
        v = decimal_calc(
            self.gamma * c_potential,
            self.p_potential,
            "-"
        )
        self.pz = z
        self.p_potential = c_potential
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
        return max(decimal_calc(
            self.eta * z,
            self.rho * self.t,
            "-"
        ), 0.0)

    def get_counter_transit(self):
        return self.counter_transit

    def get_current_state(self):
        return self.aggregater.get_current_state()


class NaiveSRS(SubgoalRS):
    def __init__(self, gamma, lr, eta, rho, aggr_id, abstractor):
        super().__init__(gamma, lr, eta, rho, aggr_id, abstractor)

    def potential(self, z):
        if self.t == 0:
            return self.eta
        else:
            return 0
