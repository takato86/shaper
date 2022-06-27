from typing import Callable, Dict
import numpy as np
from shaper.value import TableValue
from shaper.aggregater.interface import AbstractAggregater
from shaper.reward import HighReward
from shaper.utils import decimal_calc
from shaper.shaping.interface import AbstractShaping
import warnings

from shaper.value import AbstractValue


class SarsaRS(AbstractShaping):
    @classmethod
    @property
    def is_learn(cls):
        return True

    def __init__(self, gamma: float, lr: float, aggregater: AbstractAggregater, vfunc: AbstractValue,
                 is_success: Callable[[bool, dict[str, any]], bool]):
        self.gamma = gamma
        self.lr = lr
        self.aggregater = aggregater
        self.vfunc = vfunc
        self.high_reward = HighReward(gamma=gamma)
        self.t = -1  # timesteps during abstract states.
        self.pz = None  # previous abstract state.
        # for analysis varibles
        self.counter_transit = 0
        self.is_success = is_success
        self.is_pre_success = False

    @property
    def current_state(self):
        return self.aggregater.get_current_state()

    def shape(self, pz, z) -> float:
        r = decimal_calc(
            self.gamma * self.potential(z),
            self.potential(pz),
            "-"
        )
        return r

    def step(self, pre_obs: np.ndarray, action: np.ndarray, reward: float,
             obs: np.ndarray, done: bool, info: Dict[str, any]) -> float:
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
        self.__train(z, reward, done, info)
        self.pz = z
        if done:
            self.reset()
        return v

    def __train(self, z, reward, done, info):
        self.t += 1
        is_end_update = self.is_success(done, info)
        self.high_reward.update(reward, self.t)

        if self.pz != z or (is_end_update and not self.is_pre_success):
            assert self.t >= 0

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

    def start(self, obs):
        self.counter_transit = 0
        self.pz = self.aggregater(obs)

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

    def potential(self, z):
        return self.vfunc(z)

    def reset(self):
        self.t = 0
        self.pz = None
        self.high_reward.reset()
        self.aggregater.reset()

    def get_counter_transit(self):
        return self.counter_transit


class OffsetSarsaRS(SarsaRS):
    def __init__(self, gamma: float, lr: float, aggregater: AbstractAggregater, vfunc: TableValue, is_success: bool):
        super().__init__(gamma, lr, aggregater, vfunc, is_success)

    def potential(self, z: any) -> float:
        """必ず正のポテンシャルが生成されるようにオフセット."""
        potential = super().potential(z)
        min_potential = self.vfunc.get_min_value()
        return potential - min_potential
