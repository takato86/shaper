from typing import Any, Callable, Dict
import numpy as np
from shaper.aggregator.interface import AbstractAggregator
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

    def __init__(self, gamma: float, lr: float, aggregator: AbstractAggregator, vfunc: AbstractValue,
                 is_success: Callable[[bool, dict[str, Any]], bool]):
        self.gamma = gamma
        self.lr = lr
        self.aggregator = aggregator
        self.vfunc = vfunc
        self.high_reward = HighReward(gamma=gamma)
        self.t = -1  # timesteps during abstract states.
        self.pz = None  # previous abstract state.
        # for analysis varibles
        self.counter_transit = 0
        self.is_success = is_success
        self.is_already_end = False

    @property
    def current_state(self):
        return self.aggregator.get_current_state()

    def shape(self, pz, z) -> float:
        r = decimal_calc(
            self.gamma * self.potential(z),
            self.potential(pz),
            "-"
        )
        return r

    def step(self, pre_obs: np.ndarray, action: np.ndarray, reward: float,
             obs: np.ndarray, done: bool, info: Dict[str, Any]) -> float:
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
            self.pz = self.aggregator(pre_obs)
        z = self.aggregator(obs)
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

        if self.pz != z or (is_end_update and not self.is_already_end):
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
            self.is_already_end = is_end_update

    def start(self, obs):
        self.counter_transit = 0
        self.pz = self.aggregator(obs)

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
            self.pz = self.aggregator(pre_obs)
        z = self.aggregator(obs)
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
        self.is_already_end = False
        self.high_reward.reset()
        self.aggregator.reset()

    def get_counter_transit(self):
        return self.counter_transit
