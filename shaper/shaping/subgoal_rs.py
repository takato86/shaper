import logging
from typing import Any, Dict, Optional

import numpy as np
from shaper.aggregator.interface import AbstractAggregator
from shaper.shaping.interface import AbstractShaping
from shaper.utils import decimal_calc


logger = logging.getLogger(__name__)


class SubgoalRS(AbstractShaping):
    @classmethod
    @property
    def is_learn(cls) -> bool:
        return False

    def __init__(self, gamma: float, eta: float, rho: float, aggregator: AbstractAggregator):
        self.gamma = gamma
        self.eta = eta
        self.rho = rho
        self.aggregator = aggregator
        self.t: int = 0
        self.pz: Optional[int] = None
        self.p_potential: float = 0
        self.aggregator.reset()
        self.counter_transit: int = 0

    @property
    def current_state(self) -> Optional[np.ndarray]:
        return None

    def step(self, pre_obs: np.ndarray, action: np.ndarray, reward: float,
             obs: np.ndarray, done: bool, info: Dict[str, Any]) -> float:
        if self.pz is None:
            self.pz = self.aggregator(pre_obs)

        z = self.aggregator(obs)

        if self.pz == z:
            self.t += 1
        else:
            self.counter_transit += 1
            self.t = 0

        c_potential = self.potential(z)
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

    def shape(self, pz: np.ndarray, z: np.ndarray) -> float:
        r = decimal_calc(
            self.gamma * self.potential(z),
            self.potential(pz),
            "-"
        )
        return r

    def reset(self):
        self.t: int = 0
        self.pz: Optional[int] = None
        self.p_potential: float = 0
        self.aggregator.reset()

    def start(self, obs):
        self.counter_transit = 0
        self.pz = self.aggregator(obs)

    def perform(self, pre_obs, obs, reward, done, info=None):
        if self.pz is None:
            self.pz = self.aggregator(pre_obs)
        z = self.aggregator(obs)
        if self.pz == z:
            self.t += 1
        else:
            self.counter_transit += 1
            self.t = 0
        c_potential = self.potential(z)
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

    def potential(self, z):
        return max(decimal_calc(
            self.eta * z,
            self.rho * self.t,
            "-"
        ), 0.0)

    def get_counter_transit(self):
        return self.counter_transit

    def get_current_state(self):
        return self.aggregator.get_current_state()


class NaiveSRS(SubgoalRS):
    @classmethod
    @property
    def is_learn(cls):
        return False

    def __init__(self, gamma, eta, rho, aggr_id, abstractor, is_success):
        super().__init__(gamma, eta, rho, aggr_id, abstractor, is_success)

    def potential(self, z):
        if self.t == 0:
            return self.eta
        else:
            return 0


class LinearNaiveSRS(SubgoalRS):
    @classmethod
    @property
    def is_learn(cls):
        return False

    def __init__(self, gamma, eta, rho, aggr_id, abstractor, is_success):
        super().__init__(gamma, eta, rho, aggr_id, abstractor, is_success)

    def potential(self, z):
        if self.t == 0:
            return self.eta*z
        else:
            return 0