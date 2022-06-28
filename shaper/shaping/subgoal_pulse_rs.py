"""
Implementation of
Laud, Adam Daniel. 2004.
'Theory and Application of Reward Shaping in Reinforcement Learning.'
Champaign, IL, USA: University of Illinois at Urbana-Champaign.
"""

from shaper.aggregator.subgoal_based import Checker
from shaper.shaping.interface import AbstractShaping
from shaper.utils import decimal_calc


# TODO give the value function
class SubgoalPulseRS(AbstractShaping):
    @classmethod
    @property
    def is_learn(cls):
        return True

    def __init__(self, gamma, achiever, is_success):
        self.gamma = gamma
        self.obs_value = 0
        self.checker = Checker(achiever)
        self.reset()

    @property
    def current_state(self):
        return None

    def step(self, pre_obs, pre_action, reward, obs, done, info):
        p_potential = self.potential(pre_obs)
        c_potential = self.potential(obs)
        v = decimal_calc(
            self.gamma * c_potential,
            p_potential,
            "-"
        )

        if done:
            self.reset()

        return v

    def shape(self, aobs, aobs2):
        p_potential = self.potential(aobs)
        c_potential = self.potential(aobs2)
        v = decimal_calc(
            self.gamma * c_potential,
            p_potential,
            "-"
        )
        return v

    def reset(self):
        self.checker.reset()
        self.p_potential = 0

    def start(self, obs):
        pass

    def set_value(self, value):
        self.obs_value = value

    def perform(self, pre_obs, obs, reward, done, info=None):
        c_potential = self.potential(obs)
        v = decimal_calc(
            self.gamma * c_potential,
            self.p_potential,
            "-"
        )
        self.p_potential = c_potential

        if done:
            self.reset()

        return v

    def potential(self, obs):
        if self.checker(obs):
            return self.obs_value
        else:
            return 0
