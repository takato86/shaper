"""
Implementation of
Laud, Adam Daniel. 2004.
'Theory and Application of Reward Shaping in Reinforcement Learning.'
Champaign, IL, USA: University of Illinois at Urbana-Champaign.
"""

from shaner.aggregater.subgoal_based import Checker
from shaner.shaping.interface import AbstractShaping
from shaner.utils import decimal_calc


# TODO value_funcを渡せるかどうか？
class SubgoalPulseRS(AbstractShaping):
    is_learn = True

    def __init__(self, gamma, achiever):
        self.gamma = gamma
        # 状態価値を出力する関数
        self.obs_value = 0
        self.checker = Checker(achiever)
        self.reset()

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

    def potential(self, obs):
        if self.checker(obs):
            return self.obs_value
        else:
            return 0
