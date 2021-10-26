"""
Implementation of
Laud, Adam Daniel. 2004.
“Theory and Application of Reward Shaping in Reinforcement Learning.”
Champaign, IL, USA: University of Illinois at Urbana-Champaign.
"""

from shaner.aggregater.subgoal_based import Checker
from shaner.utils import decimal_calc


# TODO value_funcを渡せるかどうか？
class SubgoalPulseRS:
    def __init__(self, gamma, value_func, achiever):
        self.gamma = gamma
        # 状態価値を出力する関数
        self.value_func = value_func
        self.checker = Checker(achiever)
        self.reset()

    def reset(self):
        self.checker.reset()
        self.p_potential = 0

    def start(self, obs):
        pass

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
            return self.value_func(obs)
        else:
            return 0
