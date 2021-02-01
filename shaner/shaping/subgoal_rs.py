from shaner.factory import ValueFactory, AggregaterFactory
from shaner.reward import HighReward
from shaner.utils import decimal_calc

class SubgoalRS:
    def __init__(self, gamma, lr, env, params):
        self.gamma = gamma
        self.lr = lr
        self.eta = params['eta']
        self.rho = params['rho']
        self.aggregater = AggregaterFactory().create(params['aggr_id'],
                                                     params['params'])
        self.reset()

    def reset(self):
        self.t = 0
        self.pz = None
        self.p_potential = 0

    def start(self, obs):
        self.pz = self.aggregater(obs, False)

    def perform(self, pre_obs, obs, reward, done):
        # import pdb; pdb.set_trace()
        if self.p_potential is None:
            self.pz = self.aggregater(pre_obs, False)
        z = self.aggregater(obs, done)
        if self.pz == z:
            self.t += 1
        else:
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

    def potential(self, z):
        return max(decimal_calc(
            self.eta * z,
            self.rho * self.t,
            "-"
        ), 0)


class NaiveSRS(SubgoalRS):
    def __init__(self, gamma, lr, env, params):
        super().__init__(gamma, lr, env, params)
    
    def potential(self, z):
        if self.t == 0:
            return self.eta
        else:
            return 0
