from shaner.factory import ValueFactory, AggregaterFactory
from shaner.reward import HighReward


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
        v = self.gamma * c_potential - self.p_potential
        self.pz = z
        self.p_potential = c_potential
        if done:
            self.reset()
        return v

    def potential(self, z):
        return self.eta * z - self.rho * self.t
