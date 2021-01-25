import unittest
import shaner
import gym
import gym_pinball
import gym_fourrooms
import numpy as np


class SarsaRSTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("PinBall-v0")
        config = {
            'lr': 0.001,
            'gamma': 0.99,
            'env': self.env,
            'params': {
                'vid': 'table',
                'aggr_id': 'disc',
                'params':{
                    'env': self.env,
                    'clip_range': None,
                    'n': 2
                }
            }
        }
        self.rs = shaner.SarsaRS(**config)

    def test_init_potential(self):
        # obs = [0, 0, 0, 0]
        z = 0
        self.assertEqual(0, self.rs.potential(z))

    def test_init_value(self):
        pre_obs = [0, 0.5, 0, 0]
        obs = [0, 0, 0, 0]
        done = False
        v = self.rs.perform(pre_obs, obs, 0, done)
        self.assertEqual(0, v)

    def test_perform(self):
        pre_obs = [0, 0.7, 0, 0]
        obs = [0, 0, 0, 0]
        reward = 1
        done = False
        self.assertEqual(0, self.rs.vfunc(0))
        self.assertEqual(0, self.rs.vfunc(0))
        v = self.rs.perform(pre_obs, obs, reward, done)
        print(self.rs.pz, self.rs.aggregater(obs, False))
        self.assertEqual(0.001, self.rs.vfunc(self.rs.aggregater(pre_obs, False)))
        self.assertEqual(0, self.rs.vfunc(self.rs.aggregater(obs, False)))
        self.assertEqual(-0.001, v)


class DTATest(unittest.TestCase):
    def setUp(self):
        env_id = "FetchPickAndPlace-v1"
        self.env = gym.make(env_id)
        self.config = {
            'lr': 0.001,
            'gamma': 0.99,
            'env': self.env,
            'params': {
                'vid': 'table',
                'aggr_id': 'dta',
                'params':{
                    'env_id': env_id,
                    '_range': 0.01,
                    'n_obs': 25,
                }
            }
        }
    
    def test_init_potential(self):
        rs = shaner.SarsaRS(**self.config)
        obs = np.full(25, 0)
        z = rs.aggregater(obs, False)
        self.assertEqual(0, rs.potential(z))

    def test_init_value(self):
        rs = shaner.SarsaRS(**self.config)
        pre_obs = np.full(25, 0)
        obs = np.full(25, 0.01)
        done = False
        v = rs.perform(pre_obs, obs, 0, done)
        self.assertEqual(0, v)

    def test_perform(self):
        rs = shaner.SarsaRS(**self.config)
        pre_obs = np.full(25, -1)
        obs = np.full(25, 0)
        reward = 1
        done = False
        pz = 0
        z = 1
        self.assertEqual(0, rs.vfunc(pz))
        self.assertEqual(0, rs.vfunc(z))
        # 抽象状態が切り替わる想定
        v = rs.perform(pre_obs, obs, reward, done)
        self.assertEqual(0.001, rs.vfunc(0))
        self.assertEqual(0, rs.vfunc(z))
        self.assertEqual(-0.001, v)


class TestSubgoalRS(unittest.TestCase):
    def setUp(self):
        env_id = "ConstFourrooms-v0"
        self.env = gym.make(env_id)
        self.config = {
            'lr': 0.001,
            'gamma': 0.99,
            'env': self.env,
            'params': {
                'eta': 1,
                'rho': 0.1,
                'vid': 'table',
                'aggr_id': 'dta',
                'params':{
                    'env_id': env_id,
                    '_range': 0,
                    'n_obs': 103,
                    'subgoal_path': "tests/in/fourrooms_subgoals.csv"
                }
            }
        }

    def test_perform(self):
        rs = shaner.SubgoalRS(**self.config)
        pre_obs = 0
        obs = 1
        reward = 0
        done = False
        v = rs.perform(pre_obs, obs, reward, done)
        self.assertEqual(0.99, v)
        pre_obs = obs
        obs = 2
        v = rs.perform(pre_obs, obs, reward, done)
        # c_potential = 1 - 0.1
        # gamma * c_potential - p_potential = 0.99 * 0.9 - 1 = -0.109
        self.assertEqual(-0.109, v)


if __name__ == '__main__':
    unittest.main()
