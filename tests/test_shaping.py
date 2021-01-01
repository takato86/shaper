import unittest
import shaner
import gym
import gym_pinball


class SarsaRSTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("PinBall-v0")
        config = {
            'lr': 0.001,
            'gamma': 0.99,
            'env': self.env,
            'params': {
                'vid': 'table',
                'clip_range': None
            }
        }
        self.rs = shaner.SarsaRS(**config)

    def test_init_potential(self):
        obs = [0, 0, 0, 0]
        self.assertEqual(0, self.rs.potential(obs))

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
        self.assertEqual(0, self.rs.vfunc(pre_obs))
        self.assertEqual(0, self.rs.vfunc(obs))
        v = self.rs.perform(pre_obs, obs, reward, done)
        self.assertEqual(0.001, self.rs.vfunc(pre_obs))
        self.assertEqual(0, self.rs.vfunc(obs))
        self.assertEqual(-0.001, v)


if __name__ == '__main__':
    unittest.main()
