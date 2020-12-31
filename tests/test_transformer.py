from shaner.transformer.core import TwoDiscretizer
import gym
import gym_pinball
import unittest


class TestTwoDiscretizer(unittest.TestCase):
    def setUp(self):
        env = gym.make('PinBall-v0')
        self.transformer = TwoDiscretizer(env)

    def testCall(self):
        obs = [0, 0, 0, 0]
        self.assertEqual(0, self.transformer(obs))
        obs = [0, 0.7, 0, 0]
        self.assertEqual(2, self.transformer(obs))

