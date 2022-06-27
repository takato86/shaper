from shaper.aggregater.discretizer import Discretizer
from shaper.splitter import Splitter, NSplitter
from shaper.utils import n_ary2decimal
import gym
import gym_pinball
import unittest
import numpy as np

gym_pinball


class TestSplitter(unittest.TestCase):
    def testThreeSplit(self):
        obs = [-5, 5, 0, 3, 3.334, -3, -3.334, -1, 1]
        lower = np.array([-5] * len(obs))
        higher = np.array([5] * len(obs))
        k = 3
        splitter = Splitter(lower, higher, k)
        res = splitter.eval(obs)
        correct = n_ary2decimal([0, 2, 1, 2, 2, 0, 0, 1, 1], k)
        self.assertEqual(res, correct)


class TestNSplitter(unittest.TestCase):
    def testThreeSplit(self):
        obs = [-5, 5, 0, 3, 3.334, -3, -3.334, -1, 1]
        lower = np.array([-5] * len(obs))
        higher = np.array([5] * len(obs))
        k = 3
        splitter = NSplitter(lower, higher, k)
        res = splitter.eval(obs)
        correct = 2
        self.assertEqual(res, correct)

        obs = [-5, -5, -2, -3, -3.334, -3, -3.334, -1, -1]
        res = splitter.eval(obs)
        correct = 0
        self.assertEqual(res, correct)

        obs = [5, 5, 2, 3, 3.334, 3, 3.334, 1, 1]
        res = splitter.eval(obs)
        correct = 1
        self.assertEqual(res, correct)


class TestDiscretizer(unittest.TestCase):
    def setUp(self):
        env = gym.make('PinBall-v0')
        splitter = Splitter(env.observation_space.low, env.observation_space.high, 2)
        self.aggregater = Discretizer(splitter)

    def testTwoAggregate(self):
        obs = [0, 0, 0, 0]
        self.assertEqual(12, self.aggregater(obs))
        obs = [0, 0.7, 0, 0]
        self.assertEqual(14, self.aggregater(obs))


if __name__ == "__main__":
    unittest.main()
