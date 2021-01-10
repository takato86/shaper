from shaner.transformer.core import TwoDiscretizer, ThreeDiscretizer
from shaner.transformer.splitter import Splitter
import gym
import gym_pinball
import unittest
import numpy as np

class TestSplitter(unittest.TestCase):
    def testThreeSplit(self):
        obs = [-5, 5, 0, 3, 3.334, -3, -3.334, -1, 1]
        lower = np.array([-5] * len(obs))
        higher = np.array([5] * len(obs))
        splitter = Splitter(lower, higher, 3)
        res = splitter.eval(obs)
        correct = [0, 2, 1, 2, 2, 0, 0, 1, 1]
        self.assertListEqual(res.tolist(), correct)


class TestTwoDiscretizer(unittest.TestCase):
    def setUp(self):
        env = gym.make('PinBall-v0')
        self.transformer = TwoDiscretizer(env)

    def testCall(self):
        obs = [0, 0, 0, 0]
        self.assertEqual(0, self.transformer(obs))
        obs = [0, 0.7, 0, 0]
        self.assertEqual(2, self.transformer(obs))

class TestThreeDiscretizer(unittest.TestCase):
    def setUp(self):
        env = gym.make('PinBall-v0')
        # env = gym.make('FetchPickAndPlace-v1')
        # Memory Error
        self.transformer = ThreeDiscretizer(env)
        obs = [0, 0, 0, 0]
        correct = 27 + 9 + 3 + 1
        self.assertEqual(correct, self.transformer(obs))
    
    def testCall(self):
        pass
