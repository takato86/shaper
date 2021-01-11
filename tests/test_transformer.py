from shaner.aggregater import Discretizer
from shaner.aggregater import Splitter
from shaner.aggregater import FetchPickAndPlaceAchiever
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


class TestDiscretizer(unittest.TestCase):
    def setUp(self):
        env = gym.make('PinBall-v0')
        self.aggregater = Discretizer(env, 2)

    def testTwoAggregate(self):
        obs = [0, 0, 0, 0]
        done = False
        self.assertEqual(12, self.aggregater(obs, done))
        obs = [0, 0.7, 0, 0]
        self.assertEqual(14, self.aggregater(obs, done))


class TestFetchPickAndPlaceAchiever(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('FetchPickAndPlace-v1')
        self.achiever = FetchPickAndPlaceAchiever(0.01, 25)

    def testEval(self):
        obs = np.full(25, 0)
        res = self.achiever.eval(obs, 0)
        correct = True
        self.assertEqual(res, correct)
        obs = np.full(25, 0.5)
        res = self.achiever.eval(obs, 0)
        correct = False
        self.assertEqual(res, correct)
        obs = np.full(25, 0.01)
        res = self.achiever.eval(obs, 1)
        correct = True
        self.assertEqual(res, correct)

