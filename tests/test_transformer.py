from shaner.aggregater import Discretizer
from shaner.aggregater import Splitter
from shaner.aggregater import FetchPickAndPlaceAchiever,\
    FourroomsAchiever
import gym
import gym_pinball
import gym_fourrooms
import unittest
import numpy as np
import os

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


class TestFourroomsAchiever(unittest.TestCase):
    def testEval(self):
        env = gym.make('ConstFourrooms-v0')
        subgoal_path = os.path.join("tests", "in", "fourrooms_subgoals.csv")
        achiever = FourroomsAchiever(0, 103, subgoal_path)
        obses = [1, 2, 1, 3]
        a_states = [0, 0, 1, 1]
        corrects = [True, False, False, True]
        for obs, a_state, correct in zip(obses, a_states, corrects):
            res = achiever.eval(obs, a_state)
            self.assertEqual(res, correct)        
