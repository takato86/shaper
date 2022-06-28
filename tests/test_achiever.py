
import unittest
import gym
import numpy as np
from examples.achievers.achiever import CrowdSimAchiever, FetchPickAndPlaceAchiever, FourroomsAchiever, PinballAchiever


class TestFetchPickAndPlaceAchiever(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('FetchPickAndPlace-v1')

    def testEval(self):
        achiever = FetchPickAndPlaceAchiever(0.01, np.full((1, 25), 0))
        obs = np.full(25, 0)
        res = achiever.eval(obs, 0)
        correct = True
        self.assertEqual(res, correct)
        obs = np.full(25, 0.5)
        res = achiever.eval(obs, 0)
        correct = False
        self.assertEqual(res, correct)
        obs = np.full(25, 0.01)
        res = achiever.eval(obs, 1)
        correct = False
        self.assertEqual(res, correct)


class TestFourroomsAchiever(unittest.TestCase):
    def testEval(self):
        # env = gym.make('ConstFourrooms-v0')
        # subgoal_path = os.path.join("tests", "in", "fourrooms_subgoals.csv")
        subgoals = np.array([[1], [3]])
        achiever = FourroomsAchiever(subgoals)
        obses = [1, 2, 1, 3]
        a_states = [0, 0, 1, 1]
        corrects = [True, False, False, True]
        for obs, a_state, correct in zip(obses, a_states, corrects):
            res = achiever.eval(obs, a_state)
            self.assertEqual(res, correct)


class TestPinballAchiever(unittest.TestCase):
    def testEval(self):
        subgoals = np.array([[0.5, 0.5, np.nan, np.nan], [0.7, 0.7, np.nan, np.nan]])
        achiever = PinballAchiever(0.04, subgoals)
        obses = [
            [0.5, 0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6, 0.6],
        ]
        obses = np.array(obses)
        a_states = [0, 0]
        corrects = [True, False]
        for obs, a_state, correct in zip(obses, a_states, corrects):
            res = achiever.eval(obs, a_state)
            self.assertEqual(res, correct)


class TestCrowdSimAchiever(unittest.TestCase):
    def setUp(self):
        _range = {
            'dist': 1,
            'angle': 10
        }
        self.achiever = CrowdSimAchiever(_range)

    def testCalcAngle(self):
        vec1 = [1, 0]
        vec2 = [0, 1]
        predict = self.achiever._CrowdSimAchiever__calc_angle(vec1, vec2)
        correct = 90
        self.assertEqual(predict, correct)

    def testCalcDist(self):
        vec1 = [1, 0]
        vec2 = [0, 1]
        predict = self.achiever._CrowdSimAchiever__calc_dist(vec1, vec2)
        correct = np.sqrt(2)
        self.assertEqual(predict, correct)

    def testInRange(self):
        basis = {
            "dist": 3,
            "angle": 90
        }
        key = "dist"
        target = 4
        predict = self.achiever._CrowdSimAchiever__in_range(basis, target, key)
        correct = True
        self.assertEqual(predict, correct)
        target = 4.1
        predict = self.achiever._CrowdSimAchiever__in_range(basis, target, key)
        correct = False
        self.assertEqual(predict, correct)
        target = 2
        predict = self.achiever._CrowdSimAchiever__in_range(basis, target, key)
        correct = True
        self.assertEqual(predict, correct)
        target = 1.9
        predict = self.achiever._CrowdSimAchiever__in_range(basis, target, key)
        correct = False
        self.assertEqual(predict, correct)
        key = "angle"
        target = 100
        predict = self.achiever._CrowdSimAchiever__in_range(basis, target, key)
        correct = True
        self.assertEqual(predict, correct)
        target = 101
        predict = self.achiever._CrowdSimAchiever__in_range(basis, target, key)
        correct = False
        self.assertEqual(predict, correct)
        target = 80
        predict = self.achiever._CrowdSimAchiever__in_range(basis, target, key)
        correct = True
        self.assertEqual(predict, correct)

    def testEval(self):
        human_vel = [1, 1]
        human_pos = [2, 2]
        robot_pos = [-2, -2]
        h_r_rel_pos = [
            robot_pos[0] - human_pos[0],
            robot_pos[1] - human_pos[1]
        ]
        predict = self.achiever._CrowdSimAchiever__calc_angle(human_vel, h_r_rel_pos)
        correct = 180
        self.assertEqual(correct, predict)


if __name__ == "__main__":
    unittest.main()
