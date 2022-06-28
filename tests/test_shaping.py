import unittest
from examples.achievers.achiever import FourroomsAchiever, FetchPickAndPlaceAchiever
import shaper
import gym
import gym_pinball
import gym_fourrooms
import numpy as np
from shaper.aggregator.discretizer import Discretizer

from shaper.aggregator.subgoal_based import DynamicTrajectoryAggregation
from shaper.splitter import Splitter
from shaper.value import TableValue


gym_pinball
gym_fourrooms


class SarsaRSTest(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("PinBall-v0")
        splitter = Splitter(
            self.env.observation_space.low, self.env.observation_space.high, 2
        )
        aggregator = Discretizer(
            splitter
        )
        vfunc = TableValue(aggregator.get_n_states())
        self.rs = shaper.SarsaRS(lr=0.001, gamma=0.99, aggregator=aggregator, vfunc=vfunc, is_success=lambda x, y: x)

    def test_init_potential(self):
        # obs = [0, 0, 0, 0]
        z = 0
        self.assertEqual(0, self.rs.potential(z))

    def test_init_value(self):
        pre_obs = [0, 0.5, 0, 0]
        obs = [0, 0, 0, 0]
        action = [0, 0]
        done, reward, info = False, 0, {}
        v = self.rs.step(pre_obs, action, reward, obs, done, info)
        self.assertEqual(0, v)

    def test_perform(self):
        pre_obs = [0, 0.7, 0, 0]
        obs = [0, 0, -0.7, 0]
        reward = 1
        done = False
        action, info = [0, 0], {}
        self.assertEqual(0, self.rs.vfunc(0))
        self.assertEqual(0, self.rs.vfunc(0))
        v = self.rs.step(pre_obs, action, reward, obs, done, info)
        self.assertEqual(0.001, self.rs.vfunc(self.rs.aggregator(pre_obs)))
        self.assertEqual(0, self.rs.vfunc(self.rs.aggregator(obs)))
        # pre_potentialは前試行の結果を使うので、0
        self.assertEqual(0, v)


class DTATest(unittest.TestCase):
    def setUp(self):
        env_id = "FetchPickAndPlace-v1"
        self.env = gym.make(env_id)
        subgoal1 = np.full(28, np.nan)
        subgoal1[6:9] = [0, 0, 0]
        subgoal2 = np.full(28, np.nan)
        subgoal2[6:11] = [0, 0, 0, 0.02, 0.02]
        subgs = [subgoal1, subgoal2]
        achiever = FetchPickAndPlaceAchiever(0.01, subgs)
        aggregator = DynamicTrajectoryAggregation(achiever)
        vfunc = TableValue(3)
        self.rs = shaper.SarsaRS(gamma=0.99, lr=0.001, aggregator=aggregator, vfunc=vfunc, is_success=lambda x, y: x)

    def test_init_potential(self):
        obs = np.full(25, 0)
        z = self.rs.aggregator(obs)
        self.assertEqual(0, self.rs.potential(z))

    def test_init_value(self):
        pre_obs = np.full(25, 0)
        obs = np.full(25, 0.01)
        action, reward, done, info = self.env.action_space.sample(), 0, False, {}
        v = self.rs.step(pre_obs, action, reward, obs, done, info)
        self.assertEqual(0, v)

    def test_perform(self):
        pre_obs = np.full(25, -1)
        obs = np.full(25, 0)
        action, reward, done, info = self.env.action_space.sample(), 1, False, {}
        pz = 0
        z = 1
        self.assertEqual(0, self.rs.vfunc(pz))
        self.assertEqual(0, self.rs.vfunc(z))
        # 抽象状態が切り替わる想定
        v = self.rs.step(pre_obs, action, reward, obs, done, info)
        self.assertEqual(0.001, self.rs.vfunc(0))
        self.assertEqual(0, self.rs.vfunc(z))
        self.assertEqual(0, v)


class TestSubgoalRS(unittest.TestCase):
    def setUp(self):
        env_id = "ConstFourrooms-v0"
        self.env = gym.make(env_id)
        achiever = FourroomsAchiever([[1], [3]])
        aggregator = DynamicTrajectoryAggregation(achiever)
        self.rs = shaper.SubgoalRS(gamma=0.99, eta=1, rho=0.1, aggregator=aggregator)

    def test_perform(self):
        pre_obs, obs, action, reward, done, info = 0, 1, 0, 0, False, {}
        v = self.rs.step(pre_obs, action, reward, obs, done, info)
        self.assertEqual(0.99, v)
        pre_obs = obs
        obs = 2
        v = self.rs.step(pre_obs, action, reward, obs, done, info)
        # c_potential = 1 - 0.1
        # gamma * c_potential - p_potential = 0.99 * 0.9 - 1 = -0.109
        self.assertEqual(-0.109, v)


if __name__ == '__main__':
    unittest.main()
