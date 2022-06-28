import unittest
from shaper.splitter import Splitter
from shaper.value import TableValue
from shaper.aggregator.discretizer import Discretizer
import gym
import gym_pinball

gym_pinball


class TestTableValue(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('PinBall-v0')
        splitter = Splitter(self.env.observation_space.low, self.env.observation_space.high, 2)
        aggregator = Discretizer(splitter)
        params = {"n_states": aggregator.get_n_states()}
        self.v = TableValue(**params)

    def test_update(self):
        v = 1
        z = 0
        self.v.update(z, v)
        self.assertEqual(1, self.v(z))


if __name__ == '__main__':
    unittest.main()
