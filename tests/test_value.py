import unittest
from shaner.value import TableValue
from shaner.aggregater import Discretizer
import gym
import gym_pinball


class TestTableValue(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('PinBall-v0')
        params = {}
        aggregater = Discretizer(self.env, 2)
        self.v = TableValue(self.env, aggregater=aggregater,
                            **params)

    def test_update(self):
        v = 1
        obs = [0, 0, 0, 0]
        self.v.update(obs, v)
        self.assertEqual(1, self.v(obs))


if __name__ == '__main__':
    unittest.main()
