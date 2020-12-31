import unittest
from shaner.value import TableValue
from shaner.transformer import TwoDiscretizer
import gym
import gym_pinball


class TestTableValue(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('PinBall-v0')
        params = {}
        transformer = TwoDiscretizer(self.env)
        self.v = TableValue(self.env, transformer=transformer,
                            **params)

    def test_update(self):
        v = 1
        obs = [0, 0, 0, 0]
        self.v.update(obs, v)
        self.assertEqual(1, self.v(obs))


if __name__ == '__main__':
    unittest.main()
