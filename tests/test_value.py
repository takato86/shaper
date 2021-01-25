import unittest
from shaner.value import TableValue
from shaner.aggregater import Discretizer
import gym
import gym_pinball


class TestTableValue(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('PinBall-v0')
        aggregater = Discretizer(self.env, 2)
        params = {"n_states": aggregater.get_n_states()}
        self.v = TableValue(self.env, aggregater=aggregater,
                            **params)

    def test_update(self):
        v = 1
        z = 0
        self.v.update(z, v)
        self.assertEqual(1, self.v(z))


if __name__ == '__main__':
    unittest.main()
