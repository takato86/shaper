from shaper.reward import HighReward
import unittest


class TestHighReward(unittest.TestCase):
    def setUp(self):
        gamma = 0.99
        self.r = HighReward(gamma)

    def test_update(self):
        reward = 1
        self.r.update(reward, 1)
        self.assertEqual(0.99, self.r())
        reward = 0
        self.r.update(reward, 1)
        self.assertEqual(0.99, self.r())

    def test_reset(self):
        reward = 1
        self.r.update(reward, 1)
        self.assertEqual(0.99, self.r())
        self.r.reset()
        self.assertEqual(0, self.r())


if __name__ == "__main__":
    unittest.main()
