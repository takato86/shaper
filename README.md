# shaper
A library for shaping the reward in RL. This library includes the implementations of the following papers.

* [T. Okudo and S. Yamada, "Subgoal-Based Reward Shaping to Improve Efficiency in Reinforcement Learning," in IEEE Access, vol. 9, pp. 97557-97568, 2021, doi: 10.1109/ACCESS.2021.3090364.](https://ieeexplore.ieee.org/document/9459751)
    * You can use the same method by `DynamicTrajectoryAggregation` and `SubgoalRS`
* [T. Okudo and S. Yamada, "Reward Shaping with Dynamic Trajectory Aggregation," 2021 International Joint Conference on Neural Networks (IJCNN), 2021, pp. 1-9, doi: 10.1109/IJCNN52387.2021.9533401.](https://ieeexplore.ieee.org/document/9533401)
    * You can use the same method by `DynamicTrajectoryAggregation` and `SarsaRS`

## Installation
```
pip install -e .
```

## How to use
Please write like the following script. You can check the examples of domain-specific achiever [here](examples/achievers/achiever.py)

```python
import shaper
from shaper.achiever.interface import AbstractAchiever
from shaper.aggregator.subgoal_based import DynamicTrajectoryAggregation
import gym

# How to create the reward shaping instance.
def is_success(done, info):
    if "is_success" in info:
        return info["is_success"]
    return done

# Achiever is domain-specific. You can see the implementation examples in "examples" directory.
achiever = AbstractAchiever()
aggregator = DynamicTrajectoryAggregation(achiever)
vfunc = aggregator.create_vfunc()
rs = shaper.SarsaRS(gamma, lr, aggregator, vfunc, is_success=is_success)

# How to use in RL loop.
env = gym.create("CartPole-v1")
pre_obs = env.reset()

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    shaping_reward = rs.step(pre_obs, action, reward, obs, done, info)
```

## Aggregator objects
1. DynamicTrajectoryAggregation
2. Discretizer

## Shaping objects
1. SarsaRS
2. SubgoalRS
3. NaiveRS
