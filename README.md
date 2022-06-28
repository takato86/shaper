# shaper
A library for shaping the reward in RL.

## Installation
```
pip install shaper
```

## How to use

```python
import shaper
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
rs = shaper.SarsaRS(gamma, lr, aggregator, vfunc, is_success=us_success)

# How to use in RL loop.
env = gym.create("CartPole-v1")
pre_obs = env.reset()

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    shaping_reward = rs.step(pre_obs, action, reward, obs, done, info)
```

## Implemantation List
|method|implemented|
|---|---|
|sarsa-rs|○|
|subgoal-based reward shaping with static potential|◯|
|subgoal-based reward shaping with dynamic potential(DTA)|◯|

