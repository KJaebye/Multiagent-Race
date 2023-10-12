# Multiagent-Race
This repo provides environments for MuJoCo multi-agent self-interested competition tasks.
Where the relative information is treated as competitive message that is directly sent to the neural network. Training
under competitive condition can stimulate the potential of robot, thus obtain a higher performance than single agent training.

The reason that competitive learning can facilitate training, is the competitive information between robots naturally 
build a relationship between better actions, dominant position, and higher reward. This relationship helps to distinguish
and understand what is a more accurate state-action-reward pair.

## Usage
Gymnasium has been updated. `/test.py` is a simple example.
```
env = gym.make(cfg.ENV_NAME, render_mode="human", agent_num=cfg.EMAT.AGENT_NUM)
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    if any(terminated) or truncated:
      observation, info = env.reset()

env.close()
```
## Config
The config directory contains a simple-use args and configuration tool. Some env configs are defined in `/config/config.py`