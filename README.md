# Multiagent-Race
This repo provides environments for MuJoCo multi-agent self-interested competition tasks.
There are three classes of tasks including Race, Grasp, and Screw.
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