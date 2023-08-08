import gymnasium as gym
from envs import CUSTOM_ENVS
from config.config import cfg
import argparse
import numpy as np


# Load config options
parser = argparse.ArgumentParser(description="User's arguments from terminal.")
parser.add_argument(
      "--cfg", 
      dest="cfg_file", 
      help="Config file", 
      required=True, 
      type=str)
args = parser.parse_args()
cfg.merge_from_file(args.cfg_file)

print(cfg.TEMPLATE)

env = gym.make(cfg.ENV_NAME, render_mode="human", agent_num=cfg.EMAT.AGENT_NUM)
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    if any(terminated) or truncated:
      observation, info = env.reset()      

env.close()


