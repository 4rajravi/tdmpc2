import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Adds the parent directory to the search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tdmpc2.evaluation.evaluator import Evaluator
from tdmpc2.agent import TDMPCAgent
from tdmpc2.envs.make_env import make_env

# load checkpoint
checkpoint = torch.load("agent.pt")

agent = TDMPCAgent.load(checkpoint)

env = make_env(cfg.env)

evaluator = Evaluator(
    agent=agent,
    env=env,
    save_dir="test_results",
    episodes=10,
    record_video=True
)

evaluator.run()
