
import sys
sys.path.append('d:/000-github-repositories/ptrl-v01')
try:
    import stable_baselines3
    print(f"Stable Baselines3 Version: {stable_baselines3.__version__}")
except ImportError:
    print("SB3 not installed")

try:
    import gymnasium
    print(f"Gymnasium Version: {gymnasium.__version__}")
except ImportError:
    print("Gymnasium not installed")

try:
    import gym
    print(f"Gym Version: {gym.__version__}")
except ImportError:
    print("Gym (old) not installed")

import pandas as pd
from src.environments.buy_env import BuyEnv
env = BuyEnv(pd.DataFrame(), {'use_shared_memory': False})
print(f"BuyEnv Base Classes: {BuyEnv.__mro__}")
