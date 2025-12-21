
import stable_baselines3
import gymnasium
import gym
import sys

print(f"Stable Baselines3 Version: {stable_baselines3.__version__}")
print(f"Gymnasium Version: {gymnasium.__version__}")
try:
    print(f"Gym Version: {gym.__version__}")
except:
    print("Gym not installed")

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from src.environments.buy_env import BuyEnv
import pandas as pd

print("Checking SB3 Env compatibility...")
try:
    env = BuyEnv(pd.DataFrame(), {'use_shared_memory': False}) # Mock env
    print(f"BuyEnv type: {type(env)}")
    print(f"Is instance of gymnasium.Env: {isinstance(env, gymnasium.Env)}")
    print(f"Is instance of gym.Env: {isinstance(env, gym.Env)}")
    
    # Check what SB3 expects
    # SB3 2.0+ uses gymnasium. SB3 < 2.0 uses gym.
    # We can check by creating a DummyVecEnv and seeing if it works
    vec_env = DummyVecEnv([lambda: env])
    print("DummyVecEnv created successfully.")
    obs = vec_env.reset()
    print("DummyVecEnv reset successfully.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
