#!/usr/bin/env python3
"""
Step 2: WSL2 輕量多核訓練 (使用預處理資料)

必須先執行 prepare_training_data.py 產生:
- data/processed/buy_training_data.pkl
- data/processed/sell_training_data.pkl

用法:
    python3 scripts/train_wsl_v2.py --agent buy --timesteps 12000000
    python3 scripts/train_wsl_v2.py --agent sell --timesteps 12000000
"""

import os
import sys
import argparse
import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from loguru import logger

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback


class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            if ep_rewards:
                self.logger.record('custom/mean_reward', np.mean(ep_rewards))


def make_buy_env(training_data: pd.DataFrame, env_id: int = 0):
    """創建 BuyEnv (資料已經過濾好欄位)"""
    from src.environments.buy_env import BuyEnv
    
    def _init():
        return BuyEnv(training_data.copy(), config={'balance_data': False})
    
    return _init


def make_sell_env(training_data: pd.DataFrame, feature_cols: list, env_id: int = 0):
    """創建 SellEnv"""
    from src.environments.sell_env import SellEnv
    
    def _init():
        return SellEnv(training_data.copy(), config={
            'feature_cols': feature_cols,
            'max_holding_days': 120
        })
    
    return _init


def train_agent(args):
    logger.info("=" * 70)
    logger.info(f"WSL2 輕量多核訓練 - {args.agent.upper()} Agent")
    logger.info(f"核心數: {args.n_envs}, 總步數: {args.timesteps:,}")
    logger.info("=" * 70)
    
    # 設定路徑
    data_path = Path(f"data/processed/{args.agent}_training_data.pkl")
    model_dir = Path(f"models/{args.agent}_agent")
    checkpoint_dir = Path(f"models/checkpoints/{args.agent}_agent")
    log_dir = Path(f"logs/training/{args.agent}_agent")
    
    for d in [model_dir, checkpoint_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 載入預處理資料
    if not data_path.exists():
        logger.error(f"找不到預處理資料: {data_path}")
        logger.error("請先執行: python3 scripts/prepare_training_data.py")
        return
    
    logger.info(f"載入預處理資料: {data_path}")
    with open(data_path, 'rb') as f:
        package = pickle.load(f)
    
    training_data = package['data']
    feature_cols = package['feature_cols']
    
    logger.info(f"訓練資料: {len(training_data)} 樣本, {len(feature_cols)} 特徵")
    
    # 創建環境
    logger.info(f"創建 {args.n_envs} 個並行環境 (fork 模式)...")
    
    if args.agent == 'buy':
        env_fns = [make_buy_env(training_data, i) for i in range(args.n_envs)]
    else:
        env_fns = [make_sell_env(training_data, feature_cols, i) for i in range(args.n_envs)]
    
    env = SubprocVecEnv(env_fns, start_method='fork')
    env = VecMonitor(env, str(log_dir / "monitor.csv"))
    
    logger.info(f"環境創建完成 - 觀察空間: {env.observation_space.shape}")
    
    # 創建模型
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={'net_arch': dict(pi=[40], vf=[40])},
        tensorboard_log=str(log_dir),
        verbose=1
    )
    
    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=max(10000 // args.n_envs, 1000),
            save_path=str(checkpoint_dir),
            name_prefix=f'{args.agent}_agent'
        ),
        TensorBoardCallback()
    ]
    
    logger.info(f"開始訓練: {args.timesteps:,} 步")
    logger.info(f"TensorBoard: tensorboard --logdir={log_dir}")
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        tb_log_name="PPO"
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    fps = args.timesteps / elapsed
    
    # 儲存
    final_path = model_dir / "final_model.zip"
    model.save(str(final_path))
    
    logger.info("=" * 70)
    logger.info("訓練完成!")
    logger.info(f"  耗時: {elapsed/60:.1f} 分鐘")
    logger.info(f"  FPS: {fps:,.0f}")
    logger.info(f"  模型: {final_path}")
    logger.info("=" * 70)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='WSL2 輕量多核訓練')
    parser.add_argument('--agent', type=str, required=True, choices=['buy', 'sell'])
    parser.add_argument('--timesteps', type=int, default=12000000)
    parser.add_argument('--n-envs', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.n_envs is None:
        args.n_envs = max(mp.cpu_count() - 2, 4)
    
    logger.info(f"CPU 核心: {mp.cpu_count()}, 使用: {args.n_envs}")
    
    train_agent(args)


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
