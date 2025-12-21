#!/usr/bin/env python3
"""
WSL2 Multicore Training Script - 高效多核訓練腳本 (Linux/WSL2 專用)

針對 WSL2/Linux 優化:
- 使用 fork 啟動方法 (比 Windows spawn 快 3-4 倍)
- SubprocVecEnv 實現真正的多核並行
- 支援 CPU 親和性設定
- 載入 V3 特徵 (含 Up_Stock/Down_Stock，無 Volume)

用法:
    # 訓練 Buy Agent (8 核心，100萬步)
    python scripts/train_multicore_wsl.py --agent buy --timesteps 1000000 --n-envs 8
    
    # 訓練 Sell Agent (16 核心，1500萬步)
    python scripts/train_multicore_wsl.py --agent sell --timesteps 15000000 --n-envs 16
    
    # 從 checkpoint 恢復訓練
    python scripts/train_multicore_wsl.py --agent buy --timesteps 12000000 --n-envs 8 --resume
"""

import os
import sys
import argparse
import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

# 設定環境變數 (必須在 import torch 之前)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 添加專案根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from loguru import logger

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    CallbackList,
    BaseCallback
)


class TensorBoardCallback(BaseCallback):
    """TensorBoard 即時監控 Callback"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            if ep_rewards:
                self.logger.record('custom/mean_reward', np.mean(ep_rewards))


def load_features_v3(features_path: str) -> dict:
    """載入 V3 特徵 (含市場廣度，無 Volume)"""
    logger.info(f"載入 V3 特徵: {features_path}")
    
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    logger.info(f"載入 {len(features)} 支股票特徵")
    return features


def prepare_buy_training_data(features: dict, success_threshold: float = 0.10):
    """
    準備 Buy Agent 訓練資料
    
    過濾 Donchian Channel 突破訊號，計算是否成功 (報酬 >= 10%)
    """
    logger.info("準備 Buy Agent 訓練資料...")
    
    all_signals = []
    
    for symbol, df in features.items():
        if df is None or len(df) < 252:
            continue
        
        # 確保有必要欄位
        if 'Donchian_Upper' not in df.columns or 'High' not in df.columns:
            continue
        
        # 找出 Donchian 突破點
        df = df.copy()
        df['is_breakout'] = df['High'] > df['Donchian_Upper'].shift(1)
        
        breakout_df = df[df['is_breakout'] == True].copy()
        
        if len(breakout_df) == 0:
            continue
        
        # 計算未來 120 天最大報酬
        for idx in breakout_df.index:
            try:
                loc = df.index.get_loc(idx)
                if loc + 120 >= len(df):
                    continue
                
                buy_price = df.iloc[loc]['Close']
                future_prices = df.iloc[loc+1:loc+121]['Close']
                max_return = (future_prices.max() - buy_price) / buy_price
                
                breakout_df.loc[idx, 'actual_return'] = max_return
                breakout_df.loc[idx, 'is_successful'] = max_return >= success_threshold
            except:
                continue
        
        # 過濾有效樣本
        valid = breakout_df.dropna(subset=['actual_return'])
        if len(valid) > 0:
            valid['symbol'] = symbol
            all_signals.append(valid)
    
    if len(all_signals) == 0:
        raise ValueError("沒有有效的訓練資料")
    
    combined = pd.concat(all_signals, ignore_index=False)
    
    # 資料平衡 (1:1)
    successful = combined[combined['is_successful'] == True]
    failed = combined[combined['is_successful'] == False]
    min_count = min(len(successful), len(failed))
    
    if min_count > 0:
        balanced = pd.concat([
            successful.sample(n=min_count, random_state=42),
            failed.sample(n=min_count, random_state=42)
        ]).sample(frac=1, random_state=42)
    else:
        balanced = combined
    
    logger.info(f"訓練資料: {len(balanced)} 樣本 (成功:{min_count}, 失敗:{min_count})")
    return balanced


def prepare_sell_training_data(features: dict, success_threshold: float = 0.10, max_holding_days: int = 120):
    """
    準備 Sell Agent 訓練資料
    
    只使用成功交易 (最大報酬 >= 10%) 的持有期間資料
    """
    logger.info("準備 Sell Agent 訓練資料...")
    
    all_episodes = []
    
    for symbol, df in features.items():
        if df is None or len(df) < 252 + max_holding_days:
            continue
        
        if 'Donchian_Upper' not in df.columns:
            continue
        
        df = df.copy()
        df['is_breakout'] = df['High'] > df['Donchian_Upper'].shift(1)
        
        breakout_indices = df[df['is_breakout'] == True].index.tolist()
        
        for buy_idx in breakout_indices:
            try:
                loc = df.index.get_loc(buy_idx)
                if loc + max_holding_days >= len(df):
                    continue
                
                buy_price = df.iloc[loc]['Close']
                holding_period = df.iloc[loc:loc+max_holding_days+1].copy()
                
                # 計算持有期間報酬
                holding_period['sell_return'] = holding_period['Close'] / buy_price
                max_return = holding_period['sell_return'].max() - 1
                
                # 只保留成功交易
                if max_return >= success_threshold:
                    holding_period['buy_price'] = buy_price
                    holding_period['symbol'] = symbol
                    holding_period['episode_id'] = f"{symbol}_{buy_idx}"
                    all_episodes.append(holding_period)
            except:
                continue
    
    if len(all_episodes) == 0:
        raise ValueError("沒有有效的 Sell 訓練資料")
    
    combined = pd.concat(all_episodes, ignore_index=False)
    logger.info(f"Sell 訓練資料: {len(all_episodes)} 個 episode, {len(combined)} 個樣本")
    return combined


def make_buy_env(training_data: pd.DataFrame, feature_cols: list, env_id: int = 0):
    """創建 BuyEnv 工廠函數 (用於 SubprocVecEnv)"""
    from src.environments.buy_env import BuyEnv
    
    # 只保留需要的欄位 (BuyEnv 會自動從 DataFrame 推導 feature_cols)
    required_cols = feature_cols + ['actual_return', 'is_successful']
    available_cols = [c for c in required_cols if c in training_data.columns]
    filtered_data = training_data[available_cols].copy()
    
    def _init():
        env = BuyEnv(filtered_data, config={
            'balance_data': False  # 已經平衡過了
        })
        return env
    
    return _init


def make_sell_env(training_data: pd.DataFrame, feature_cols: list, env_id: int = 0):
    """創建 SellEnv 工廠函數 (用於 SubprocVecEnv)"""
    from src.environments.sell_env import SellEnv
    
    # 只保留需要的欄位
    required_cols = feature_cols + ['Close', 'Open', 'buy_price', 'symbol', 'episode_id', 'sell_return']
    available_cols = [c for c in required_cols if c in training_data.columns]
    filtered_data = training_data[available_cols].copy()
    
    def _init():
        env = SellEnv(filtered_data, config={
            'feature_cols': feature_cols,
            'max_holding_days': 120
        })
        return env
    
    return _init


def get_feature_cols(sample_df: pd.DataFrame) -> list:
    """取得用於模型的特徵欄位 (正規化後的 69 維)"""
    
    # 正規化後的欄位
    norm_cols = [c for c in sample_df.columns if '_norm' in c]
    
    # 不需正規化但要包含的欄位
    extra_cols = [
        'Return', 'Index_Return',
        'SuperTrend_14', 'SuperTrend_21',
        'Up_Stock', 'Down_Stock',
        'RS_Rate_5', 'RS_Rate_10', 'RS_Rate_20', 'RS_Rate_40',
        'RS_Momentum', 'RS_Trend'
    ]
    extra_cols = [c for c in extra_cols if c in sample_df.columns]
    
    feature_cols = norm_cols + extra_cols
    
    # 確保沒有 Volume
    feature_cols = [c for c in feature_cols if 'Volume' not in c]
    
    logger.info(f"特徵欄位數: {len(feature_cols)}")
    return feature_cols


def train_agent(args):
    """主訓練函數"""
    
    logger.info("=" * 70)
    logger.info(f"WSL2 多核訓練 - {args.agent.upper()} Agent")
    logger.info(f"核心數: {args.n_envs}, 總步數: {args.timesteps:,}")
    logger.info("=" * 70)
    
    # 設定路徑
    features_path = Path(args.features_path)
    model_dir = Path(f"models/{args.agent}_agent")
    checkpoint_dir = Path(f"models/checkpoints/{args.agent}_agent")
    log_dir = Path(f"logs/training/{args.agent}_agent")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入特徵
    features = load_features_v3(str(features_path))
    
    # 取得特徵欄位
    sample_symbol = list(features.keys())[0]
    feature_cols = get_feature_cols(features[sample_symbol])
    
    # 準備訓練資料
    if args.agent == 'buy':
        training_data = prepare_buy_training_data(features)
        env_factory = lambda i: make_buy_env(training_data, feature_cols, i)
    else:
        training_data = prepare_sell_training_data(features)
        env_factory = lambda i: make_sell_env(training_data, feature_cols, i)
    
    # 創建多核環境 (使用 fork)
    logger.info(f"創建 {args.n_envs} 個並行環境 (fork 模式)...")
    
    env_fns = [env_factory(i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns, start_method='fork')
    env = VecMonitor(env, str(log_dir / "monitor.csv"))
    
    logger.info(f"環境創建完成 - 觀察空間: {env.observation_space.shape}")
    
    # 創建或載入模型
    if args.resume:
        # 尋找最新 checkpoint
        checkpoints = list(checkpoint_dir.glob(f"{args.agent}_agent_*_steps.zip"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[-2]))
            logger.info(f"從 checkpoint 恢復: {latest}")
            model = PPO.load(str(latest), env=env)
            resumed_steps = int(latest.stem.split('_')[-2])
        else:
            logger.warning("找不到 checkpoint，從頭開始訓練")
            args.resume = False
    
    if not args.resume:
        resumed_steps = 0
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
    
    # 設定 Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=max(10000 // args.n_envs, 1000),
            save_path=str(checkpoint_dir),
            name_prefix=f'{args.agent}_agent',
            save_replay_buffer=False,
            save_vecnormalize=True
        ),
        TensorBoardCallback()
    ]
    
    # 計算剩餘步數
    remaining_steps = args.timesteps - resumed_steps
    if remaining_steps <= 0:
        logger.info(f"已完成 {resumed_steps:,} 步，達到目標 {args.timesteps:,}")
        return
    
    logger.info(f"開始訓練: {remaining_steps:,} 步 (已完成: {resumed_steps:,})")
    logger.info(f"TensorBoard: tensorboard --logdir={log_dir}")
    
    # 訓練
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=remaining_steps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=False,
        tb_log_name="PPO"
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    fps = remaining_steps / elapsed
    
    # 儲存最終模型
    final_path = model_dir / "final_model.zip"
    model.save(str(final_path))
    
    logger.info("=" * 70)
    logger.info("訓練完成!")
    logger.info(f"  總步數: {args.timesteps:,}")
    logger.info(f"  耗時: {elapsed/3600:.2f} 小時")
    logger.info(f"  FPS: {fps:,.0f}")
    logger.info(f"  模型: {final_path}")
    logger.info("=" * 70)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='WSL2 多核訓練腳本')
    parser.add_argument('--agent', type=str, required=True, choices=['buy', 'sell'],
                        help='訓練的 Agent 類型 (buy/sell)')
    parser.add_argument('--timesteps', type=int, default=12000000,
                        help='總訓練步數 (預設 12M)')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='並行環境數 (預設: CPU 核心數 - 2)')
    parser.add_argument('--features-path', type=str, 
                        default='data/processed/features_v3.pkl',
                        help='V3 特徵檔案路徑')
    parser.add_argument('--resume', action='store_true',
                        help='從最新 checkpoint 恢復訓練')
    
    args = parser.parse_args()
    
    # 自動設定核心數
    if args.n_envs is None:
        args.n_envs = max(mp.cpu_count() - 2, 4)
    
    logger.info(f"偵測到 {mp.cpu_count()} 個 CPU 核心，使用 {args.n_envs} 個環境")
    
    train_agent(args)


if __name__ == '__main__':
    # Linux/WSL2 使用 fork (最快)
    mp.set_start_method('fork', force=True)
    main()
