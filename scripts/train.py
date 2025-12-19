"""
Pro Trader RL 訓練腳本

根據論文流程訓練 Buy Agent 和 Sell Agent:
1. 載入並處理 S&P 500 資料
2. 計算 69 個特徵並正規化
3. 產生 Donchian Channel 買入訊號
4. 訓練 Buy Agent (過濾買入訊號)
5. 訓練 Sell Agent (決定賣出時機)
6. 儲存模型

使用方式:
    python scripts/train.py --config config/default_config.yaml
    python scripts/train.py --buy-only
    python scripts/train.py --sell-only
"""

import os
import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# 加入專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.data import DataLoader, FeatureCalculator, DataNormalizer
from src.environments import BuyEnv, SellEnv
from src.agents import BuyAgent, SellAgent
from src.rules import DonchianChannel, StopLossRule


def load_config(config_path: str) -> dict:
    """載入設定檔"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_buy_data(config: dict) -> tuple:
    """
    準備 Buy Agent 訓練資料
    
    Returns:
        train_df: 訓練資料 (< 2017-10-16)
        eval_df: 評估資料 (>= 2017-10-16)
        index_data: 指數資料
    """
    logger.info("開始準備 Buy Agent 訓練資料...")
    data_config = config.get('data', {})
    loader = DataLoader(data_config)
    feature_calc = FeatureCalculator(config.get('features', {}))
    normalizer = DataNormalizer()
    donchian = DonchianChannel(period=config.get('features', {}).get('donchian_period', 20))
    
    symbols = loader.load_symbols_list()
    index_data = loader.load_index()
    success_threshold = config.get('buy_env', {}).get('success_threshold', 0.10)
    max_holding_days = config.get('sell_env', {}).get('max_holding_days', 120)
    
    # 論文的時間分割點
    train_end_date = pd.Timestamp(config.get('backtest', {}).get('train_end_date', '2017-10-15'))
    
    train_signals = []
    eval_signals = []
    logger.info(f"掃描 {len(symbols)} 支股票產生買入樣本...")
    logger.info(f"訓練期: < {train_end_date.date()}, 評估期: >= {train_end_date.date()}")
    
    for i, symbol in enumerate(symbols):
        df = loader.load_symbol(symbol)
        if df is None: continue
        try:
            cache_file = Path(f"data/cache/{symbol}_features.pkl")
            df_normalized = pd.read_pickle(cache_file) if cache_file.exists() else normalizer.normalize(feature_calc.calculate_all_features(df, index_data))
            
            buy_signals = donchian.generate_buy_signals(df_normalized)
            signal_indices = buy_signals[buy_signals == 1].index
            
            for signal_date in signal_indices:
                if signal_date not in df_normalized.index: continue
                signal_loc = df_normalized.index.get_loc(signal_date)
                buy_price = df_normalized.iloc[signal_loc]['Close']
                
                future_data = df_normalized.iloc[signal_loc+1 : signal_loc+1+max_holding_days]
                if len(future_data) < 5: continue
                
                max_return = (future_data['High'].max() - buy_price) / buy_price
                label = 2 if max_return >= success_threshold else 1
                
                # 只保留正規化特徵
                feature_cols = normalizer.get_normalized_feature_columns()
                available_cols = [c for c in feature_cols if c in df_normalized.columns]
                
                row = df_normalized.iloc[signal_loc][available_cols].copy()
                row['label'] = label
                row['signal_date'] = signal_date  # 保留日期用於分割
                
                # 根據日期分配到訓練或評估集
                if signal_date <= train_end_date:
                    train_signals.append(row)
                else:
                    eval_signals.append(row)
        except: continue
        if (i+1) % 100 == 0: logger.info(f"進度: {i+1}/{len(symbols)}")

    train_df = pd.DataFrame(train_signals) if train_signals else pd.DataFrame()
    eval_df = pd.DataFrame(eval_signals) if eval_signals else pd.DataFrame()
    
    # 移除 signal_date 欄位 (不作為特徵)
    if 'signal_date' in train_df.columns: train_df = train_df.drop(columns=['signal_date'])
    if 'signal_date' in eval_df.columns: eval_df = eval_df.drop(columns=['signal_date'])
    
    logger.info(f"Buy 資料準備完成: 訓練 {len(train_df)} 筆, 評估 {len(eval_df)} 筆")
    return train_df, eval_df, index_data



def prepare_sell_data(config: dict) -> tuple:
    """
    準備 Sell Agent 訓練資料 (成功交易的 episodes)
    
    Returns:
        train_episodes: 訓練 episodes (< 2017-10-16)
        eval_episodes: 評估 episodes (>= 2017-10-16)
    """
    logger.info("開始準備 Sell Agent 訓練資料...")
    data_config = config.get('data', {})
    loader = DataLoader(data_config)
    feature_calc = FeatureCalculator(config.get('features', {}))
    normalizer = DataNormalizer()
    donchian = DonchianChannel(period=config.get('features', {}).get('donchian_period', 20))
    
    symbols = loader.load_symbols_list()
    index_data = loader.load_index()
    success_threshold = config.get('buy_env', {}).get('success_threshold', 0.10)
    max_holding_days = config.get('sell_env', {}).get('max_holding_days', 120)
    
    # 論文的時間分割點
    train_end_date = pd.Timestamp(config.get('backtest', {}).get('train_end_date', '2017-10-15'))
    
    train_episodes = {}
    eval_episodes = {}
    train_count = 0
    eval_count = 0
    max_episodes = config.get('training', {}).get('max_sell_episodes', 50000)
    
    logger.info(f"掃描股票產生 Sell episodes (訓練上限 {max_episodes})...")
    logger.info(f"訓練期: < {train_end_date.date()}, 評估期: >= {train_end_date.date()}")
    
    # 隨機打亂股票順序，讓抽樣更平均
    import random
    random.shuffle(symbols)
    
    for i, symbol in enumerate(symbols):
        if train_count >= max_episodes: break
        df = loader.load_symbol(symbol)
        if df is None: continue
        try:
            cache_file = Path(f"data/cache/{symbol}_features.pkl")
            df_normalized = pd.read_pickle(cache_file) if cache_file.exists() else normalizer.normalize(feature_calc.calculate_all_features(df, index_data))
            
            buy_signals = donchian.generate_buy_signals(df_normalized)
            signal_indices = buy_signals[buy_signals == 1].index
            
            for signal_date in signal_indices:
                if signal_date not in df_normalized.index: continue
                signal_loc = df_normalized.index.get_loc(signal_date)
                buy_price = df_normalized.iloc[signal_loc]['Close']
                
                future_data = df_normalized.iloc[signal_loc+1 : signal_loc+1+max_holding_days]
                if len(future_data) < 5: continue
                
                # 論文要求: Sell Agent 只從 >= 10% 報酬的成功交易中學習
                max_return = (future_data['High'].max() - buy_price) / buy_price
                if max_return < 0.10: continue
                
                # 只保留正規化特徵 + Close (for reward calculation)
                feature_cols = normalizer.get_normalized_feature_columns()
                available_cols = [c for c in feature_cols if c in df_normalized.columns]
                keep_cols = available_cols + ['Close'] if 'Close' in df_normalized.columns else available_cols
                
                episode_df = df_normalized.iloc[signal_loc : signal_loc+1+max_holding_days][keep_cols].copy()
                
                episode_data = {
                    'features': episode_df,
                    'buy_price': buy_price,
                    'buy_date': signal_date
                }
                
                # 根據日期分配到訓練或評估集
                if signal_date <= train_end_date:
                    if train_count < max_episodes:
                        train_episodes[f"{symbol}_{signal_date.date()}"] = episode_data
                        train_count += 1
                else:
                    eval_episodes[f"{symbol}_{signal_date.date()}"] = episode_data
                    eval_count += 1
        except: continue
        if (i+1) % 50 == 0: logger.info(f"進度: {i+1}/{len(symbols)} (訓練 {train_count}, 評估 {eval_count})")

    logger.info(f"Sell 資料準備完成: 訓練 {len(train_episodes)} 個, 評估 {len(eval_episodes)} 個")
    return train_episodes, eval_episodes



def make_buy_env(data, config):
    def _init():
        return Monitor(BuyEnv(data, config))
    return _init


def make_sell_env(episodes, config):
    def _init():
        return Monitor(SellEnv(episodes, config))
    return _init


def train_buy_agent(config: dict, train_data: pd.DataFrame, eval_data: pd.DataFrame, resume: bool = False):
    """訓練 Buy Agent (使用時間分割的資料)"""
    logger.info("=" * 50)
    logger.info("開始訓練 Buy Agent")
    logger.info("=" * 50)
    
    n_envs = config.get('training', {}).get('n_envs', 1)
    buy_env_config = config.get('buy_env', {})
    
    logger.info(f"訓練資料: {len(train_data)} 筆, 評估資料: {len(eval_data)} 筆")
    
    if n_envs > 1:
        env = SubprocVecEnv([make_buy_env(train_data, buy_env_config) for _ in range(n_envs)])
        logger.info(f"使用 SubprocVecEnv 並行環境: {n_envs}")
    else:
        env = DummyVecEnv([make_buy_env(train_data, buy_env_config)])
    
    # 注意: 暫時禁用 eval_env，避免訓練中斷
    # 可以用 TensorBoard 的 rollout/ep_rew_mean 監控，訓練完成後用 backtest 驗證
    eval_env = None  # DummyVecEnv([make_buy_env(eval_data, buy_env_config)]) if len(eval_data) > 0 else None
    
    agent = BuyAgent(config.get('buy_agent', {}))
    total_timesteps = config.get('buy_agent', {}).get('total_timesteps', 1000000)
    
    agent.train(env=env, total_timesteps=total_timesteps, eval_env=eval_env, resume=resume)
    env.close()
    if eval_env: eval_env.close()
    return agent




def train_sell_agent(config: dict, train_episodes: dict, eval_episodes: dict, resume: bool = False):
    """訓練 Sell Agent (使用時間分割的資料)"""
    logger.info("=" * 50)
    logger.info("開始訓練 Sell Agent")
    logger.info("=" * 50)
    
    n_envs = config.get('training', {}).get('n_envs', 1)
    sell_env_config = config.get('sell_env', {})
    
    logger.info(f"訓練 episodes: {len(train_episodes)} 個, 評估 episodes: {len(eval_episodes)} 個")
    
    if n_envs > 1:
        env = SubprocVecEnv([make_sell_env(train_episodes, sell_env_config) for _ in range(n_envs)])
        logger.info(f"使用 SubprocVecEnv 並行環境: {n_envs}")
    else:
        env = DummyVecEnv([make_sell_env(train_episodes, sell_env_config)])
    
    # 注意: 暫時禁用 eval_env，避免訓練中斷
    # 可以用 TensorBoard 的 rollout/ep_rew_mean 監控，訓練完成後用 backtest 驗證
    eval_env = None  # DummyVecEnv([make_sell_env(eval_episodes, sell_env_config)]) if len(eval_episodes) > 0 else None
    
    agent = SellAgent(config.get('sell_agent', {}))
    total_timesteps = config.get('sell_agent', {}).get('total_timesteps', 1000000)
    
    agent.train(env=env, total_timesteps=total_timesteps, eval_env=eval_env, resume=resume)
    env.close()
    if eval_env: eval_env.close()
    return agent




def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='Pro Trader RL 訓練腳本')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--buy-only', action='store_true')
    parser.add_argument('--sell-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='從檢查點恢復訓練')
    args = parser.parse_args()
    
    config_path = project_root / args.config
    config = load_config(str(config_path))
    
    # 訓練 Buy Agent
    if not args.sell_only:
        train_data, eval_data, _ = prepare_buy_data(config)
        train_buy_agent(config, train_data, eval_data, resume=args.resume)
        # 訓練完後釋放記憶體
        del train_data, eval_data
        import gc
        gc.collect()
    
    # 訓練 Sell Agent
    if not args.buy_only:
        train_episodes, eval_episodes = prepare_sell_data(config)
        train_sell_agent(config, train_episodes, eval_episodes, resume=args.resume)
        del train_episodes, eval_episodes
        import gc
        gc.collect()

    
    logger.info("=" * 50)
    logger.info("全部訓練任務完成!")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
