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

from src.data import DataLoader, FeatureCalculator, DataNormalizer
from src.environments import BuyEnv, SellEnv
from src.agents import BuyAgent, SellAgent
from src.rules import DonchianChannel, StopLossRule


def load_config(config_path: str) -> dict:
    """載入設定檔"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_training_data(config: dict) -> tuple:
    """
    準備訓練資料
    
    Returns:
        (buy_signals_data, sell_episodes_data, index_data)
    """
    logger.info("開始準備訓練資料...")
    
    # 1. 初始化模組
    data_config = config.get('data', {})
    loader = DataLoader(data_config)
    feature_calc = FeatureCalculator(config.get('features', {}))
    normalizer = DataNormalizer()
    donchian = DonchianChannel(period=config.get('features', {}).get('donchian_period', 20))
    
    # 2. 載入資料
    symbols = loader.load_symbols_list()
    if not symbols:
        logger.warning("找不到股票清單，嘗試從 Wikipedia 取得")
        symbols = loader.get_sp500_symbols()
    
    index_data = loader.load_index()
    
    # 3. 處理每支股票
    all_buy_signals = []
    all_sell_episodes = {}
    success_threshold = config.get('buy_env', {}).get('success_threshold', 0.10)
    max_holding_days = config.get('sell_env', {}).get('max_holding_days', 120)
    
    for symbol in symbols[:50]:  # 先用前 50 支進行測試
        df = loader.load_symbol(symbol)
        if df is None:
            continue
        
        try:
            # 計算特徵
            df_features = feature_calc.calculate_all_features(df, index_data)
            
            # 正規化
            df_normalized = normalizer.normalize(df_features)
            
            # 產生買入訊號
            buy_signals = donchian.generate_buy_signals(df_normalized)
            signal_indices = buy_signals[buy_signals == 1].index
            
            # 為每個買入訊號計算結果
            for signal_date in signal_indices:
                if signal_date not in df_normalized.index:
                    continue
                
                signal_loc = df_normalized.index.get_loc(signal_date)
                buy_price = df_normalized.iloc[signal_loc]['Close']
                
                # 計算後 120 天的報酬
                future_end = min(signal_loc + max_holding_days, len(df_normalized))
                future_prices = df_normalized.iloc[signal_loc:future_end]['Close']
                
                if len(future_prices) < 10:
                    continue
                
                max_return = (future_prices.max() - buy_price) / buy_price
                is_successful = max_return >= success_threshold
                
                # 提取正規化特徵
                feature_cols = normalizer.get_normalized_feature_columns()
                available_cols = [c for c in feature_cols if c in df_normalized.columns]
                
                signal_features = df_normalized.loc[signal_date, available_cols].values
                
                all_buy_signals.append({
                    'symbol': symbol,
                    'date': signal_date,
                    'features': signal_features,
                    'actual_return': max_return,
                    'is_successful': is_successful
                })
                
                # 如果是成功的買入，加入 Sell Agent 訓練資料
                if is_successful:
                    key = f"{symbol}_{signal_date.strftime('%Y%m%d')}"
                    all_sell_episodes[key] = {
                        'features': df_normalized.iloc[signal_loc:future_end],
                        'buy_date': signal_date,
                        'buy_price': buy_price
                    }
        
        except Exception as e:
            logger.warning(f"處理 {symbol} 時發生錯誤: {e}")
            continue
    
    # 轉換為 DataFrame
    if all_buy_signals:
        feature_cols = normalizer.get_normalized_feature_columns()
        available_cols = [c for c in feature_cols if c in df_normalized.columns]
        
        buy_df = pd.DataFrame([
            {**{col: sig['features'][i] if i < len(sig['features']) else np.nan 
                for i, col in enumerate(available_cols)},
             'actual_return': sig['actual_return'],
             'is_successful': sig['is_successful']}
            for sig in all_buy_signals
        ])
        
        logger.info(f"準備完成: {len(buy_df)} 個買入訊號, {len(all_sell_episodes)} 個賣出 episodes")
    else:
        buy_df = pd.DataFrame()
    
    return buy_df, all_sell_episodes, index_data


def train_buy_agent(config: dict, training_data: pd.DataFrame):
    """訓練 Buy Agent"""
    logger.info("=" * 50)
    logger.info("開始訓練 Buy Agent")
    logger.info("=" * 50)
    
    if len(training_data) < 100:
        logger.error(f"訓練資料不足: {len(training_data)} 個樣本")
        return None
    
    # 建立環境
    buy_env_config = config.get('buy_env', {})
    env = BuyEnv(training_data, buy_env_config)
    
    # 建立 Agent
    buy_agent_config = config.get('buy_agent', {})
    agent = BuyAgent(buy_agent_config)
    
    # 訓練
    total_timesteps = config.get('training', {}).get('buy_agent_steps', 1000000)
    
    agent.train(
        env=env,
        total_timesteps=total_timesteps,
        eval_env=None,  # 可以分割資料建立評估環境
        resume=False
    )
    
    logger.info("Buy Agent 訓練完成!")
    return agent


def train_sell_agent(config: dict, sell_episodes: dict):
    """訓練 Sell Agent"""
    logger.info("=" * 50)
    logger.info("開始訓練 Sell Agent")
    logger.info("=" * 50)
    
    if len(sell_episodes) < 50:
        logger.error(f"訓練資料不足: {len(sell_episodes)} 個 episodes")
        return None
    
    # 建立環境
    sell_env_config = config.get('sell_env', {})
    env = SellEnv(sell_episodes, sell_env_config)
    
    # 建立 Agent
    sell_agent_config = config.get('sell_agent', {})
    agent = SellAgent(sell_agent_config)
    
    # 訓練
    total_timesteps = config.get('training', {}).get('sell_agent_steps', 1000000)
    
    agent.train(
        env=env,
        total_timesteps=total_timesteps,
        eval_env=None,
        resume=False
    )
    
    logger.info("Sell Agent 訓練完成!")
    return agent


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='Pro Trader RL 訓練腳本')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='設定檔路徑')
    parser.add_argument('--buy-only', action='store_true',
                       help='只訓練 Buy Agent')
    parser.add_argument('--sell-only', action='store_true',
                       help='只訓練 Sell Agent')
    parser.add_argument('--resume', action='store_true',
                       help='從檢查點恢復訓練')
    args = parser.parse_args()
    
    # 載入設定
    config_path = project_root / args.config
    if not config_path.exists():
        logger.error(f"設定檔不存在: {config_path}")
        return
    
    config = load_config(str(config_path))
    
    # 準備訓練資料
    buy_data, sell_episodes, index_data = prepare_training_data(config)
    
    # 訓練 Agent
    if not args.sell_only:
        buy_agent = train_buy_agent(config, buy_data)
    
    if not args.buy_only:
        sell_agent = train_sell_agent(config, sell_episodes)
    
    logger.info("=" * 50)
    logger.info("訓練完成!")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
