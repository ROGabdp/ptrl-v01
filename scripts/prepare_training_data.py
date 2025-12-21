#!/usr/bin/env python3
"""
Step 1: 預處理訓練資料 (只需執行一次)

將 2.4GB 的 features_v3.pkl 預處理成較小的訓練資料檔案:
- buy_training_data.pkl (~50MB)
- sell_training_data.pkl (~100MB)

用法:
    python3 scripts/prepare_training_data.py
"""

import os
import sys
import pickle
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger


def get_feature_cols(sample_df: pd.DataFrame) -> list:
    """取得用於模型的特徵欄位"""
    norm_cols = [c for c in sample_df.columns if '_norm' in c]
    extra_cols = [
        'Return', 'Index_Return',
        'SuperTrend_14', 'SuperTrend_21',
        'Up_Stock', 'Down_Stock',
        'RS_Rate_5', 'RS_Rate_10', 'RS_Rate_20', 'RS_Rate_40',
        'RS_Momentum', 'RS_Trend'
    ]
    extra_cols = [c for c in extra_cols if c in sample_df.columns]
    feature_cols = norm_cols + extra_cols
    feature_cols = [c for c in feature_cols if 'Volume' not in c]
    return feature_cols


def prepare_buy_data(features: dict, feature_cols: list, success_threshold: float = 0.10):
    """準備 Buy Agent 訓練資料"""
    logger.info("準備 Buy Agent 訓練資料...")
    
    all_signals = []
    
    for symbol, df in features.items():
        if df is None or len(df) < 252:
            continue
        
        if 'Donchian_Upper' not in df.columns or 'High' not in df.columns:
            continue
        
        df = df.copy()
        df['is_breakout'] = df['High'] > df['Donchian_Upper'].shift(1)
        breakout_df = df[df['is_breakout'] == True].copy()
        
        if len(breakout_df) == 0:
            continue
        
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
        
        valid = breakout_df.dropna(subset=['actual_return'])
        if len(valid) > 0:
            all_signals.append(valid)
    
    combined = pd.concat(all_signals, ignore_index=False)
    
    # 平衡資料
    successful = combined[combined['is_successful'] == True]
    failed = combined[combined['is_successful'] == False]
    min_count = min(len(successful), len(failed))
    
    balanced = pd.concat([
        successful.sample(n=min_count, random_state=42),
        failed.sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    # 只保留需要的欄位
    required_cols = feature_cols + ['actual_return', 'is_successful']
    available_cols = [c for c in required_cols if c in balanced.columns]
    
    result = balanced[available_cols].copy()
    logger.info(f"Buy 訓練資料: {len(result)} 樣本, {len(available_cols)} 欄位")
    return result


def prepare_sell_data(features: dict, feature_cols: list, success_threshold: float = 0.10, max_holding_days: int = 120):
    """準備 Sell Agent 訓練資料"""
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
                
                holding_period['sell_return'] = holding_period['Close'] / buy_price
                max_return = holding_period['sell_return'].max() - 1
                
                if max_return >= success_threshold:
                    holding_period['buy_price'] = buy_price
                    holding_period['episode_id'] = f"{symbol}_{loc}"
                    all_episodes.append(holding_period)
            except:
                continue
    
    combined = pd.concat(all_episodes, ignore_index=False)
    
    # 只保留需要的欄位
    required_cols = feature_cols + ['Close', 'Open', 'buy_price', 'episode_id', 'sell_return']
    available_cols = [c for c in required_cols if c in combined.columns]
    
    result = combined[available_cols].copy()
    logger.info(f"Sell 訓練資料: {len(all_episodes)} episodes, {len(result)} 樣本")
    return result


def main():
    logger.info("=" * 60)
    logger.info("預處理訓練資料")
    logger.info("=" * 60)
    
    features_path = "data/processed/features_v3.pkl"
    output_dir = Path("data/processed")
    
    # 載入特徵
    logger.info(f"載入 {features_path}...")
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    logger.info(f"載入 {len(features)} 支股票")
    
    # 取得特徵欄位
    sample = list(features.keys())[0]
    feature_cols = get_feature_cols(features[sample])
    logger.info(f"特徵欄位數: {len(feature_cols)}")
    
    # 準備 Buy 資料
    buy_data = prepare_buy_data(features, feature_cols)
    buy_path = output_dir / "buy_training_data.pkl"
    with open(buy_path, 'wb') as f:
        pickle.dump({'data': buy_data, 'feature_cols': feature_cols}, f)
    logger.info(f"Buy 資料已儲存: {buy_path} ({buy_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # 釋放記憶體
    del buy_data
    
    # 準備 Sell 資料
    sell_data = prepare_sell_data(features, feature_cols)
    sell_path = output_dir / "sell_training_data.pkl"
    with open(sell_path, 'wb') as f:
        pickle.dump({'data': sell_data, 'feature_cols': feature_cols}, f)
    logger.info(f"Sell 資料已儲存: {sell_path} ({sell_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    logger.info("=" * 60)
    logger.info("預處理完成！現在可以執行多核訓練:")
    logger.info("  python3 scripts/train_wsl_v2.py --agent buy")
    logger.info("  python3 scripts/train_wsl_v2.py --agent sell")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
