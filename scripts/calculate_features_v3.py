"""
Calculate Features V3 - 包含市場廣度指標的完整特徵計算腳本

功能:
1. 載入所有 S&P 500 股票資料
2. 計算市場廣度 (Up_Stock / Down_Stock)
3. 計算每支股票的 69 維特徵
4. 合併市場廣度指標
5. 正規化並儲存

用法:
    python scripts/calculate_features_v3.py
"""

import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

from src.data import DataLoader, FeatureCalculator, DataNormalizer, MarketBreadthCalculator


def main():
    """計算 V3 特徵 (含市場廣度)"""
    
    logger.info("=" * 60)
    logger.info("Pro Trader RL V3 - 特徵計算")
    logger.info("=" * 60)
    
    # 設定
    data_dir = "data/raw"
    processed_dir = "data/processed"
    cache_file = f"{processed_dir}/features_v3.pkl"
    breadth_cache = f"{processed_dir}/market_breadth.csv"
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. 初始化模組
    logger.info("[1/5] 初始化模組...")
    loader = DataLoader({'data_dir': data_dir})
    calculator = FeatureCalculator()
    normalizer = DataNormalizer()
    breadth_calc = MarketBreadthCalculator(normalize=True)
    
    # 2. 載入所有股票資料
    logger.info("[2/5] 載入所有股票資料...")
    all_data = loader.load_all()
    logger.info(f"載入 {len(all_data)} 支股票")
    
    # 載入指數資料
    index_df = loader.load_index()
    if index_df is None:
        logger.warning("找不到指數資料，將跳過指數相關特徵")
    
    # 3. 計算市場廣度
    logger.info("[3/5] 計算市場廣度 (Up_Stock / Down_Stock)...")
    breadth_df = breadth_calc.calculate_and_cache(all_data, breadth_cache)
    logger.info(f"市場廣度計算完成: {len(breadth_df)} 天")
    logger.info(f"  Up_Stock 平均: {breadth_df['Up_Stock'].mean():.3f}")
    logger.info(f"  Down_Stock 平均: {breadth_df['Down_Stock'].mean():.3f}")
    
    # 4. 計算每支股票的特徵
    logger.info("[4/5] 計算每支股票特徵並合併市場廣度...")
    all_features = {}
    failed_symbols = []
    
    for symbol, df in tqdm(all_data.items(), desc="計算特徵"):
        if df is None or len(df) < 252:  # 至少需要 1 年資料
            failed_symbols.append(symbol)
            continue
        
        try:
            # 計算基本特徵
            features = calculator.calculate_all_features(df, index_df)
            
            # 正規化
            features = normalizer.normalize(features)
            
            # 移除 feature_calculator 中預設的 NaN 欄位 (避免重複)
            if 'Up_Stock' in features.columns:
                features = features.drop(columns=['Up_Stock'])
            if 'Down_Stock' in features.columns:
                features = features.drop(columns=['Down_Stock'])
            
            # 合併市場廣度 (按日期 join)
            if len(breadth_df) > 0:
                features = features.join(breadth_df[['Up_Stock', 'Down_Stock']], how='left')
            else:
                features['Up_Stock'] = 0.5
                features['Down_Stock'] = 0.5
            
            # 填充缺失值
            features['Up_Stock'] = features['Up_Stock'].fillna(0.5)
            features['Down_Stock'] = features['Down_Stock'].fillna(0.5)
            
            all_features[symbol] = features
            
        except Exception as e:
            logger.warning(f"{symbol} 特徵計算失敗: {e}")
            failed_symbols.append(symbol)
    
    logger.info(f"成功計算 {len(all_features)} 支股票特徵")
    if failed_symbols:
        logger.warning(f"失敗股票: {len(failed_symbols)} 支")
    
    # 5. 儲存特徵
    logger.info("[5/5] 儲存特徵快取...")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(all_features, f)
    
    logger.info(f"特徵已儲存至: {cache_file}")
    
    # 驗證特徵維度
    sample_symbol = list(all_features.keys())[0]
    sample_df = all_features[sample_symbol]
    
    # 計算正規化後的特徵欄位數 (排除原始欄位和非特徵欄位)
    norm_cols = [c for c in sample_df.columns if '_norm' in c or c in [
        'Return', 'Index_Return', 'SuperTrend_14', 'SuperTrend_21',
        'Up_Stock', 'Down_Stock', 'RS_Rate_5', 'RS_Rate_10', 
        'RS_Rate_20', 'RS_Rate_40', 'RS_Momentum', 'RS_Trend'
    ]]
    
    logger.info("=" * 60)
    logger.info("特徵計算完成!")
    logger.info(f"  股票數: {len(all_features)}")
    logger.info(f"  樣本股票: {sample_symbol}")
    logger.info(f"  總欄位數: {len(sample_df.columns)}")
    logger.info(f"  正規化欄位數 (約): {len(norm_cols)}")
    logger.info(f"  包含 Up_Stock: {'Up_Stock' in sample_df.columns}")
    logger.info(f"  包含 Down_Stock: {'Down_Stock' in sample_df.columns}")
    logger.info("=" * 60)
    
    return all_features


if __name__ == '__main__':
    main()
