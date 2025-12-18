#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日資料更新腳本

功能:
- 更新 S&P 500 成分股列表
- 下載/更新個股歷史資料
- 更新指數資料
- 計算並快取特徵

使用方式:
    python scripts/data_update.py
    python scripts/data_update.py --symbols AAPL MSFT GOOGL
    python scripts/data_update.py --all
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# 設定專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.data import DataLoader, FeatureCalculator, DataNormalizer


def load_config(config_path: str) -> dict:
    """載入 YAML 設定檔"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str = 'logs/daily_ops/'):
    """設定日誌"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"data_update_{datetime.now().strftime('%Y%m%d')}.log")
    
    logger.add(
        log_file,
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )


def update_symbols_list(loader: DataLoader) -> list:
    """更新股票列表"""
    logger.info("正在更新 S&P 500 成分股列表...")
    
    try:
        symbols = loader.get_sp500_symbols()
        
        # 儲存列表
        symbols_file = project_root / 'config' / 'sp500_symbols.txt'
        with open(symbols_file, 'w') as f:
            f.write('\n'.join(symbols))
        
        logger.info(f"已更新 {len(symbols)} 支股票")
        return symbols
        
    except Exception as e:
        logger.error(f"更新股票列表失敗: {e}")
        
        # 嘗試載入現有列表
        existing = loader.load_symbols_list()
        if existing:
            logger.info(f"使用現有列表: {len(existing)} 支股票")
            return existing
        
        return []


def update_stock_data(loader: DataLoader, symbols: list, 
                      start_date: str = None, end_date: str = None) -> dict:
    """
    更新個股資料
    
    Args:
        loader: DataLoader 實例
        symbols: 股票代碼列表
        start_date: 起始日期
        end_date: 結束日期
        
    Returns:
        更新結果統計
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # 預設更新最近 30 天
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info(f"正在更新 {len(symbols)} 支股票資料 ({start_date} ~ {end_date})...")
    
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    for i, symbol in enumerate(symbols):
        try:
            # 檢查是否需要更新
            existing = loader.load_symbol(symbol)
            
            if existing is not None and len(existing) > 0:
                last_date = existing.index[-1]
                if last_date >= pd.to_datetime(end_date):
                    results['skipped'].append(symbol)
                    continue
            
            # 下載更新
            df = loader.download_symbol(symbol, start_date, end_date)
            
            if df is not None and len(df) > 0:
                results['success'].append(symbol)
            else:
                results['failed'].append(symbol)
                
        except Exception as e:
            logger.debug(f"更新 {symbol} 失敗: {e}")
            results['failed'].append(symbol)
        
        # 進度顯示
        if (i + 1) % 50 == 0:
            logger.info(f"進度: {i+1}/{len(symbols)}")
    
    logger.info(f"更新完成 - 成功: {len(results['success'])}, "
               f"失敗: {len(results['failed'])}, 跳過: {len(results['skipped'])}")
    
    return results


def update_index_data(loader: DataLoader, start_date: str = None, 
                      end_date: str = None) -> bool:
    """更新指數資料"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info("正在更新指數資料...")
    
    try:
        df = loader.download_index(start_date, end_date)
        
        if df is not None and len(df) > 0:
            logger.info(f"指數資料更新成功: {len(df)} 筆")
            return True
        else:
            logger.warning("指數資料更新失敗")
            return False
            
    except Exception as e:
        logger.error(f"更新指數資料失敗: {e}")
        return False


def calculate_features(loader: DataLoader, symbols: list, 
                       feature_calc: FeatureCalculator,
                       normalizer: DataNormalizer,
                       cache_dir: str = 'data/cache/') -> int:
    """
    計算並快取特徵
    
    Args:
        loader: DataLoader 實例
        symbols: 股票列表
        feature_calc: FeatureCalculator 實例
        normalizer: DataNormalizer 實例
        cache_dir: 快取目錄
        
    Returns:
        成功計算的股票數量
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"正在計算 {len(symbols)} 支股票的特徵...")
    
    index_data = loader.load_index()
    
    if index_data is None:
        logger.error("無法載入指數資料，跳過特徵計算")
        return 0
    
    success_count = 0
    
    for i, symbol in enumerate(symbols):
        try:
            df = loader.load_symbol(symbol)
            
            if df is None or len(df) < 250:  # 需要至少一年資料
                continue
            
            # 計算特徵
            features = feature_calc.calculate_all_features(df, index_data)
            
            # 正規化
            normalized = normalizer.normalize(features)
            
            # 儲存快取
            cache_file = os.path.join(cache_dir, f"{symbol}_features.pkl")
            normalized.to_pickle(cache_file)
            
            success_count += 1
            
        except Exception as e:
            logger.debug(f"計算 {symbol} 特徵失敗: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            logger.info(f"特徵計算進度: {i+1}/{len(symbols)}")
    
    logger.info(f"特徵計算完成: {success_count}/{len(symbols)}")
    
    return success_count


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Pro Trader RL 每日資料更新')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='設定檔路徑')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                       help='指定股票代碼')
    parser.add_argument('--all', action='store_true',
                       help='更新所有股票')
    parser.add_argument('--start', type=str, default=None,
                       help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='結束日期 (YYYY-MM-DD)')
    parser.add_argument('--features', action='store_true',
                       help='同時計算特徵')
    parser.add_argument('--update-list', action='store_true',
                       help='更新股票列表')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging()
    
    logger.info("=" * 50)
    logger.info("Pro Trader RL 每日資料更新")
    logger.info(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    # 載入設定
    config_path = project_root / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {}
    
    # 初始化模組
    loader = DataLoader(config.get('data', {}))
    
    # 決定要更新的股票
    if args.symbols:
        symbols = args.symbols
    elif args.all or args.update_list:
        symbols = update_symbols_list(loader)
    else:
        symbols = loader.load_symbols_list()
        if not symbols:
            symbols = update_symbols_list(loader)
    
    if not symbols:
        logger.error("無法取得股票列表")
        return
    
    # 更新指數
    update_index_data(loader, args.start, args.end)
    
    # 更新個股
    results = update_stock_data(loader, symbols, args.start, args.end)
    
    # 計算特徵 (可選)
    if args.features:
        feature_calc = FeatureCalculator(config.get('features', {}))
        normalizer = DataNormalizer()
        calculate_features(loader, symbols, feature_calc, normalizer)
    
    # 輸出摘要
    print("\n" + "=" * 50)
    print("資料更新摘要")
    print("=" * 50)
    print(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"股票總數: {len(symbols)}")
    print(f"更新成功: {len(results['success'])}")
    print(f"更新失敗: {len(results['failed'])}")
    print(f"資料已是最新: {len(results['skipped'])}")
    print("=" * 50)


if __name__ == '__main__':
    main()
