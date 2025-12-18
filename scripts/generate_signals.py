#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日交易訊號產生腳本

功能:
- 掃描 Donchian Channel 突破訊號
- 使用 Buy Agent 過濾訊號
- 檢查現有持倉的賣出條件
- 產生交易建議

使用方式:
    python scripts/generate_signals.py
    python scripts/generate_signals.py --date 2023-12-18
    python scripts/generate_signals.py --positions positions.csv
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# 設定專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger

from src.data import DataLoader, FeatureCalculator, DataNormalizer
from src.agents import BuyAgent, SellAgent
from src.rules import StopLossRule, DonchianChannel


def load_config(config_path: str) -> dict:
    """載入 YAML 設定檔"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str = 'logs/daily_ops/'):
    """設定日誌"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"signals_{datetime.now().strftime('%Y%m%d')}.log")
    
    logger.add(
        log_file,
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )


def load_positions(filepath: str) -> list:
    """
    載入現有持倉
    
    Args:
        filepath: 持倉 CSV 檔案路徑
        
    Returns:
        持倉列表
    """
    if not os.path.exists(filepath):
        logger.warning(f"持倉檔案不存在: {filepath}")
        return []
    
    try:
        df = pd.read_csv(filepath)
        
        positions = []
        for _, row in df.iterrows():
            positions.append({
                'symbol': row['symbol'],
                'buy_date': pd.to_datetime(row['buy_date']),
                'buy_price': row['buy_price'],
                'shares': row.get('shares', 1)
            })
        
        logger.info(f"載入 {len(positions)} 筆持倉")
        return positions
        
    except Exception as e:
        logger.error(f"載入持倉失敗: {e}")
        return []


def scan_buy_signals(loader: DataLoader, feature_calc: FeatureCalculator,
                     normalizer: DataNormalizer, buy_agent: BuyAgent,
                     donchian: DonchianChannel, target_date: datetime,
                     symbols: list = None) -> list:
    """
    掃描買入訊號
    
    Args:
        loader: DataLoader
        feature_calc: FeatureCalculator
        normalizer: DataNormalizer
        buy_agent: BuyAgent
        donchian: DonchianChannel
        target_date: 目標日期
        symbols: 股票列表 (可選)
        
    Returns:
        買入訊號列表
    """
    if symbols is None:
        symbols = loader.load_symbols_list() or []
    
    logger.info(f"掃描 {len(symbols)} 支股票的買入訊號...")
    
    index_data = loader.load_index()
    signals = []
    
    for symbol in symbols:
        try:
            df = loader.load_symbol(symbol)
            
            if df is None or len(df) < 250:
                continue
            
            if target_date not in df.index:
                continue
            
            idx = df.index.get_loc(target_date)
            
            # 檢查 Donchian 突破
            dc_signals = donchian.calculate(df)
            
            if idx >= len(dc_signals) or not dc_signals.iloc[idx]['buy_signal']:
                continue
            
            # 計算特徵
            features_df = feature_calc.calculate_all_features(df, index_data)
            normalized = normalizer.normalize(features_df)
            
            # 取得正規化特徵
            feature_cols = normalizer.get_normalized_feature_columns()
            available_cols = [c for c in feature_cols if c in normalized.columns]
            
            if target_date not in normalized.index:
                continue
            
            obs = normalized.loc[target_date, available_cols].values.astype(np.float32)
            obs = np.nan_to_num(obs, nan=0.0)
            
            # Buy Agent 過濾
            confidence = 1.0
            should_buy = True
            
            if buy_agent.model is not None:
                action = buy_agent.predict(obs)
                probs = buy_agent.predict_proba(obs)
                confidence = float(probs[1])
                should_buy = action == 1
            
            if should_buy:
                signals.append({
                    'symbol': symbol,
                    'date': target_date,
                    'price': df.loc[target_date, 'Close'],
                    'donchian_upper': dc_signals.iloc[idx]['upper'],
                    'confidence': confidence,
                    'recommendation': 'BUY'
                })
                
        except Exception as e:
            logger.debug(f"處理 {symbol} 時錯誤: {e}")
            continue
    
    # 按信心度排序，取 Top 10
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    signals = signals[:10]
    
    logger.info(f"找到 {len(signals)} 個買入訊號")
    
    return signals


def check_sell_signals(loader: DataLoader, feature_calc: FeatureCalculator,
                       normalizer: DataNormalizer, sell_agent: SellAgent,
                       stop_loss: StopLossRule, positions: list,
                       target_date: datetime) -> list:
    """
    檢查賣出條件
    
    Args:
        loader: DataLoader
        feature_calc: FeatureCalculator
        normalizer: DataNormalizer
        sell_agent: SellAgent
        stop_loss: StopLossRule
        positions: 持倉列表
        target_date: 目標日期
        
    Returns:
        賣出訊號列表
    """
    logger.info(f"檢查 {len(positions)} 筆持倉的賣出條件...")
    
    index_data = loader.load_index()
    signals = []
    
    for pos in positions:
        symbol = pos['symbol']
        buy_date = pos['buy_date']
        buy_price = pos['buy_price']
        
        try:
            df = loader.load_symbol(symbol)
            
            if df is None or target_date not in df.index:
                continue
            
            current_price = df.loc[target_date, 'Close']
            holding_days = (target_date - buy_date).days
            current_return = (current_price - buy_price) / buy_price
            
            # 取得價格歷史
            buy_idx = df.index.get_loc(buy_date) if buy_date in df.index else 0
            current_idx = df.index.get_loc(target_date)
            price_history = df.iloc[buy_idx:current_idx+1]['Close']
            
            # 檢查停損
            stop_result = stop_loss.check(
                buy_price=buy_price,
                current_price=current_price,
                holding_days=holding_days,
                price_history=price_history
            )
            
            if stop_result.should_stop:
                signals.append({
                    'symbol': symbol,
                    'date': target_date,
                    'price': current_price,
                    'buy_price': buy_price,
                    'return_pct': current_return,
                    'holding_days': holding_days,
                    'recommendation': 'SELL',
                    'reason': stop_result.stop_type
                })
                continue
            
            # 計算特徵
            features_df = feature_calc.calculate_all_features(df, index_data)
            normalized = normalizer.normalize(features_df)
            
            feature_cols = normalizer.get_normalized_feature_columns()
            available_cols = [c for c in feature_cols if c in normalized.columns]
            
            obs = normalized.loc[target_date, available_cols].values.astype(np.float32)
            obs = np.nan_to_num(obs, nan=0.0)
            
            # 加入 SellReturn
            sell_return = current_price / buy_price
            obs = np.concatenate([obs, [sell_return]])
            
            # Sell Agent 判斷
            should_sell = False
            sell_prob = 0.0
            hold_prob = 1.0
            
            if sell_agent.model is not None:
                probs = sell_agent.predict_proba(obs)
                hold_prob = float(probs[0])
                sell_prob = float(probs[1])
                
                # 論文: |sell_prob - hold_prob| > 0.85
                if abs(sell_prob - hold_prob) > 0.85 and sell_prob > hold_prob:
                    should_sell = True
            
            signals.append({
                'symbol': symbol,
                'date': target_date,
                'price': current_price,
                'buy_price': buy_price,
                'return_pct': current_return,
                'holding_days': holding_days,
                'sell_prob': sell_prob,
                'hold_prob': hold_prob,
                'recommendation': 'SELL' if should_sell else 'HOLD',
                'reason': 'agent' if should_sell else 'hold'
            })
            
        except Exception as e:
            logger.warning(f"檢查 {symbol} 時錯誤: {e}")
            continue
    
    logger.info(f"找到 {len([s for s in signals if s['recommendation'] == 'SELL'])} 個賣出訊號")
    
    return signals


def save_signals(buy_signals: list, sell_signals: list, 
                 output_dir: str = 'outputs/signals/'):
    """儲存訊號結果"""
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y%m%d')
    
    # 儲存買入訊號
    if buy_signals:
        buy_df = pd.DataFrame(buy_signals)
        buy_file = os.path.join(output_dir, f'buy_signals_{date_str}.csv')
        buy_df.to_csv(buy_file, index=False)
        logger.info(f"買入訊號已儲存: {buy_file}")
    
    # 儲存賣出訊號
    if sell_signals:
        sell_df = pd.DataFrame(sell_signals)
        sell_file = os.path.join(output_dir, f'sell_signals_{date_str}.csv')
        sell_df.to_csv(sell_file, index=False)
        logger.info(f"賣出訊號已儲存: {sell_file}")


def print_summary(buy_signals: list, sell_signals: list, target_date: datetime):
    """印出摘要"""
    print("\n" + "=" * 60)
    print(f"Pro Trader RL 每日交易訊號 - {target_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    print("\n【買入建議】Top 10")
    print("-" * 60)
    if buy_signals:
        for i, sig in enumerate(buy_signals[:10], 1):
            print(f"{i:2d}. {sig['symbol']:6s} @ ${sig['price']:8.2f} | "
                  f"信心度: {sig['confidence']:.2%}")
    else:
        print("   無買入訊號")
    
    print("\n【持倉狀態】")
    print("-" * 60)
    if sell_signals:
        for sig in sell_signals:
            status = "🔴 賣出" if sig['recommendation'] == 'SELL' else "🟢 持有"
            print(f"   {sig['symbol']:6s} | 報酬: {sig['return_pct']:+7.2%} | "
                  f"天數: {sig['holding_days']:3d} | {status} ({sig['reason']})")
    else:
        print("   無持倉")
    
    print("=" * 60 + "\n")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Pro Trader RL 每日交易訊號')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='設定檔路徑')
    parser.add_argument('--date', type=str, default=None,
                       help='目標日期 (YYYY-MM-DD)')
    parser.add_argument('--positions', type=str, default=None,
                       help='持倉 CSV 檔案路徑')
    parser.add_argument('--output', type=str, default='outputs/signals/',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging()
    
    logger.info("=" * 50)
    logger.info("Pro Trader RL 每日交易訊號")
    logger.info(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    # 載入設定
    config_path = project_root / args.config
    config = load_config(str(config_path)) if config_path.exists() else {}
    
    # 決定目標日期
    if args.date:
        target_date = pd.to_datetime(args.date)
    else:
        target_date = pd.Timestamp.now().normalize()
    
    logger.info(f"目標日期: {target_date.strftime('%Y-%m-%d')}")
    
    # 初始化模組
    loader = DataLoader(config.get('data', {}))
    feature_calc = FeatureCalculator(config.get('features', {}))
    normalizer = DataNormalizer()
    donchian = DonchianChannel(period=config.get('features', {}).get('donchian_period', 20))
    stop_loss = StopLossRule(config.get('stop_loss', {}))
    
    # 載入 Agents
    buy_agent = BuyAgent(config.get('buy_agent', {}))
    sell_agent = SellAgent(config.get('sell_agent', {}))
    
    try:
        buy_agent.load_best_model()
        logger.info("Buy Agent 模型已載入")
    except:
        logger.warning("無法載入 Buy Agent 模型，將不進行過濾")
    
    try:
        sell_agent.load_best_model()
        logger.info("Sell Agent 模型已載入")
    except:
        logger.warning("無法載入 Sell Agent 模型")
    
    # 載入持倉
    positions = []
    if args.positions:
        positions = load_positions(args.positions)
    
    # 掃描買入訊號
    buy_signals = scan_buy_signals(
        loader, feature_calc, normalizer, buy_agent,
        donchian, target_date
    )
    
    # 檢查賣出條件
    sell_signals = check_sell_signals(
        loader, feature_calc, normalizer, sell_agent,
        stop_loss, positions, target_date
    )
    
    # 儲存結果
    save_signals(buy_signals, sell_signals, args.output)
    
    # 印出摘要
    print_summary(buy_signals, sell_signals, target_date)


if __name__ == '__main__':
    main()
