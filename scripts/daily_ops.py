"""
Pro Trader RL 日常運營腳本

每日執行流程:
1. 更新股票資料
2. 計算當日特徵
3. 掃描買入訊號並過濾
4. 檢查持倉的賣出條件
5. 產生交易建議報告

使用方式:
    python scripts/daily_ops.py
    python scripts/daily_ops.py --date 2024-01-15
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# 加入專案根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml
from loguru import logger

from src.data import DataLoader, FeatureCalculator, DataNormalizer
from src.agents import BuyAgent, SellAgent
from src.rules import DonchianChannel, StopLossRule


def load_config(config_path: str) -> dict:
    """載入設定檔"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def update_data(loader: DataLoader, symbols: list) -> int:
    """更新股票資料"""
    logger.info("開始更新股票資料...")
    
    updated_count = 0
    for symbol in symbols:
        try:
            result = loader.update_symbol(symbol)
            if result:
                updated_count += 1
        except Exception as e:
            logger.warning(f"更新 {symbol} 失敗: {e}")
    
    # 更新指數
    try:
        loader.update_symbol(loader.index_symbol)
    except Exception as e:
        logger.warning(f"更新指數失敗: {e}")
    
    logger.info(f"資料更新完成: {updated_count}/{len(symbols)} 支股票")
    return updated_count


def scan_buy_signals(config: dict, loader: DataLoader, 
                      buy_agent: BuyAgent, target_date: datetime) -> list:
    """
    掃描買入訊號
    
    Returns:
        買入訊號列表 [(symbol, confidence, features), ...]
    """
    logger.info(f"掃描買入訊號: {target_date.strftime('%Y-%m-%d')}")
    
    signals = []
    symbols = loader.load_symbols_list()
    index_data = loader.load_index()
    
    feature_calc = FeatureCalculator(config.get('features', {}))
    normalizer = DataNormalizer()
    donchian = DonchianChannel(period=config.get('features', {}).get('donchian_period', 20))
    
    for symbol in symbols:
        df = loader.load_symbol(symbol)
        if df is None or target_date not in df.index:
            continue
        
        try:
            # 計算特徵
            df_features = feature_calc.calculate_all_features(df, index_data)
            df_normalized = normalizer.normalize(df_features)
            
            # 檢查 Donchian 突破
            idx = df_normalized.index.get_loc(target_date)
            if idx < 20:
                continue
            
            high = df_normalized.iloc[idx]['High']
            donchian_upper = df_normalized.iloc[idx-20:idx]['High'].max()
            
            if high > donchian_upper:
                # 取得正規化特徵
                feature_cols = normalizer.get_normalized_feature_columns()
                available_cols = [c for c in feature_cols if c in df_normalized.columns]
                obs = df_normalized.loc[target_date, available_cols].values.astype(np.float32)
                obs = np.nan_to_num(obs, nan=0.0)
                
                # Buy Agent 過濾
                if buy_agent and buy_agent.model:
                    action = buy_agent.predict(obs)
                    probs = buy_agent.predict_proba(obs)
                    confidence = probs[1]  # 買入機率
                    
                    if action == 1:
                        signals.append({
                            'symbol': symbol,
                            'confidence': float(confidence),
                            'price': df_normalized.iloc[idx]['Close'],
                            'donchian_upper': donchian_upper
                        })
                else:
                    # 無 Agent 時直接輸出訊號
                    signals.append({
                        'symbol': symbol,
                        'confidence': None,
                        'price': df_normalized.iloc[idx]['Close'],
                        'donchian_upper': donchian_upper
                    })
        
        except Exception as e:
            logger.debug(f"處理 {symbol} 時錯誤: {e}")
            continue
    
    # 按信心度排序
    signals.sort(key=lambda x: x['confidence'] or 0, reverse=True)
    
    logger.info(f"找到 {len(signals)} 個買入訊號")
    return signals


def check_sell_signals(config: dict, loader: DataLoader, 
                        sell_agent: SellAgent, positions: list,
                        target_date: datetime) -> list:
    """
    檢查持倉的賣出條件
    
    Args:
        positions: 持倉列表 [(symbol, buy_price, buy_date), ...]
    
    Returns:
        賣出建議列表
    """
    logger.info(f"檢查賣出訊號: {len(positions)} 個持倉")
    
    sell_signals = []
    stop_loss = StopLossRule(config.get('stop_loss', {}))
    index_data = loader.load_index()
    
    feature_calc = FeatureCalculator(config.get('features', {}))
    normalizer = DataNormalizer()
    
    for pos in positions:
        symbol = pos['symbol']
        buy_price = pos['buy_price']
        buy_date = pd.to_datetime(pos['buy_date'])
        
        df = loader.load_symbol(symbol)
        if df is None or target_date not in df.index:
            continue
        
        try:
            current_price = df.loc[target_date, 'Close']
            holding_days = (target_date - buy_date).days
            
            # 取得價格歷史
            buy_idx = df.index.get_loc(buy_date) if buy_date in df.index else 0
            current_idx = df.index.get_loc(target_date)
            price_history = df.iloc[buy_idx:current_idx+1]['Close']
            
            # 1. 檢查停損
            stop_result = stop_loss.check(
                buy_price=buy_price,
                current_price=current_price,
                holding_days=holding_days,
                price_history=price_history
            )
            
            if stop_result.should_stop:
                sell_signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': stop_result.stop_type,
                    'current_return': stop_result.current_return,
                    'holding_days': holding_days
                })
                continue
            
            # 2. Sell Agent 判斷
            if sell_agent and sell_agent.model:
                df_features = feature_calc.calculate_all_features(df, index_data)
                df_normalized = normalizer.normalize(df_features)
                
                feature_cols = normalizer.get_normalized_feature_columns()
                available_cols = [c for c in feature_cols if c in df_normalized.columns]
                features = df_normalized.loc[target_date, available_cols].values.astype(np.float32)
                
                # 加入 SellReturn
                sell_return = current_price / buy_price
                obs = np.concatenate([np.nan_to_num(features, nan=0.0), [sell_return]])
                
                action, should_sell = sell_agent.predict_with_threshold(obs)
                
                if should_sell:
                    sell_signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'reason': 'agent',
                        'current_return': (current_price - buy_price) / buy_price,
                        'holding_days': holding_days
                    })
                else:
                    sell_signals.append({
                        'symbol': symbol,
                        'action': 'HOLD',
                        'reason': 'agent',
                        'current_return': (current_price - buy_price) / buy_price,
                        'holding_days': holding_days
                    })
        
        except Exception as e:
            logger.warning(f"處理 {symbol} 賣出檢查時錯誤: {e}")
            continue
    
    return sell_signals


def generate_report(buy_signals: list, sell_signals: list, 
                    target_date: datetime, output_dir: str = 'outputs/daily/'):
    """產生每日交易建議報告"""
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = target_date.strftime('%Y%m%d')
    
    # 買入建議
    if buy_signals:
        buy_df = pd.DataFrame(buy_signals)
        buy_df.to_csv(os.path.join(output_dir, f'buy_signals_{date_str}.csv'), index=False)
    
    # 賣出建議
    if sell_signals:
        sell_df = pd.DataFrame(sell_signals)
        sell_df.to_csv(os.path.join(output_dir, f'sell_signals_{date_str}.csv'), index=False)
    
    # 文字報告
    report_path = os.path.join(output_dir, f'daily_report_{date_str}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Pro Trader RL 每日報告\n")
        f.write(f"日期: {target_date.strftime('%Y-%m-%d')}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"買入訊號 ({len(buy_signals)} 個):\n")
        f.write("-" * 30 + "\n")
        for sig in buy_signals[:10]:
            f.write(f"  {sig['symbol']}: 價格 ${sig['price']:.2f}, 信心度 {sig.get('confidence', 'N/A')}\n")
        
        f.write(f"\n賣出建議 ({len([s for s in sell_signals if s['action'] == 'SELL'])} 個):\n")
        f.write("-" * 30 + "\n")
        for sig in sell_signals:
            if sig['action'] == 'SELL':
                f.write(f"  {sig['symbol']}: 報酬 {sig['current_return']:.2%}, "
                       f"原因: {sig['reason']}, 持有 {sig['holding_days']} 天\n")
    
    logger.info(f"報告已儲存至: {output_dir}")
    return report_path


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='Pro Trader RL 日常運營')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--date', type=str, default=None, help='指定日期 (YYYY-MM-DD)')
    parser.add_argument('--no-update', action='store_true', help='不更新資料')
    parser.add_argument('--positions', type=str, default=None, help='持倉檔案路徑')
    args = parser.parse_args()
    
    # 載入設定
    config_path = project_root / args.config
    config = load_config(str(config_path))
    
    # 初始化
    loader = DataLoader(config.get('data', {}))
    
    # 目標日期
    if args.date:
        target_date = pd.to_datetime(args.date)
    else:
        target_date = pd.Timestamp.now().normalize()
    
    logger.info(f"執行日期: {target_date.strftime('%Y-%m-%d')}")
    
    # 更新資料
    if not args.no_update:
        symbols = loader.load_symbols_list() or loader.get_sp500_symbols()
        update_data(loader, symbols)
    
    # 載入 Agent
    buy_agent = BuyAgent(config.get('buy_agent', {}))
    sell_agent = SellAgent(config.get('sell_agent', {}))
    
    try:
        buy_agent.load_best_model()
    except:
        logger.warning("無法載入 Buy Agent 模型")
    
    try:
        sell_agent.load_best_model()
    except:
        logger.warning("無法載入 Sell Agent 模型")
    
    # 掃描買入訊號
    buy_signals = scan_buy_signals(config, loader, buy_agent, target_date)
    
    # 載入/檢查持倉
    positions = []
    if args.positions and os.path.exists(args.positions):
        positions = pd.read_csv(args.positions).to_dict('records')
    
    # 檢查賣出條件
    sell_signals = check_sell_signals(config, loader, sell_agent, positions, target_date)
    
    # 產生報告
    report_path = generate_report(buy_signals, sell_signals, target_date)
    
    # 輸出摘要
    print("\n" + "=" * 50)
    print("每日運營摘要")
    print("=" * 50)
    print(f"日期: {target_date.strftime('%Y-%m-%d')}")
    print(f"買入訊號: {len(buy_signals)} 個")
    print(f"賣出建議: {len([s for s in sell_signals if s['action'] == 'SELL'])} 個")
    print(f"報告路徑: {report_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()
