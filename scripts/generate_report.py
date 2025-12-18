#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交易紀錄與報告產生腳本

功能:
- 讀取交易紀錄
- 計算績效指標
- 產生視覺化報告
- 匯出 CSV/PDF 報告

使用方式:
    python scripts/generate_report.py
    python scripts/generate_report.py --trades trades.csv
    python scripts/generate_report.py --portfolio portfolio_state.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# 設定專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger

from src.trading import PortfolioManager
from src.evaluation import PerformanceEvaluator, Visualizer


def load_config(config_path: str) -> dict:
    """載入 YAML 設定檔"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str = 'logs/daily_ops/'):
    """設定日誌"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"report_{datetime.now().strftime('%Y%m%d')}.log")
    
    logger.add(
        log_file,
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )


def load_trades_from_csv(filepath: str) -> list:
    """從 CSV 載入交易紀錄"""
    if not os.path.exists(filepath):
        logger.warning(f"交易紀錄檔案不存在: {filepath}")
        return []
    
    try:
        df = pd.read_csv(filepath)
        
        from dataclasses import dataclass
        
        @dataclass
        class Trade:
            symbol: str
            action: str
            date: datetime
            price: float
            shares: int
            fee: float
            total_value: float
            reason: str = None
            return_pct: float = None
            holding_days: int = None
        
        trades = []
        for _, row in df.iterrows():
            trades.append(Trade(
                symbol=row['symbol'],
                action=row['action'],
                date=pd.to_datetime(row['date']),
                price=row['price'],
                shares=row.get('shares', 1),
                fee=row.get('fee', 0),
                total_value=row.get('total_value', row['price']),
                reason=row.get('reason'),
                return_pct=row.get('return_pct'),
                holding_days=row.get('holding_days')
            ))
        
        logger.info(f"載入 {len(trades)} 筆交易紀錄")
        return trades
        
    except Exception as e:
        logger.error(f"載入交易紀錄失敗: {e}")
        return []


def load_equity_curve(filepath: str) -> pd.Series:
    """從 CSV 載入權益曲線"""
    if not os.path.exists(filepath):
        return pd.Series(dtype=float)
    
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df['equity']
        
    except Exception as e:
        logger.error(f"載入權益曲線失敗: {e}")
        return pd.Series(dtype=float)


def load_portfolio_state(filepath: str) -> dict:
    """載入投資組合狀態"""
    if not os.path.exists(filepath):
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"載入投資組合狀態失敗: {e}")
        return {}


def calculate_metrics(trades: list, equity_curve: pd.Series) -> dict:
    """計算績效指標"""
    evaluator = PerformanceEvaluator()
    
    if len(equity_curve) > 0:
        metrics = evaluator.calculate_all(equity_curve, trades)
    else:
        # 從交易計算簡單指標
        sell_trades = [t for t in trades if t.action == 'SELL']
        if not sell_trades:
            return {}
        
        returns = [t.return_pct for t in sell_trades if t.return_pct is not None]
        winning = [r for r in returns if r > 0]
        
        metrics = type('Metrics', (), {
            'to_dict': lambda self: {
                'total_trades': len(sell_trades),
                'winning_trades': len(winning),
                'win_rate': len(winning) / len(returns) if returns else 0,
                'avg_return': np.mean(returns) if returns else 0,
                'total_return': np.sum(returns) if returns else 0
            }
        })()
    
    return metrics.to_dict()


def generate_text_report(trades: list, metrics: dict, 
                         positions: list = None) -> str:
    """產生文字報告"""
    lines = [
        "=" * 60,
        "Pro Trader RL 交易報告",
        f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "【績效摘要】",
        "-" * 60
    ]
    
    if metrics:
        lines.append(f"總報酬率: {metrics.get('total_return', 0):.2%}")
        lines.append(f"年化報酬: {metrics.get('annualized_return', 0):.2%}")
        lines.append(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        lines.append(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        lines.append(f"勝率: {metrics.get('win_rate', 0):.2%}")
        lines.append(f"總交易: {metrics.get('total_trades', 0)} 筆")
    else:
        lines.append("無績效資料")
    
    lines.append("")
    lines.append("【現有持倉】")
    lines.append("-" * 60)
    
    if positions:
        for pos in positions:
            lines.append(f"  {pos.get('symbol', 'N/A'):6s} | "
                        f"買入: ${pos.get('buy_price', 0):.2f} | "
                        f"股數: {pos.get('shares', 0)}")
    else:
        lines.append("  無持倉")
    
    lines.append("")
    lines.append("【近期交易】(最近 10 筆)")
    lines.append("-" * 60)
    
    recent_trades = sorted(trades, key=lambda x: x.date, reverse=True)[:10]
    for trade in recent_trades:
        action_icon = "📈" if trade.action == "BUY" else "📉"
        ret_str = f"{trade.return_pct:+.2%}" if trade.return_pct else "N/A"
        lines.append(f"  {action_icon} {trade.symbol:6s} {trade.action:4s} @ ${trade.price:.2f} | "
                    f"報酬: {ret_str}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def generate_visual_report(trades: list, equity_curve: pd.Series,
                           metrics: dict, output_dir: str) -> list:
    """產生視覺化報告"""
    viz = Visualizer()
    generated_files = []
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 權益曲線圖
    if len(equity_curve) > 0:
        try:
            fig = viz.plot_equity_curve(equity_curve, title='Pro Trader RL Equity Curve')
            filepath = os.path.join(output_dir, f'equity_curve_{timestamp}.png')
            viz.save_figure(fig, f'equity_curve_{timestamp}.png', output_dir)
            generated_files.append(filepath)
        except Exception as e:
            logger.warning(f"產生權益曲線圖失敗: {e}")
    
    # 月度報酬圖
    if len(equity_curve) > 30:
        try:
            fig = viz.plot_monthly_returns(equity_curve, title='Monthly Returns')
            filepath = os.path.join(output_dir, f'monthly_returns_{timestamp}.png')
            viz.save_figure(fig, f'monthly_returns_{timestamp}.png', output_dir)
            generated_files.append(filepath)
        except Exception as e:
            logger.warning(f"產生月度報酬圖失敗: {e}")
    
    # 交易分布圖
    if trades:
        try:
            fig = viz.plot_trade_distribution(trades, title='Trade Distribution')
            filepath = os.path.join(output_dir, f'trade_distribution_{timestamp}.png')
            viz.save_figure(fig, f'trade_distribution_{timestamp}.png', output_dir)
            generated_files.append(filepath)
        except Exception as e:
            logger.warning(f"產生交易分布圖失敗: {e}")
    
    # 完整報告
    if len(equity_curve) > 0 and trades and metrics:
        try:
            filepath = viz.create_backtest_report(equity_curve, trades, metrics, None, output_dir)
            generated_files.append(filepath)
        except Exception as e:
            logger.warning(f"產生完整報告失敗: {e}")
    
    return generated_files


def export_summary_csv(trades: list, metrics: dict, output_dir: str):
    """匯出摘要 CSV"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 指標摘要
    if metrics:
        metrics_df = pd.DataFrame([metrics])
        metrics_file = os.path.join(output_dir, f'metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"指標已匯出: {metrics_file}")
    
    # 交易明細
    if trades:
        trades_data = []
        for t in trades:
            trades_data.append({
                'symbol': t.symbol,
                'action': t.action,
                'date': t.date,
                'price': t.price,
                'shares': t.shares,
                'return_pct': t.return_pct,
                'holding_days': t.holding_days,
                'reason': t.reason
            })
        
        trades_df = pd.DataFrame(trades_data)
        trades_file = os.path.join(output_dir, f'trades_{timestamp}.csv')
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"交易明細已匯出: {trades_file}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Pro Trader RL 報告產生器')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='設定檔路徑')
    parser.add_argument('--trades', type=str, default=None,
                       help='交易紀錄 CSV 檔案')
    parser.add_argument('--equity', type=str, default=None,
                       help='權益曲線 CSV 檔案')
    parser.add_argument('--portfolio', type=str, default=None,
                       help='投資組合狀態 JSON 檔案')
    parser.add_argument('--output', type=str, default='outputs/reports/',
                       help='輸出目錄')
    parser.add_argument('--no-visual', action='store_true',
                       help='不產生視覺化報告')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging()
    
    logger.info("=" * 50)
    logger.info("Pro Trader RL 報告產生器")
    logger.info(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    os.makedirs(args.output, exist_ok=True)
    
    # 載入資料
    trades = []
    equity_curve = pd.Series(dtype=float)
    positions = []
    
    if args.trades:
        trades = load_trades_from_csv(args.trades)
    
    if args.equity:
        equity_curve = load_equity_curve(args.equity)
    
    if args.portfolio:
        state = load_portfolio_state(args.portfolio)
        if 'positions' in state:
            positions = list(state['positions'].values())
        if 'trade_history' in state:
            from dataclasses import dataclass
            
            @dataclass
            class Trade:
                symbol: str
                action: str
                date: datetime
                price: float
                shares: int
                fee: float
                total_value: float
                reason: str = None
                return_pct: float = None
                holding_days: int = None
            
            for t_dict in state['trade_history']:
                trades.append(Trade(
                    symbol=t_dict['symbol'],
                    action=t_dict['action'],
                    date=pd.to_datetime(t_dict['date']),
                    price=t_dict['price'],
                    shares=t_dict['shares'],
                    fee=t_dict['fee'],
                    total_value=t_dict['total_value'],
                    reason=t_dict.get('reason'),
                    return_pct=t_dict.get('return_pct'),
                    holding_days=t_dict.get('holding_days')
                ))
    
    if not trades and len(equity_curve) == 0:
        logger.warning("無資料可產生報告")
        print("請提供 --trades 或 --equity 或 --portfolio 參數")
        return
    
    # 計算指標
    metrics = calculate_metrics(trades, equity_curve)
    
    # 產生文字報告
    text_report = generate_text_report(trades, metrics, positions)
    print(text_report)
    
    # 儲存文字報告
    report_file = os.path.join(args.output, 
                               f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(text_report)
    logger.info(f"文字報告已儲存: {report_file}")
    
    # 匯出 CSV
    export_summary_csv(trades, metrics, args.output)
    
    # 產生視覺化報告
    if not args.no_visual:
        generated = generate_visual_report(trades, equity_curve, metrics, args.output)
        if generated:
            print(f"\n視覺化報告已產生: {len(generated)} 個檔案")
            for f in generated:
                print(f"  - {f}")
    
    print(f"\n報告已儲存至: {args.output}")


if __name__ == '__main__':
    main()
