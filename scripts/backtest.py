"""
Pro Trader RL 回測腳本

執行流程:
1. 載入訓練好的模型
2. 設定回測期間
3. 執行回測
4. 產生績效報告

使用方式:
    python scripts/backtest.py --start 2022-01-01 --end 2023-12-31
    python scripts/backtest.py --config config/default_config.yaml
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from src.data import DataLoader, FeatureCalculator, DataNormalizer
from src.agents import BuyAgent, SellAgent
from src.rules import StopLossRule
from src.backtest import BacktestEngine


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_backtest(config: dict, start_date: str, end_date: str, 
                 symbols: list = None) -> dict:
    """
    執行回測
    
    Returns:
        回測結果字典
    """
    logger.info(f"開始回測: {start_date} ~ {end_date}")
    
    # 初始化模組
    data_config = config.get('data', {})
    loader = DataLoader(data_config)
    feature_calc = FeatureCalculator(config.get('features', {}))
    normalizer = DataNormalizer()
    stop_loss = StopLossRule(config.get('stop_loss', {}))
    
    # 載入 Agent
    buy_agent = BuyAgent(config.get('buy_agent', {}))
    sell_agent = SellAgent(config.get('sell_agent', {}))
    
    try:
        buy_agent.load_best_model()
        logger.info("Buy Agent 模型載入成功")
    except Exception as e:
        logger.warning(f"無法載入 Buy Agent: {e}")
    
    try:
        sell_agent.load_best_model()
        logger.info("Sell Agent 模型載入成功")
    except Exception as e:
        logger.warning(f"無法載入 Sell Agent: {e}")
    
    # 初始化回測引擎
    backtest_config = config.get('backtest', {})
    engine = BacktestEngine(backtest_config)
    
    # 設定模組
    engine.set_modules(
        data_loader=loader,
        feature_calculator=feature_calc,
        normalizer=normalizer,
        buy_agent=buy_agent,
        sell_agent=sell_agent,
        stop_loss_rule=stop_loss
    )
    
    # 執行回測
    result = engine.run(start_date, end_date, symbols)
    
    # 產生報告
    summary = engine.generate_report(result)
    
    return {
        'result': result,
        'summary': summary
    }


def plot_results(result, output_path: str = 'outputs/reports/'):
    """繪製回測結果圖表"""
    os.makedirs(output_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 權益曲線
    ax1 = axes[0, 0]
    if len(result.equity_curve) > 0:
        result.equity_curve.plot(ax=ax1, linewidth=1.5)
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
    
    # 2. 回撤
    ax2 = axes[0, 1]
    if len(result.equity_curve) > 0:
        cummax = result.equity_curve.cummax()
        drawdown = (cummax - result.equity_curve) / cummax * 100
        drawdown.plot(ax=ax2, color='red', linewidth=1)
        ax2.fill_between(drawdown.index, 0, drawdown, alpha=0.3, color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
    
    # 3. 月度報酬
    ax3 = axes[1, 0]
    if len(result.daily_returns) > 0:
        monthly_returns = result.daily_returns.resample('ME').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        colors = ['green' if x >= 0 else 'red' for x in monthly_returns]
        monthly_returns.plot(kind='bar', ax=ax3, color=colors, alpha=0.7)
        ax3.set_title('Monthly Returns')
        ax3.set_ylabel('Return (%)')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. 績效統計
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    Performance Summary
    ─────────────────────
    Total Return:       {result.total_return:.2%}
    Annualized Return:  {result.annualized_return:.2%}
    Sharpe Ratio:       {result.sharpe_ratio:.2f}
    Max Drawdown:       {result.max_drawdown:.2%}
    
    Trade Statistics
    ─────────────────────
    Total Trades:       {result.total_trades}
    Win Rate:           {result.win_rate:.2%}
    Avg Holding Days:   {result.avg_holding_days:.1f}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=12, fontfamily='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_path = os.path.join(output_path, f'backtest_result_{timestamp}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"圖表已儲存至: {fig_path}")
    return fig_path


def main():
    parser = argparse.ArgumentParser(description='Pro Trader RL 回測')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--start', type=str, default='2022-01-01')
    parser.add_argument('--end', type=str, default='2023-12-31')
    parser.add_argument('--symbols', type=str, nargs='+', default=None)
    parser.add_argument('--output', type=str, default='outputs/reports/')
    args = parser.parse_args()
    
    # 載入設定
    config_path = project_root / args.config
    config = load_config(str(config_path))
    
    # 執行回測
    results = run_backtest(config, args.start, args.end, args.symbols)
    
    # 繪製結果
    result = results['result']
    plot_results(result, args.output)
    
    # 輸出摘要
    print("\n" + "=" * 50)
    print("回測結果摘要")
    print("=" * 50)
    print(f"期間: {args.start} ~ {args.end}")
    print(f"初始資金: ${result.initial_capital:,.0f}")
    print(f"最終資金: ${result.final_capital:,.0f}")
    print(f"總報酬率: {result.total_return:.2%}")
    print(f"年化報酬: {result.annualized_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown:.2%}")
    print(f"交易次數: {result.total_trades}")
    print(f"勝率: {result.win_rate:.2%}")
    print("=" * 50)


if __name__ == '__main__':
    main()
