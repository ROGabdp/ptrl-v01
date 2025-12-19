"""
Pro Trader RL 回測腳本

執行流程:
1. 載入訓練好的模型
2. 設定回測期間
3. 執行回測
4. 產生績效報告

使用方式:
    python scripts/backtest.py --start 2017-10-16 --end 2023-10-15
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


def plot_results(result, output_path: str = 'outputs/reports/', config: dict = None, start_date=None, end_date=None):
    """繪製回測結果圖表 (含 S&P 500 基準比較)"""
    os.makedirs(output_path, exist_ok=True)
    
    # 計算 S&P 500 基準績效
    sp500_metrics = None
    sp500_equity = None
    
    if config:
        try:
            data_config = config.get('data', {})
            loader = DataLoader(data_config)
            index_df = loader.load_index()
            
            # 過濾期間
            start = pd.Timestamp(start_date) if start_date else result.equity_curve.index[0]
            end = pd.Timestamp(end_date) if end_date else result.equity_curve.index[-1]
            mask = (index_df.index >= start) & (index_df.index <= end)
            sp500_period = index_df.loc[mask].copy()
            
            if not sp500_period.empty:
                # 模擬 Buy & Hold
                initial_capital = result.initial_capital
                sp500_period['Return'] = sp500_period['Close'].pct_change().fillna(0)
                sp500_period['Equity'] = initial_capital * (1 + sp500_period['Return']).cumprod()
                sp500_equity = sp500_period['Equity']
                
                # 計算基準指標
                total_return = (sp500_equity.iloc[-1] - initial_capital) / initial_capital
                days = (sp500_equity.index[-1] - sp500_equity.index[0]).days
                annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
                
                # Sharpe Ratio (假設無風險利率 0)
                daily_std = sp500_period['Return'].std()
                sharpe_ratio = (sp500_period['Return'].mean() / daily_std) * (252 ** 0.5) if daily_std > 0 else 0
                
                # Max Drawdown
                cummax = sp500_equity.cummax()
                drawdown = (cummax - sp500_equity) / cummax
                max_drawdown = drawdown.max()
                
                sp500_metrics = {
                    'final_capital': sp500_equity.iloc[-1],
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                }
                logger.info("S&P 500 基準資料載入成功")
        except Exception as e:
            logger.warning(f"無法計算基準指標: {e}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 權益曲線
    ax1 = axes[0, 0]
    if len(result.equity_curve) > 0:
        result.equity_curve.plot(ax=ax1, linewidth=2, label='AI Trader', color='blue')
        
        # 繪製 S&P 500 基準線
        if sp500_equity is not None:
            sp500_equity.plot(ax=ax1, linewidth=1.5, label='S&P 500 (Buy & Hold)', color='gray', alpha=0.7, linestyle='--')
        
        ax1.set_title('Equity Curve Comparison')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 回撤
    ax2 = axes[0, 1]
    if len(result.equity_curve) > 0:
        cummax = result.equity_curve.cummax()
        drawdown = (cummax - result.equity_curve) / cummax * 100
        drawdown.plot(ax=ax2, color='red', linewidth=1, label='AI Drawdown')
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
    
    # AI 績效文字
    stats_text = "=== AI Trader Performance ===\n"
    stats_text += f"{'Final Capital:':<18} ${result.final_capital:,.0f}\n"
    stats_text += f"{'Total Return:':<18} {result.total_return:>8.2%}\n"
    stats_text += f"{'CAGR:':<18} {result.annualized_return:>8.2%}\n"
    stats_text += f"{'Sharpe Ratio:':<18} {result.sharpe_ratio:>8.2f}\n"
    stats_text += f"{'Max Drawdown:':<18} {result.max_drawdown:>8.2%}\n"
    
    if sp500_metrics:
        stats_text += "\n=== S&P 500 Benchmark ===\n"
        stats_text += f"{'Final Capital:':<18} ${sp500_metrics['final_capital']:,.0f}\n"
        stats_text += f"{'Total Return:':<18} {sp500_metrics['total_return']:>8.2%}\n"
        stats_text += f"{'CAGR:':<18} {sp500_metrics['annualized_return']:>8.2%}\n"
        stats_text += f"{'Sharpe Ratio:':<18} {sp500_metrics['sharpe_ratio']:>8.2f}\n"
        stats_text += f"{'Max Drawdown:':<18} {sp500_metrics['max_drawdown']:>8.2%}\n"

    stats_text += "\n=== Trade Stats ===\n"
    stats_text += f"{'Total Trades:':<18} {result.total_trades}\n"
    stats_text += f"{'Win Rate:':<18} {result.win_rate:>8.2%}\n"
    stats_text += f"{'Avg Holding:':<18} {result.avg_holding_days:.1f} days"

    ax4.text(0.05, 0.95, stats_text, fontsize=11, fontfamily='monospace',
             verticalalignment='top', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig_path = os.path.join(output_path, f'backtest_result_{timestamp}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"圖表已儲存至: {fig_path}")
    return fig_path, sp500_metrics


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
    
    # 繪製結果 (傳入 config 以計算基準)
    result = results['result']
    _, sp500_metrics = plot_results(result, args.output, config, args.start, args.end)
    
    # 輸出摘要
    print("\n" + "=" * 60)
    print(f"{'Performance Metric':<20} | {'AI Trader':>15} | {'S&P 500 (B&H)':>15}")
    print("-" * 60)
    print(f"{'Initial Capital':<20} | ${result.initial_capital:,.0f} | ${result.initial_capital:,.0f}")
    
    sp500_final = sp500_metrics['final_capital'] if sp500_metrics else 0
    print(f"{'Final Capital':<20} | ${result.final_capital:,.0f} | ${sp500_final:,.0f}")
    
    sp500_ret = sp500_metrics['total_return'] if sp500_metrics else 0
    print(f"{'Total Return':<20} | {result.total_return:>15.2%} | {sp500_ret:>15.2%}")
    
    sp500_ar = sp500_metrics['annualized_return'] if sp500_metrics else 0
    print(f"{'Annualized Return':<20} | {result.annualized_return:>15.2%} | {sp500_ar:>15.2%}")
    
    sp500_sr = sp500_metrics['sharpe_ratio'] if sp500_metrics else 0
    print(f"{'Sharpe Ratio':<20} | {result.sharpe_ratio:>15.2f} | {sp500_sr:>15.2f}")
    
    sp500_mdd = sp500_metrics['max_drawdown'] if sp500_metrics else 0
    print(f"{'Max Drawdown':<20} | {result.max_drawdown:>15.2%} | {sp500_mdd:>15.2%}")
    print("-" * 60)
    print(f"交易次數: {result.total_trades}")
    print(f"勝率: {result.win_rate:.2%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
