"""
PerformanceEvaluator - 績效評估器

根據論文計算各種績效指標:
- 年化報酬率
- 夏普比率 (Sharpe Ratio)
- 最大回撤 (Maximum Drawdown)
- 勝率與交易統計

使用方式:
    evaluator = PerformanceEvaluator()
    metrics = evaluator.calculate_all(equity_curve, trades)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class PerformanceMetrics:
    """績效指標"""
    # 報酬指標
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_return: float = 0.0
    
    # 風險指標
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    
    # 交易統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # 持有統計
    avg_holding_days: float = 0.0
    max_holding_days: int = 0
    
    def to_dict(self) -> dict:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_holding_days': self.avg_holding_days,
            'max_holding_days': self.max_holding_days
        }
    
    def __str__(self) -> str:
        return (f"績效摘要:\n"
                f"  總報酬: {self.total_return:.2%}\n"
                f"  年化報酬: {self.annualized_return:.2%}\n"
                f"  夏普比率: {self.sharpe_ratio:.2f}\n"
                f"  最大回撤: {self.max_drawdown:.2%}\n"
                f"  勝率: {self.win_rate:.2%}\n"
                f"  總交易: {self.total_trades} 筆")


class PerformanceEvaluator:
    """
    績效評估器
    
    計算各種績效指標，包括:
    - 報酬率 (總報酬、年化報酬)
    - 風險指標 (夏普比率、最大回撤、波動率)
    - 交易統計 (勝率、獲利因子)
    
    使用方式:
        evaluator = PerformanceEvaluator()
        metrics = evaluator.calculate_all(equity_curve, trades)
    """
    
    def __init__(self, risk_free_rate: float = 0.0, trading_days: int = 252):
        """
        初始化績效評估器
        
        Args:
            risk_free_rate: 無風險利率 (預設 0)
            trading_days: 年交易日數 (預設 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def calculate_returns(self, equity_curve: pd.Series) -> Dict:
        """
        計算報酬率指標
        
        Args:
            equity_curve: 權益曲線 (日期為索引)
            
        Returns:
            報酬率指標字典
        """
        if len(equity_curve) < 2:
            return {'total_return': 0, 'annualized_return': 0, 'cumulative_return': 0}
        
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        
        # 總報酬率
        total_return = (final_value - initial_value) / initial_value
        
        # 累積報酬率 (百分比)
        cumulative_return = total_return * 100
        
        # 年化報酬率
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25 if days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_return
        }
    
    def calculate_sharpe_ratio(self, equity_curve: pd.Series) -> float:
        """
        計算夏普比率
        
        Sharpe Ratio = (平均報酬 - 無風險利率) / 報酬標準差 * sqrt(252)
        
        Args:
            equity_curve: 權益曲線
            
        Returns:
            夏普比率
        """
        if len(equity_curve) < 2:
            return 0.0
        
        # 計算日報酬率
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0.0
        
        # 年化夏普比率
        excess_return = daily_returns.mean() - self.risk_free_rate / self.trading_days
        sharpe = excess_return / daily_returns.std() * np.sqrt(self.trading_days)
        
        return sharpe
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        計算最大回撤 (MDD)
        
        Args:
            equity_curve: 權益曲線
            
        Returns:
            最大回撤 (0 到 1 之間的數值)
        """
        if len(equity_curve) < 2:
            return 0.0
        
        # 計算累積最大值
        cummax = equity_curve.cummax()
        
        # 計算回撤
        drawdown = (cummax - equity_curve) / cummax
        
        return drawdown.max()
    
    def calculate_volatility(self, equity_curve: pd.Series) -> float:
        """
        計算年化波動率
        
        Args:
            equity_curve: 權益曲線
            
        Returns:
            年化波動率
        """
        if len(equity_curve) < 2:
            return 0.0
        
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return 0.0
        
        return daily_returns.std() * np.sqrt(self.trading_days)
    
    def calculate_trade_statistics(self, trades: List) -> Dict:
        """
        計算交易統計
        
        Args:
            trades: 交易列表 (需包含 return_pct 和 holding_days 屬性)
            
        Returns:
            交易統計字典
        """
        # 篩選已平倉交易
        closed_trades = [t for t in trades if hasattr(t, 'return_pct') and t.return_pct is not None]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_holding_days': 0,
                'max_holding_days': 0
            }
        
        # 分類交易
        winning = [t for t in closed_trades if t.return_pct > 0]
        losing = [t for t in closed_trades if t.return_pct <= 0]
        
        # 基本統計
        total_trades = len(closed_trades)
        winning_trades = len(winning)
        losing_trades = len(losing)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 平均獲利/虧損
        avg_win = np.mean([t.return_pct for t in winning]) if winning else 0
        avg_loss = np.mean([t.return_pct for t in losing]) if losing else 0
        
        # 獲利因子
        total_profit = sum(t.return_pct for t in winning) if winning else 0
        total_loss = abs(sum(t.return_pct for t in losing)) if losing else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 持有天數統計
        holding_days = [t.holding_days for t in closed_trades if t.holding_days is not None]
        avg_holding_days = np.mean(holding_days) if holding_days else 0
        max_holding_days = max(holding_days) if holding_days else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days,
            'max_holding_days': max_holding_days
        }
    
    def calculate_all(self, equity_curve: pd.Series, 
                      trades: List = None) -> PerformanceMetrics:
        """
        計算所有績效指標
        
        Args:
            equity_curve: 權益曲線
            trades: 交易列表 (可選)
            
        Returns:
            PerformanceMetrics 績效指標
        """
        metrics = PerformanceMetrics()
        
        # 報酬指標
        returns = self.calculate_returns(equity_curve)
        metrics.total_return = returns['total_return']
        metrics.annualized_return = returns['annualized_return']
        metrics.cumulative_return = returns['cumulative_return']
        
        # 風險指標
        metrics.sharpe_ratio = self.calculate_sharpe_ratio(equity_curve)
        metrics.max_drawdown = self.calculate_max_drawdown(equity_curve)
        metrics.volatility = self.calculate_volatility(equity_curve)
        
        # 交易統計
        if trades:
            trade_stats = self.calculate_trade_statistics(trades)
            metrics.total_trades = trade_stats['total_trades']
            metrics.winning_trades = trade_stats['winning_trades']
            metrics.losing_trades = trade_stats['losing_trades']
            metrics.win_rate = trade_stats['win_rate']
            metrics.avg_win = trade_stats['avg_win']
            metrics.avg_loss = trade_stats['avg_loss']
            metrics.profit_factor = trade_stats['profit_factor']
            metrics.avg_holding_days = trade_stats['avg_holding_days']
            metrics.max_holding_days = trade_stats['max_holding_days']
        
        return metrics
    
    def compare_with_benchmark(self, strategy_curve: pd.Series,
                               benchmark_curve: pd.Series) -> Dict:
        """
        與基準進行比較
        
        Args:
            strategy_curve: 策略權益曲線
            benchmark_curve: 基準權益曲線 (如大盤指數)
            
        Returns:
            比較結果
        """
        strategy_metrics = self.calculate_all(strategy_curve)
        benchmark_metrics = self.calculate_all(benchmark_curve)
        
        # 超額報酬 (Alpha)
        alpha = strategy_metrics.annualized_return - benchmark_metrics.annualized_return
        
        # 相對夏普
        sharpe_diff = strategy_metrics.sharpe_ratio - benchmark_metrics.sharpe_ratio
        
        # 相對回撤
        mdd_improvement = benchmark_metrics.max_drawdown - strategy_metrics.max_drawdown
        
        return {
            'strategy': strategy_metrics.to_dict(),
            'benchmark': benchmark_metrics.to_dict(),
            'alpha': alpha,
            'sharpe_improvement': sharpe_diff,
            'mdd_improvement': mdd_improvement,
            'outperformed': strategy_metrics.total_return > benchmark_metrics.total_return
        }
    
    def generate_summary(self, metrics: PerformanceMetrics) -> str:
        """產生績效摘要文字"""
        return str(metrics)


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    
    print("=== PerformanceEvaluator 測試 ===")
    
    # 建立模擬權益曲線
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    equity = 10000 * (1 + returns).cumprod()
    equity_curve = pd.Series(equity, index=dates)
    
    # 計算績效
    evaluator = PerformanceEvaluator()
    metrics = evaluator.calculate_all(equity_curve)
    
    print(metrics)
    print(f"\n詳細指標: {metrics.to_dict()}")
