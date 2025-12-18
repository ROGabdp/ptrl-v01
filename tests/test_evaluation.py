"""
Evaluation 模組單元測試

測試:
- PerformanceEvaluator: 績效指標計算
- Visualizer: 圖表生成
"""

import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

sys.path.insert(0, 'd:/000-github-repositories/ptrl-v01')

from src.evaluation.performance_evaluator import PerformanceEvaluator, PerformanceMetrics


# 模擬交易資料類別
@dataclass
class MockTrade:
    symbol: str
    return_pct: float
    holding_days: int


class TestPerformanceMetrics:
    """PerformanceMetrics 測試"""
    
    def test_metrics_creation(self):
        """測試指標建立"""
        metrics = PerformanceMetrics(
            total_return=0.25,
            annualized_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.10
        )
        
        assert metrics.total_return == 0.25
        assert metrics.sharpe_ratio == 1.5
    
    def test_metrics_to_dict(self):
        """測試轉換為字典"""
        metrics = PerformanceMetrics(total_return=0.25)
        data = metrics.to_dict()
        
        assert 'total_return' in data
        assert data['total_return'] == 0.25
    
    def test_metrics_str(self):
        """測試字串表示"""
        metrics = PerformanceMetrics(
            total_return=0.25,
            win_rate=0.6,
            total_trades=50
        )
        
        text = str(metrics)
        assert '績效摘要' in text
        assert '25.00%' in text


class TestPerformanceEvaluator:
    """PerformanceEvaluator 測試"""
    
    @pytest.fixture
    def evaluator(self):
        """建立評估器"""
        return PerformanceEvaluator()
    
    @pytest.fixture
    def sample_equity_curve(self):
        """建立樣本權益曲線"""
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        equity = 10000 * (1 + returns).cumprod()
        return pd.Series(equity, index=dates)
    
    @pytest.fixture
    def sample_trades(self):
        """建立樣本交易"""
        return [
            MockTrade('AAPL', 0.15, 30),
            MockTrade('MSFT', 0.08, 45),
            MockTrade('GOOGL', -0.05, 20),
            MockTrade('NVDA', 0.25, 60),
            MockTrade('TSLA', -0.10, 15),
        ]
    
    def test_calculate_returns(self, evaluator, sample_equity_curve):
        """測試報酬率計算"""
        returns = evaluator.calculate_returns(sample_equity_curve)
        
        assert 'total_return' in returns
        assert 'annualized_return' in returns
        assert isinstance(returns['total_return'], float)
    
    def test_calculate_sharpe_ratio(self, evaluator, sample_equity_curve):
        """測試夏普比率計算"""
        sharpe = evaluator.calculate_sharpe_ratio(sample_equity_curve)
        
        assert isinstance(sharpe, float)
        # 夏普比率通常在 -3 到 5 之間
        assert -5 < sharpe < 10
    
    def test_calculate_max_drawdown(self, evaluator, sample_equity_curve):
        """測試最大回撤計算"""
        mdd = evaluator.calculate_max_drawdown(sample_equity_curve)
        
        assert isinstance(mdd, float)
        assert 0 <= mdd <= 1  # MDD 應該在 0-100% 之間
    
    def test_calculate_volatility(self, evaluator, sample_equity_curve):
        """測試波動率計算"""
        vol = evaluator.calculate_volatility(sample_equity_curve)
        
        assert isinstance(vol, float)
        assert 0 <= vol <= 2  # 年化波動率通常小於 200%
    
    def test_calculate_trade_statistics(self, evaluator, sample_trades):
        """測試交易統計"""
        stats = evaluator.calculate_trade_statistics(sample_trades)
        
        assert stats['total_trades'] == 5
        assert stats['winning_trades'] == 3
        assert stats['losing_trades'] == 2
        assert stats['win_rate'] == pytest.approx(0.6, rel=1e-3)
    
    def test_calculate_all(self, evaluator, sample_equity_curve, sample_trades):
        """測試完整計算"""
        metrics = evaluator.calculate_all(sample_equity_curve, sample_trades)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 5
        assert metrics.sharpe_ratio is not None
    
    def test_empty_equity_curve(self, evaluator):
        """測試空權益曲線"""
        empty = pd.Series(dtype=float)
        returns = evaluator.calculate_returns(empty)
        
        assert returns['total_return'] == 0
    
    def test_empty_trades(self, evaluator):
        """測試空交易列表"""
        stats = evaluator.calculate_trade_statistics([])
        
        assert stats['total_trades'] == 0
        assert stats['win_rate'] == 0
    
    def test_compare_with_benchmark(self, evaluator, sample_equity_curve):
        """測試基準比較"""
        # 建立基準曲線 (較差表現)
        np.random.seed(0)
        benchmark_returns = np.random.normal(0.0005, 0.015, len(sample_equity_curve))
        benchmark = sample_equity_curve.iloc[0] * (1 + benchmark_returns).cumprod()
        benchmark = pd.Series(benchmark, index=sample_equity_curve.index)
        
        comparison = evaluator.compare_with_benchmark(sample_equity_curve, benchmark)
        
        assert 'strategy' in comparison
        assert 'benchmark' in comparison
        assert 'alpha' in comparison
        assert 'outperformed' in comparison


class TestSpecialCases:
    """特殊情況測試"""
    
    @pytest.fixture
    def evaluator(self):
        return PerformanceEvaluator()
    
    def test_constant_equity(self, evaluator):
        """測試恆定權益曲線"""
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        equity = pd.Series([10000] * 100, index=dates)
        
        sharpe = evaluator.calculate_sharpe_ratio(equity)
        mdd = evaluator.calculate_max_drawdown(equity)
        
        # 恆定曲線: 零報酬、零回撤
        assert sharpe == 0.0
        assert mdd == 0.0
    
    def test_monotonic_increasing(self, evaluator):
        """測試單調上升曲線"""
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        equity = pd.Series(10000 + np.arange(100) * 10, index=dates)
        
        returns = evaluator.calculate_returns(equity)
        mdd = evaluator.calculate_max_drawdown(equity)
        
        assert returns['total_return'] > 0
        assert mdd == pytest.approx(0.0, abs=1e-6)  # 無回撤
    
    def test_profit_factor(self, evaluator):
        """測試獲利因子計算"""
        trades = [
            MockTrade('A', 0.20, 10),  # 獲利 20%
            MockTrade('B', 0.30, 15),  # 獲利 30%
            MockTrade('C', -0.10, 5),  # 虧損 10%
        ]
        
        stats = evaluator.calculate_trade_statistics(trades)
        
        # 獲利因子 = 總獲利 / 總虧損 = (0.20+0.30) / 0.10 = 5.0
        assert stats['profit_factor'] == pytest.approx(5.0, rel=1e-3)


# =============================================================================
# 測試執行
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
