"""
Trading 模組單元測試

測試:
- PortfolioManager: 持倉管理、交易執行
- TradeExecutor: Agent 整合、停損檢查
"""

import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, 'd:/000-github-repositories/ptrl-v01')

from src.trading.portfolio_manager import PortfolioManager, Position, Trade
from src.trading.trade_executor import TradeExecutor


class TestPosition:
    """Position 類別測試"""
    
    def test_position_creation(self):
        """測試持倉建立"""
        pos = Position(
            symbol='AAPL',
            buy_date=datetime(2023, 1, 1),
            buy_price=150.0,
            shares=10,
            cost_basis=1505.0
        )
        
        assert pos.symbol == 'AAPL'
        assert pos.buy_price == 150.0
        assert pos.shares == 10
    
    def test_position_return(self):
        """測試報酬率計算"""
        pos = Position(
            symbol='AAPL',
            buy_date=datetime(2023, 1, 1),
            buy_price=100.0,
            shares=10,
            cost_basis=1001.0
        )
        
        # 上漲 20%
        assert pos.get_return(120.0) == pytest.approx(0.2, rel=1e-3)
        
        # 下跌 10%
        assert pos.get_return(90.0) == pytest.approx(-0.1, rel=1e-3)
    
    def test_position_holding_days(self):
        """測試持有天數計算"""
        pos = Position(
            symbol='AAPL',
            buy_date=datetime(2023, 1, 1),
            buy_price=100.0,
            shares=10,
            cost_basis=1001.0
        )
        
        current_date = datetime(2023, 4, 1)  # 90 天後
        holding = pos.get_holding_days(current_date)
        
        assert holding == 90
    
    def test_position_serialization(self):
        """測試序列化與反序列化"""
        pos = Position(
            symbol='AAPL',
            buy_date=datetime(2023, 1, 1),
            buy_price=150.0,
            shares=10,
            cost_basis=1505.0
        )
        
        # 序列化
        data = pos.to_dict()
        assert 'symbol' in data
        assert 'buy_date' in data
        
        # 反序列化
        pos2 = Position.from_dict(data)
        assert pos2.symbol == pos.symbol
        assert pos2.buy_price == pos.buy_price


class TestPortfolioManager:
    """PortfolioManager 類別測試"""
    
    @pytest.fixture
    def portfolio(self):
        """建立預設投資組合"""
        return PortfolioManager({
            'initial_capital': 10000,
            'max_positions': 10,
            'max_position_pct': 0.10,
            'trading_fee': 0.001
        })
    
    def test_initial_state(self, portfolio):
        """測試初始狀態"""
        assert portfolio.cash == 10000
        assert portfolio.initial_capital == 10000
        assert portfolio.max_positions == 10
        assert portfolio.get_position_count() == 0
    
    def test_open_position(self, portfolio):
        """測試開倉"""
        trade = portfolio.open_position('AAPL', 100.0, datetime(2023, 1, 1))
        
        assert trade is not None
        assert trade.symbol == 'AAPL'
        assert trade.action == 'BUY'
        assert portfolio.get_position_count() == 1
        assert portfolio.cash < 10000
    
    def test_open_position_respects_max_position_pct(self, portfolio):
        """測試倉位大小限制"""
        trade = portfolio.open_position('AAPL', 100.0, datetime(2023, 1, 1))
        
        # 預期投資金額為初始資金的 10% = $1000
        # 100元/股，買 10 股 = $1000
        position = portfolio.get_position('AAPL')
        
        # 考慮可能的資金調整
        assert position.shares * 100 <= 1100  # 容許一些誤差
    
    def test_cannot_open_duplicate_position(self, portfolio):
        """測試無法重複開倉"""
        portfolio.open_position('AAPL', 100.0, datetime(2023, 1, 1))
        trade2 = portfolio.open_position('AAPL', 110.0, datetime(2023, 1, 2))
        
        assert trade2 is None
        assert portfolio.get_position_count() == 1
    
    def test_max_positions_limit(self, portfolio):
        """測試最大持倉數限制"""
        # 建立一個小投資組合
        pm = PortfolioManager({
            'initial_capital': 10000,
            'max_positions': 2,
            'max_position_pct': 0.40
        })
        
        pm.open_position('AAPL', 100.0, datetime(2023, 1, 1))
        pm.open_position('MSFT', 100.0, datetime(2023, 1, 1))
        trade3 = pm.open_position('GOOGL', 100.0, datetime(2023, 1, 1))
        
        assert trade3 is None
        assert pm.get_position_count() == 2
    
    def test_close_position(self, portfolio):
        """測試平倉"""
        portfolio.open_position('AAPL', 100.0, datetime(2023, 1, 1))
        
        trade = portfolio.close_position('AAPL', 120.0, datetime(2023, 3, 1), 'agent')
        
        assert trade is not None
        assert trade.action == 'SELL'
        assert trade.return_pct is not None
        assert trade.return_pct == pytest.approx(0.2, rel=1e-2)
        assert portfolio.get_position_count() == 0
    
    def test_cannot_close_nonexistent_position(self, portfolio):
        """測試無法平倉不存在的持倉"""
        trade = portfolio.close_position('AAPL', 100.0, datetime(2023, 1, 1), 'agent')
        
        assert trade is None
    
    def test_equity_calculation(self, portfolio):
        """測試權益計算"""
        initial_equity = portfolio.get_equity({})
        assert initial_equity == 10000
        
        portfolio.open_position('AAPL', 100.0, datetime(2023, 1, 1))
        
        # 假設價格上漲到 120
        equity_after = portfolio.get_equity({'AAPL': 120.0})
        
        # 權益應該增加 (考慮手續費)
        position = portfolio.get_position('AAPL')
        expected = portfolio.cash + 120.0 * position.shares
        
        assert equity_after == pytest.approx(expected, rel=1e-3)
    
    def test_statistics(self, portfolio):
        """測試統計功能"""
        portfolio.open_position('AAPL', 100.0, datetime(2023, 1, 1))
        portfolio.close_position('AAPL', 120.0, datetime(2023, 3, 1), 'agent')
        
        stats = portfolio.get_statistics()
        
        assert stats['total_trades'] == 2  # 買+賣
        assert stats['buy_trades'] == 1
        assert stats['sell_trades'] == 1
        assert stats['winning_trades'] == 1
    
    def test_reset(self, portfolio):
        """測試重置功能"""
        portfolio.open_position('AAPL', 100.0, datetime(2023, 1, 1))
        portfolio.reset()
        
        assert portfolio.cash == 10000
        assert portfolio.get_position_count() == 0
        assert len(portfolio.trade_history) == 0


class TestTradeExecutor:
    """TradeExecutor 類別測試"""
    
    @pytest.fixture
    def executor(self):
        """建立預設執行器"""
        pm = PortfolioManager({'initial_capital': 10000})
        return TradeExecutor(pm)
    
    def test_initialization(self, executor):
        """測試初始化"""
        assert executor.portfolio is not None
        assert executor.buy_agent is None
        assert executor.sell_agent is None
    
    def test_evaluate_buy_signal_without_agent(self, executor):
        """無 Agent 時評估買入訊號"""
        features = np.random.randn(69).astype(np.float32)
        should_buy, confidence = executor.evaluate_buy_signal('AAPL', features, 100.0)
        
        # 無 Agent 時應該通過
        assert should_buy is True
        assert confidence == 1.0
    
    def test_add_buy_candidate(self, executor):
        """測試加入買入候選"""
        features = np.random.randn(69).astype(np.float32)
        
        executor.add_buy_candidate('AAPL', features, 100.0, datetime(2023, 1, 1))
        executor.add_buy_candidate('MSFT', features, 200.0, datetime(2023, 1, 1))
        
        assert len(executor.daily_buy_candidates) == 2
    
    def test_execute_daily_buys(self, executor):
        """測試執行每日買入"""
        features = np.random.randn(69).astype(np.float32)
        
        executor.add_buy_candidate('AAPL', features, 100.0, datetime(2023, 1, 1))
        executor.add_buy_candidate('MSFT', features, 200.0, datetime(2023, 1, 1))
        
        trades = executor.execute_daily_buys()
        
        assert len(trades) == 2
        assert executor.portfolio.get_position_count() == 2
        assert len(executor.daily_buy_candidates) == 0  # 已清空
    
    def test_process_buy_signal(self, executor):
        """測試即時買入處理"""
        features = np.random.randn(69).astype(np.float32)
        
        trade = executor.process_buy_signal('AAPL', features, datetime(2023, 1, 1), 100.0)
        
        assert trade is not None
        assert executor.portfolio.get_position_count() == 1
    
    def test_reset_daily(self, executor):
        """測試每日重置"""
        features = np.random.randn(69).astype(np.float32)
        executor.add_buy_candidate('AAPL', features, 100.0, datetime(2023, 1, 1))
        
        executor.reset_daily()
        
        assert len(executor.daily_buy_candidates) == 0


# =============================================================================
# 測試執行
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
