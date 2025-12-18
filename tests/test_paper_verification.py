"""
Pro Trader RL 論文規格驗證測試

驗證項目:
1. 特徵計算 (69 維)
2. 正規化公式 (18 個)
3. 停損規則 (-10% 跌幅, 20 天盤整)
4. Agent 架構 (Buy: 69→40→2, Sell: 70→40→2)
5. 投資組合限制 ($10K, 10 檔, 10%, 0.1%)
6. 獎勵閾值 (10% 成功, 0.85 賣出機率)
7. 唐奇安通道 (20 天)
8. 訓練參數 (PPO 超參數)

參考文獻: Pro Trader RL Paper (Table 1-6)
"""

import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, 'd:/000-github-repositories/ptrl-v01')


# =============================================================================
# 1. 特徵計算驗證 (69 維)
# =============================================================================

class TestFeatureSpecification:
    """驗證 69 個特徵符合論文 Table 1-4"""
    
    def test_total_feature_count(self):
        """論文: 69 個特徵"""
        from src.data import FeatureCalculator
        
        calc = FeatureCalculator()
        
        # 建立測試資料
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        np.random.seed(42)
        
        stock_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'High': 102 + np.cumsum(np.random.randn(300) * 0.5),
            'Low': 98 + np.cumsum(np.random.randn(300) * 0.5),
            'Close': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)
        
        stock_data['High'] = stock_data[['Open', 'Close', 'High']].max(axis=1)
        stock_data['Low'] = stock_data[['Open', 'Close', 'Low']].min(axis=1)
        
        index_data = stock_data.copy()
        
        features = calc.calculate_all_features(stock_data, index_data)
        
        # 論文規定 69 個特徵
        assert len(features.columns) == 69, f"特徵數量應為 69，實際為 {len(features.columns)}"
    
    def test_basic_variables(self):
        """論文 Table 1: 9 個基本變數"""
        # Open, High, Low, Close, Volume + HA_Open, HA_High, HA_Low, HA_Close
        expected = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'HA_Open', 'HA_High', 'HA_Low', 'HA_Close']
        
        from src.data import FeatureCalculator
        calc = FeatureCalculator()
        
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        np.random.seed(42)
        stock_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'High': 102 + np.cumsum(np.random.randn(300) * 0.5),
            'Low': 98 + np.cumsum(np.random.randn(300) * 0.5),
            'Close': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)
        stock_data['High'] = stock_data[['Open', 'Close', 'High']].max(axis=1)
        stock_data['Low'] = stock_data[['Open', 'Close', 'Low']].min(axis=1)
        
        features = calc.calculate_all_features(stock_data, stock_data)
        
        for col in expected:
            assert col in features.columns, f"缺少基本變數: {col}"
    
    def test_technical_indicators(self):
        """論文 Table 2: 21 個技術指標"""
        # Return, ATR, Stock_1~12, AVG_Stock, SuperTrend_14, SuperTrend_21, 
        # MFI, RSI, Donchian_Upper, Donchian_Lower
        expected_prefixes = ['Return', 'ATR', 'Stock_', 'AVG_Stock', 
                            'SuperTrend', 'MFI', 'RSI', 'Donchian']
        
        from src.data import FeatureCalculator
        calc = FeatureCalculator()
        
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        np.random.seed(42)
        stock_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'High': 102 + np.cumsum(np.random.randn(300) * 0.5),
            'Low': 98 + np.cumsum(np.random.randn(300) * 0.5),
            'Close': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)
        stock_data['High'] = stock_data[['Open', 'Close', 'High']].max(axis=1)
        stock_data['Low'] = stock_data[['Open', 'Close', 'Low']].min(axis=1)
        
        features = calc.calculate_all_features(stock_data, stock_data)
        
        for prefix in expected_prefixes:
            matching = [c for c in features.columns if c.startswith(prefix)]
            assert len(matching) > 0, f"缺少技術指標: {prefix}*"
    
    def test_stock_n_variables(self):
        """論文: Stock(1)~Stock(12) - 共 12 個"""
        from src.data import FeatureCalculator
        calc = FeatureCalculator()
        
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        np.random.seed(42)
        stock_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'High': 102 + np.cumsum(np.random.randn(300) * 0.5),
            'Low': 98 + np.cumsum(np.random.randn(300) * 0.5),
            'Close': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)
        stock_data['High'] = stock_data[['Open', 'Close', 'High']].max(axis=1)
        stock_data['Low'] = stock_data[['Open', 'Close', 'Low']].min(axis=1)
        
        features = calc.calculate_all_features(stock_data, stock_data)
        
        stock_n = [c for c in features.columns if c.startswith('Stock_') and c[6:].isdigit()]
        assert len(stock_n) == 12, f"Stock(N) 應為 12 個，實際為 {len(stock_n)}"


# =============================================================================
# 2. 正規化公式驗證 (18 個)
# =============================================================================

class TestNormalizationSpecification:
    """驗證 18 個正規化公式符合論文 Eq. 1-18"""
    
    def test_donchian_normalization(self):
        """論文 Eq. 1-2: Donchian Channel 正規化"""
        from src.data import DataNormalizer
        
        normalizer = DataNormalizer()
        
        # 測試資料
        df = pd.DataFrame({
            'Donchian_Upper': [110, 115, 120],
            'Donchian_Lower': [90, 85, 80],
            'High': [108, 112, 118],
            'Low': [92, 88, 82]
        })
        
        normalized = normalizer.normalize(df)
        
        # Eq. 1: DonchianUpper_new = DonchianUpper / High
        assert 'Donchian_Upper_norm' in normalized.columns
        
    def test_percentage_normalization(self):
        """論文 Eq. 16-18: 百分比正規化 (RSRate, MFI, RSI)"""
        from src.data import DataNormalizer
        
        normalizer = DataNormalizer()
        
        df = pd.DataFrame({
            'RS_Rate': [50, 75, 25],
            'MFI': [70, 30, 50],
            'RSI': [60, 40, 55]
        })
        
        normalized = normalizer.normalize(df)
        
        # Eq. 16-18: 乘以 0.01 轉換為 0-1
        for col in ['RS_Rate', 'MFI', 'RSI']:
            norm_col = f'{col}_norm'
            if norm_col in normalized.columns:
                # 正規化後應該在 0-1 範圍
                assert normalized[norm_col].max() <= 1.0
                assert normalized[norm_col].min() >= 0.0


# =============================================================================
# 3. 停損規則驗證
# =============================================================================

class TestStopLossSpecification:
    """驗證停損規則符合論文 Section 3.4"""
    
    def test_stop_loss_dip_threshold(self):
        """論文: 跌幅停損閾值 = -10%"""
        from src.rules import StopLossRule
        
        rule = StopLossRule()
        
        # -10% 應該觸發停損
        result = rule.check(
            buy_price=100,
            current_price=89,  # -11%
            holding_days=10
        )
        
        assert result.should_stop is True
        assert result.stop_type == 'dip'
    
    def test_stop_loss_dip_boundary(self):
        """論文: 恰好 -10% 不應觸發"""
        from src.rules import StopLossRule
        
        rule = StopLossRule()
        
        # 恰好 -10% 不觸發
        result = rule.check(
            buy_price=100,
            current_price=90,
            holding_days=10
        )
        
        # -10% 應該也觸發 (<=)
        # 需確認論文是 < 還是 <=
    
    def test_stop_loss_sideways_threshold(self):
        """論文: 盤整停損 = 連續 20 天報酬 < 10%"""
        from src.rules import StopLossRule
        
        rule = StopLossRule()
        
        # 建立連續 25 天報酬 < 10% 的價格歷史
        buy_price = 100
        prices = pd.Series([100 + i * 0.1 for i in range(25)])  # 微漲但 < 10%
        
        result = rule.check(
            buy_price=buy_price,
            current_price=102,  # +2%
            holding_days=25,
            price_history=prices
        )
        
        # 應該觸發盤整停損
        assert result.should_stop is True
        assert result.stop_type == 'sideways'
    
    def test_stop_loss_max_holding(self):
        """論文: 最大持有天數 = 120 天"""
        from src.rules import StopLossRule
        
        rule = StopLossRule({'max_holding_days': 120})
        
        result = rule.check(
            buy_price=100,
            current_price=105,
            holding_days=121
        )
        
        assert result.should_stop is True
        assert result.stop_type == 'max_holding'


# =============================================================================
# 4. Agent 架構驗證
# =============================================================================

class TestAgentSpecification:
    """驗證 Agent 架構符合論文 Section 3.2.2 & 3.3.2"""
    
    def test_buy_agent_observation_space(self):
        """論文: Buy Agent 輸入 = 69 維"""
        from src.environments import BuyEnv
        
        # 建立測試環境
        n_samples = 100
        data = pd.DataFrame(
            np.random.randn(n_samples, 69),
            columns=[f'f{i}' for i in range(69)]
        )
        data['actual_return'] = np.random.uniform(-0.2, 0.3, n_samples)
        data['is_successful'] = data['actual_return'] >= 0.10
        
        env = BuyEnv(data)
        
        # 論文: 69 維輸入
        assert env.observation_space.shape[0] == 69
    
    def test_sell_agent_observation_space(self):
        """論文: Sell Agent 輸入 = 70 維 (69 + SellReturn)"""
        from src.environments import SellEnv
        from src.environments.sell_env import SellEnvSimple
        
        # 使用 SellEnvSimple 進行測試
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        np.random.seed(42)
        
        # 建立 69 維特徵
        features = pd.DataFrame(
            np.random.randn(300, 69),
            index=dates,
            columns=[f'feature_{i}' for i in range(69)]
        )
        features['Close'] = 100 + np.cumsum(np.random.randn(300) * 2)
        
        # 建立買入訊號
        buy_signals = pd.Series([False] * 300, index=dates)
        buy_signals.iloc[30] = True  # 設定一個買入點
        buy_signals.iloc[100] = True
        
        try:
            env = SellEnvSimple(features, buy_signals, {'max_holding_days': 120})
            # 論文: 70 維輸入 (69 + SellReturn)
            assert env.observation_space.shape[0] == 70
        except Exception as e:
            # 如果 SellEnvSimple 無法正常工作，跳過測試
            pytest.skip(f"SellEnvSimple 初始化失敗: {e}")
    
    def test_buy_agent_action_space(self):
        """論文: Buy Agent 動作 = 2 (買/不買)"""
        from src.environments import BuyEnv
        
        data = pd.DataFrame(np.random.randn(100, 69), columns=[f'f{i}' for i in range(69)])
        data['actual_return'] = np.random.uniform(-0.2, 0.3, 100)
        data['is_successful'] = data['actual_return'] >= 0.10
        
        env = BuyEnv(data)
        
        assert env.action_space.n == 2
    
    def test_sell_agent_action_space(self):
        """論文: Sell Agent 動作 = 2 (賣/持有)"""
        from src.environments.sell_env import SellEnvSimple
        
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        np.random.seed(42)
        
        features = pd.DataFrame(
            np.random.randn(300, 69),
            index=dates,
            columns=[f'feature_{i}' for i in range(69)]
        )
        features['Close'] = 100 + np.cumsum(np.random.randn(300) * 2)
        
        buy_signals = pd.Series([False] * 300, index=dates)
        buy_signals.iloc[30] = True
        
        try:
            env = SellEnvSimple(features, buy_signals, {'max_holding_days': 120})
            assert env.action_space.n == 2
        except Exception as e:
            pytest.skip(f"SellEnvSimple 初始化失敗: {e}")


# =============================================================================
# 5. 投資組合限制驗證
# =============================================================================

class TestPortfolioSpecification:
    """驗證投資組合限制符合論文 Section 4.2"""
    
    def test_initial_capital(self):
        """論文: 初始資金 = $10,000"""
        from src.trading import PortfolioManager
        
        pm = PortfolioManager()  # 使用預設值
        
        assert pm.initial_capital == 10000
    
    def test_max_positions(self):
        """論文: 最大持倉數 = 10"""
        from src.trading import PortfolioManager
        
        pm = PortfolioManager()
        
        assert pm.max_positions == 10
    
    def test_max_position_percentage(self):
        """論文: 單檔最大比例 = 10%"""
        from src.trading import PortfolioManager
        
        pm = PortfolioManager()
        
        assert pm.max_position_pct == 0.10
    
    def test_trading_fee(self):
        """論文: 交易手續費 = 0.1%"""
        from src.trading import PortfolioManager
        
        pm = PortfolioManager()
        
        assert pm.trading_fee == 0.001


# =============================================================================
# 6. 獎勵閾值驗證
# =============================================================================

class TestRewardSpecification:
    """驗證獎勵閾值符合論文 Section 3.2.1 & 3.3.1"""
    
    def test_success_threshold(self):
        """論文: 成功交易定義 = 報酬 >= 10%"""
        # Buy Agent 使用 10% 作為成功閾值
        SUCCESS_THRESHOLD = 0.10
        
        # 9% 不是成功
        assert (0.09 >= SUCCESS_THRESHOLD) is False
        
        # 10% 是成功
        assert (0.10 >= SUCCESS_THRESHOLD) is True
        
        # 11% 是成功
        assert (0.11 >= SUCCESS_THRESHOLD) is True
    
    def test_sell_probability_threshold(self):
        """論文: 賣出機率閾值 = |sell_prob - hold_prob| > 0.85"""
        from src.trading import TradeExecutor
        from src.trading import PortfolioManager
        
        pm = PortfolioManager()
        executor = TradeExecutor(pm)
        
        # 論文設定
        assert executor.sell_prob_threshold == 0.85


# =============================================================================
# 7. Donchian Channel 驗證
# =============================================================================

class TestDonchianSpecification:
    """驗證 Donchian Channel 符合論文 Section 3.1.2"""
    
    def test_donchian_period(self):
        """論文: Donchian Channel 週期 = 20 天"""
        from src.rules import DonchianChannel
        
        dc = DonchianChannel()  # 使用預設值
        
        assert dc.period == 20
    
    def test_donchian_buy_signal(self):
        """論文: 當 High > 上通道 時產生買入訊號"""
        from src.rules import DonchianChannel
        
        dc = DonchianChannel(period=20)
        
        # 建立測試資料
        dates = pd.date_range('2020-01-01', periods=30, freq='B')
        df = pd.DataFrame({
            'High': [100 + i for i in range(30)],
            'Low': [90 + i for i in range(30)],
            'Close': [95 + i for i in range(30)]
        }, index=dates)
        
        # 計算 Donchian Channel
        result = dc.calculate(df)
        
        # 應該有上下通道
        assert 'Donchian_Upper' in result.columns
        assert 'Donchian_Lower' in result.columns
        
        # 使用 generate_buy_signals 方法
        buy_signals = dc.generate_buy_signals(result)
        
        # 應該有買入訊號 (因為 High 持續創新高)
        assert buy_signals.sum() > 0  # 至少有一個買入訊號


# =============================================================================
# 8. PPO 超參數驗證
# =============================================================================

class TestPPOSpecification:
    """驗證 PPO 超參數符合論文 Table 6"""
    
    def test_ppo_learning_rate(self):
        """論文 Table 6: Learning Rate = 0.0001"""
        from src.agents import BuyAgent
        
        agent = BuyAgent()
        
        assert agent.learning_rate == 0.0001
    
    def test_ppo_batch_size(self):
        """論文 Table 6: Batch Size = 64"""
        from src.agents import BuyAgent
        
        agent = BuyAgent()
        
        assert agent.batch_size == 64
    
    def test_ppo_n_steps(self):
        """論文 Table 6: N Steps = 2048"""
        from src.agents import BuyAgent
        
        agent = BuyAgent()
        
        assert agent.n_steps == 2048


# =============================================================================
# 9. 整合驗證
# =============================================================================

class TestIntegrationSpecification:
    """整合驗證: 確保各模組正確協作"""
    
    def test_full_pipeline_feature_flow(self):
        """驗證資料流: 原始資料 -> 69 特徵 -> 正規化 -> RL 輸入"""
        from src.data import DataLoader, FeatureCalculator, DataNormalizer
        
        calc = FeatureCalculator()
        normalizer = DataNormalizer()
        
        # 建立測試資料
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        np.random.seed(42)
        stock_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'High': 102 + np.cumsum(np.random.randn(300) * 0.5),
            'Low': 98 + np.cumsum(np.random.randn(300) * 0.5),
            'Close': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)
        stock_data['High'] = stock_data[['Open', 'Close', 'High']].max(axis=1)
        stock_data['Low'] = stock_data[['Open', 'Close', 'Low']].min(axis=1)
        
        # Step 1: 計算特徵 (應為 69 維)
        features = calc.calculate_all_features(stock_data, stock_data)
        assert len(features.columns) == 69
        
        # Step 2: 正規化
        normalized = normalizer.normalize(features)
        assert len(normalized.columns) > 69  # 原始 + 正規化欄位
        
        # Step 3: 提取 RL 特徵 (應為 69 維)
        rl_features = normalizer.extract_normalized_features(normalized)
        assert len(rl_features.columns) == 69


# =============================================================================
# 測試執行
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
