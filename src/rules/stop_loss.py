"""
StopLossRule - 停損規則模組

根據論文實作兩種停損規則:

1. 跌幅停損 (Stop Loss on Dips)
   - 當報酬率 < -10% 時觸發
   - 立即賣出

2. 盤整停損 (Stop Loss on Sideways)
   - 持有 120 天內，連續 20 天報酬率未達 10%
   - 強制賣出

重要: 停損規則的優先級高於 Sell Agent 的決策
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class StopLossResult:
    """停損判斷結果"""
    should_stop: bool           # 是否應該停損
    stop_type: Optional[str]    # 停損類型: 'dip' / 'sideways' / None
    current_return: float       # 當前報酬率
    holding_days: int           # 已持有天數
    message: str                # 說明訊息


class StopLossRule:
    """
    停損規則管理器
    
    論文定義的兩種停損規則:
    1. 跌幅停損: 報酬率 < -10%
    2. 盤整停損: 持有 120 天內連續 20 天報酬率 < 10%
    
    使用方式:
        rule = StopLossRule(config)
        result = rule.check(buy_price, current_price, holding_days, price_history)
    """
    
    def __init__(self, config: dict = None):
        """
        初始化停損規則
        
        Args:
            config: 設定字典
                - dip_threshold: 跌幅停損閾值 (預設 -0.10)
                - sideways_days: 盤整觀察天數 (預設 20)
                - sideways_threshold: 盤整獲利閾值 (預設 0.10)
                - max_holding_days: 最大持有天數 (預設 120)
        """
        config = config or {}
        
        # 跌幅停損參數
        self.dip_threshold = config.get('dip_threshold', -0.10)
        
        # 盤整停損參數
        self.sideways_days = config.get('sideways_days', 20)
        self.sideways_threshold = config.get('sideways_threshold', 0.10)
        self.max_holding_days = config.get('max_holding_days', 120)
        
        logger.info(f"StopLossRule 初始化完成 - 跌幅停損: {self.dip_threshold:.0%}, "
                   f"盤整停損: {self.sideways_days}天內未達{self.sideways_threshold:.0%}")
    
    def check(self, 
              buy_price: float,
              current_price: float,
              holding_days: int,
              price_history: Optional[pd.Series] = None) -> StopLossResult:
        """
        檢查是否應該停損
        
        Args:
            buy_price: 買入價格
            current_price: 當前價格
            holding_days: 已持有天數
            price_history: 持有期間的價格序列 (用於盤整判斷)
            
        Returns:
            StopLossResult 停損判斷結果
        """
        # 計算當前報酬率
        current_return = (current_price - buy_price) / buy_price
        
        # 1. 檢查跌幅停損
        if current_return < self.dip_threshold:
            return StopLossResult(
                should_stop=True,
                stop_type='dip',
                current_return=current_return,
                holding_days=holding_days,
                message=f"跌幅停損觸發: 報酬率 {current_return:.2%} < {self.dip_threshold:.0%}"
            )
        
        # 2. 檢查盤整停損
        if holding_days >= self.sideways_days and price_history is not None:
            sideways_result = self._check_sideways(buy_price, price_history, holding_days)
            if sideways_result.should_stop:
                return sideways_result
        
        # 3. 檢查最大持有天數
        if holding_days >= self.max_holding_days:
            return StopLossResult(
                should_stop=True,
                stop_type='max_holding',
                current_return=current_return,
                holding_days=holding_days,
                message=f"達到最大持有天數: {holding_days} >= {self.max_holding_days}"
            )
        
        # 未觸發停損
        return StopLossResult(
            should_stop=False,
            stop_type=None,
            current_return=current_return,
            holding_days=holding_days,
            message="未觸發停損規則"
        )
    
    def _check_sideways(self, 
                        buy_price: float, 
                        price_history: pd.Series,
                        holding_days: int) -> StopLossResult:
        """
        檢查盤整停損
        
        論文定義: 持有 120 天內，連續 20 天的報酬率都未達到 10%
        """
        if len(price_history) < self.sideways_days:
            return StopLossResult(
                should_stop=False,
                stop_type=None,
                current_return=0.0,
                holding_days=holding_days,
                message="價格歷史不足"
            )
        
        # 取最近 N 天的價格
        recent_prices = price_history.tail(self.sideways_days)
        
        # 計算每天的報酬率
        daily_returns = (recent_prices - buy_price) / buy_price
        
        # 檢查是否所有天都未達到閾值
        all_below_threshold = (daily_returns < self.sideways_threshold).all()
        
        current_return = daily_returns.iloc[-1] if len(daily_returns) > 0 else 0.0
        
        if all_below_threshold:
            return StopLossResult(
                should_stop=True,
                stop_type='sideways',
                current_return=float(current_return),
                holding_days=holding_days,
                message=f"盤整停損觸發: 連續 {self.sideways_days} 天報酬率 < {self.sideways_threshold:.0%}"
            )
        
        return StopLossResult(
            should_stop=False,
            stop_type=None,
            current_return=float(current_return),
            holding_days=holding_days,
            message="盤整檢查通過"
        )
    
    def check_batch(self, positions: List[Dict]) -> List[StopLossResult]:
        """
        批次檢查多個持倉的停損狀態
        
        Args:
            positions: 持倉列表，每個持倉包含:
                - buy_price: 買入價格
                - current_price: 當前價格
                - holding_days: 持有天數
                - price_history: 價格歷史 (可選)
                
        Returns:
            StopLossResult 列表
        """
        results = []
        
        for pos in positions:
            result = self.check(
                buy_price=pos['buy_price'],
                current_price=pos['current_price'],
                holding_days=pos['holding_days'],
                price_history=pos.get('price_history')
            )
            results.append(result)
        
        return results


class DonchianChannel:
    """
    Donchian Channel 策略
    
    買入訊號: 當日最高價 > 過去 N 天最高價 (突破上通道)
    賣出訊號: 當日最低價 < 過去 N 天最低價 (跌破下通道)
    """
    
    def __init__(self, period: int = 20):
        """
        Args:
            period: 通道週期 (預設 20 天)
        """
        self.period = period
        logger.info(f"DonchianChannel 初始化完成 - 週期: {period} 天")
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 Donchian Channel
        
        Args:
            df: 包含 High, Low 的 DataFrame
            
        Returns:
            加入 Donchian_Upper, Donchian_Lower 的 DataFrame
        """
        result = df.copy()
        
        # 上通道 (過去 N 天最高價)
        result['Donchian_Upper'] = df['High'].rolling(window=self.period).max()
        
        # 下通道 (過去 N 天最低價)
        result['Donchian_Lower'] = df['Low'].rolling(window=self.period).min()
        
        return result
    
    def generate_buy_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        產生買入訊號
        
        買入條件: 當日最高價 > 前一天的上通道
        """
        if 'Donchian_Upper' not in df.columns:
            df = self.calculate(df)
        
        # 突破上通道
        signal = df['High'] > df['Donchian_Upper'].shift(1)
        
        # 只保留第一天的突破 (移除連續訊號)
        signal = signal & (~signal.shift(1).fillna(False))
        
        return signal.astype(int)
    
    def generate_sell_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        產生賣出訊號
        
        賣出條件: 當日最低價 < 前一天的下通道
        """
        if 'Donchian_Lower' not in df.columns:
            df = self.calculate(df)
        
        # 跌破下通道
        signal = df['Low'] < df['Donchian_Lower'].shift(1)
        
        return signal.astype(int)


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    # 測試停損規則
    print("=== 停損規則測試 ===")
    
    rule = StopLossRule()
    
    # 測試 1: 跌幅停損
    result = rule.check(
        buy_price=100,
        current_price=85,  # -15%
        holding_days=10
    )
    print(f"跌幅停損測試: {result}")
    
    # 測試 2: 正常持有
    result = rule.check(
        buy_price=100,
        current_price=105,  # +5%
        holding_days=10
    )
    print(f"正常持有測試: {result}")
    
    # 測試 3: 盤整停損
    price_history = pd.Series([105, 104, 106, 103, 105] * 4)  # 20 天都在 3-6% 波動
    result = rule.check(
        buy_price=100,
        current_price=105,
        holding_days=25,
        price_history=price_history
    )
    print(f"盤整停損測試: {result}")
    
    print("\n=== Donchian Channel 測試 ===")
    
    # 創建測試資料
    dates = pd.date_range('2023-01-01', periods=30)
    df = pd.DataFrame({
        'High': np.random.uniform(100, 110, 30),
        'Low': np.random.uniform(90, 100, 30),
        'Close': np.random.uniform(95, 105, 30)
    }, index=dates)
    
    donchian = DonchianChannel(period=20)
    df_with_donchian = donchian.calculate(df)
    
    buy_signals = donchian.generate_buy_signals(df_with_donchian)
    print(f"買入訊號數量: {buy_signals.sum()}")
    
    print("\n停損規則模組測試完成!")
