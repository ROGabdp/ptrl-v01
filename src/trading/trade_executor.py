"""
TradeExecutor - 交易執行器

整合 Buy Agent、Sell Agent 和 StopLoss Rule 的交易執行器:
- 處理買入訊號 (結合 BuyAgent 過濾)
- 處理賣出決策 (結合 SellAgent 判斷)
- 執行停損規則

使用方式:
    executor = TradeExecutor(config, portfolio_manager)
    executor.set_agents(buy_agent, sell_agent)
    executor.set_stop_loss(stop_loss_rule)
    trade = executor.process_buy_signal(symbol, features, date, price)
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .portfolio_manager import PortfolioManager, Trade, Position


class TradeExecutor:
    """
    交易執行器
    
    負責:
    - 整合 BuyAgent + SellAgent 決策
    - 執行停損規則檢查
    - 訂單生成與執行
    - 交易紀錄
    
    使用方式:
        executor = TradeExecutor(config, portfolio_manager)
        executor.set_agents(buy_agent, sell_agent)
        trade = executor.process_buy_signal(symbol, features, date, price)
    """
    
    def __init__(self, portfolio_manager: PortfolioManager, config: dict = None):
        """
        初始化交易執行器
        
        Args:
            portfolio_manager: 投資組合管理器
            config: 設定字典
                - buy_confidence_threshold: 買入信心閾值 (預設 0.5)
                - sell_prob_threshold: 賣出機率閾值 (預設 0.85，論文設定)
                - use_top_n: 是否使用 Top N 選股 (預設 True)
                - top_n: Top N 數量 (預設 10)
        """
        config = config or {}
        
        self.portfolio = portfolio_manager
        
        # 決策參數
        self.buy_confidence_threshold = config.get('buy_confidence_threshold', 0.5)
        self.sell_prob_threshold = config.get('sell_prob_threshold', 0.85)
        self.use_top_n = config.get('use_top_n', True)
        self.top_n = config.get('top_n', 10)
        
        # Agent 參考
        self.buy_agent = None
        self.sell_agent = None
        self.stop_loss_rule = None
        
        # 每日買入候選 (用於 Top N 選股)
        self.daily_buy_candidates: List[Dict] = []
        
        logger.info(f"TradeExecutor 初始化 - 買入閾值: {self.buy_confidence_threshold}, "
                   f"賣出閾值: {self.sell_prob_threshold}")
    
    def set_agents(self, buy_agent=None, sell_agent=None):
        """設定 Agent"""
        if buy_agent:
            self.buy_agent = buy_agent
            logger.info("BuyAgent 已設定")
        if sell_agent:
            self.sell_agent = sell_agent
            logger.info("SellAgent 已設定")
    
    def set_stop_loss(self, stop_loss_rule):
        """設定停損規則"""
        self.stop_loss_rule = stop_loss_rule
        logger.info("StopLossRule 已設定")
    
    def evaluate_buy_signal(self, symbol: str, features: np.ndarray,
                            price: float) -> Tuple[bool, float]:
        """
        評估買入訊號
        
        Args:
            symbol: 股票代碼
            features: 正規化後的 69 維特徵
            price: 當前價格
            
        Returns:
            (should_buy, confidence): 是否買入及信心度
        """
        # 無 Agent 時直接通過
        if self.buy_agent is None or self.buy_agent.model is None:
            return True, 1.0
        
        try:
            # 確保特徵是正確的形狀
            obs = np.array(features).astype(np.float32)
            obs = np.nan_to_num(obs, nan=0.0)
            
            # 取得預測
            action = self.buy_agent.predict(obs)
            probs = self.buy_agent.predict_proba(obs)
            confidence = float(probs[1])  # 買入機率
            
            should_buy = action == 1 and confidence >= self.buy_confidence_threshold
            
            return should_buy, confidence
            
        except Exception as e:
            logger.warning(f"BuyAgent 預測失敗 ({symbol}): {e}")
            return False, 0.0
    
    def add_buy_candidate(self, symbol: str, features: np.ndarray,
                          price: float, date: datetime):
        """
        加入買入候選 (用於 Top N 選股)
        
        Args:
            symbol: 股票代碼
            features: 正規化後的特徵
            price: 當前價格
            date: 日期
        """
        should_buy, confidence = self.evaluate_buy_signal(symbol, features, price)
        
        if should_buy:
            self.daily_buy_candidates.append({
                'symbol': symbol,
                'features': features,
                'price': price,
                'date': date,
                'confidence': confidence
            })
    
    def execute_daily_buys(self) -> List[Trade]:
        """
        執行每日買入 (從候選中選擇 Top N)
        
        Returns:
            成功執行的交易列表
        """
        if not self.daily_buy_candidates:
            return []
        
        # 按信心度排序
        self.daily_buy_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 選擇 Top N
        if self.use_top_n:
            candidates = self.daily_buy_candidates[:self.top_n]
        else:
            candidates = self.daily_buy_candidates
        
        executed_trades = []
        
        for candidate in candidates:
            # 檢查是否可以開倉
            if not self.portfolio.can_open_position():
                break
            
            # 執行買入
            trade = self.portfolio.open_position(
                symbol=candidate['symbol'],
                price=candidate['price'],
                date=candidate['date']
            )
            
            if trade:
                executed_trades.append(trade)
        
        # 清空候選
        self.daily_buy_candidates = []
        
        return executed_trades
    
    def process_buy_signal(self, symbol: str, features: np.ndarray,
                           date: datetime, price: float) -> Optional[Trade]:
        """
        處理買入訊號 (即時執行，不使用 Top N)
        
        Args:
            symbol: 股票代碼
            features: 正規化後的 69 維特徵
            date: 日期
            price: 當前價格
            
        Returns:
            Trade 物件，若未執行則返回 None
        """
        # 檢查是否可以開倉
        if not self.portfolio.can_open_position():
            return None
        
        # 評估買入訊號
        should_buy, confidence = self.evaluate_buy_signal(symbol, features, price)
        
        if not should_buy:
            return None
        
        # 執行買入
        trade = self.portfolio.open_position(symbol, price, date)
        
        return trade
    
    def evaluate_sell_decision(self, position: Position, features: np.ndarray,
                               current_price: float) -> Tuple[bool, float, float]:
        """
        評估賣出決策
        
        Args:
            position: 持倉
            features: 正規化後的 69 維特徵
            current_price: 當前價格
            
        Returns:
            (should_sell, sell_prob, hold_prob): 是否賣出及機率
        """
        # 無 Agent 時不賣
        if self.sell_agent is None or self.sell_agent.model is None:
            return False, 0.0, 1.0
        
        try:
            # 計算 SellReturn (公式 20)
            sell_return = current_price / position.buy_price
            
            # 組合觀察向量 (69 特徵 + SellReturn)
            obs = np.array(features).astype(np.float32)
            obs = np.nan_to_num(obs, nan=0.0)
            obs = np.concatenate([obs, [sell_return]])
            
            # 取得預測
            probs = self.sell_agent.predict_proba(obs)
            hold_prob = float(probs[0])
            sell_prob = float(probs[1])
            
            # 論文判斷邏輯: |sell_prob - hold_prob| > 0.85 且 sell_prob > hold_prob
            prob_diff = abs(sell_prob - hold_prob)
            should_sell = prob_diff > self.sell_prob_threshold and sell_prob > hold_prob
            
            return should_sell, sell_prob, hold_prob
            
        except Exception as e:
            logger.warning(f"SellAgent 預測失敗 ({position.symbol}): {e}")
            return False, 0.0, 1.0
    
    def check_stop_loss(self, position: Position, current_price: float,
                        current_date: datetime, price_history=None):
        """
        檢查停損條件
        
        Args:
            position: 持倉
            current_price: 當前價格
            current_date: 當前日期
            price_history: 價格歷史 (用於盤整判斷)
            
        Returns:
            StopLossResult 或 None
        """
        if self.stop_loss_rule is None:
            return None
        
        holding_days = position.get_holding_days(current_date)
        
        return self.stop_loss_rule.check(
            buy_price=position.buy_price,
            current_price=current_price,
            holding_days=holding_days,
            price_history=price_history
        )
    
    def process_sell_decision(self, symbol: str, features: np.ndarray,
                              current_price: float, current_date: datetime,
                              price_history=None) -> Optional[Trade]:
        """
        處理賣出決策
        
        Args:
            symbol: 股票代碼
            features: 正規化後的 69 維特徵
            current_price: 當前價格
            current_date: 當前日期
            price_history: 價格歷史
            
        Returns:
            Trade 物件，若未執行則返回 None
        """
        position = self.portfolio.get_position(symbol)
        if position is None:
            return None
        
        # 1. 優先檢查停損 (覆蓋 Agent 決策)
        stop_result = self.check_stop_loss(position, current_price, current_date, price_history)
        
        if stop_result and stop_result.should_stop:
            trade = self.portfolio.close_position(
                symbol=symbol,
                price=current_price,
                date=current_date,
                reason=f'stop_loss_{stop_result.stop_type}'
            )
            return trade
        
        # 2. 使用 Sell Agent 判斷
        should_sell, sell_prob, hold_prob = self.evaluate_sell_decision(
            position, features, current_price
        )
        
        if should_sell:
            trade = self.portfolio.close_position(
                symbol=symbol,
                price=current_price,
                date=current_date,
                reason='agent'
            )
            return trade
        
        return None
    
    def process_all_positions(self, all_features: Dict[str, np.ndarray],
                              all_prices: Dict[str, float],
                              all_histories: Dict[str, any],
                              current_date: datetime) -> List[Trade]:
        """
        處理所有持倉的賣出決策
        
        Args:
            all_features: 所有股票的特徵 {symbol: features}
            all_prices: 所有股票的價格 {symbol: price}
            all_histories: 所有股票的價格歷史 {symbol: history}
            current_date: 當前日期
            
        Returns:
            執行的交易列表
        """
        executed_trades = []
        
        # 複製持倉列表 (因為會在迴圈中修改)
        positions = list(self.portfolio.get_positions().keys())
        
        for symbol in positions:
            if symbol not in all_prices:
                continue
            
            features = all_features.get(symbol, np.zeros(69))
            price = all_prices[symbol]
            history = all_histories.get(symbol)
            
            trade = self.process_sell_decision(
                symbol=symbol,
                features=features,
                current_price=price,
                current_date=current_date,
                price_history=history
            )
            
            if trade:
                executed_trades.append(trade)
        
        return executed_trades
    
    def reset_daily(self):
        """重置每日狀態"""
        self.daily_buy_candidates = []


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    import numpy as np
    from datetime import datetime
    
    print("=== TradeExecutor 測試 ===")
    
    # 初始化
    pm = PortfolioManager({'initial_capital': 10000})
    executor = TradeExecutor(pm)
    
    # 模擬買入訊號
    features = np.random.randn(69).astype(np.float32)
    
    # 加入候選
    executor.add_buy_candidate('AAPL', features, 150.0, datetime(2023, 1, 1))
    executor.add_buy_candidate('MSFT', features, 300.0, datetime(2023, 1, 1))
    executor.add_buy_candidate('GOOGL', features, 100.0, datetime(2023, 1, 1))
    
    print(f"候選數量: {len(executor.daily_buy_candidates)}")
    
    # 執行買入 (無 Agent 時全部通過)
    trades = executor.execute_daily_buys()
    print(f"執行交易數: {len(trades)}")
    print(f"持倉數: {pm.get_position_count()}")
