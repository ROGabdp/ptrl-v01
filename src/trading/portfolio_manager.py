"""
PortfolioManager - 投資組合管理器

根據論文設定管理投資組合:
- 初始資金: $10,000
- 最大持倉數: 10 檔
- 單檔最大比例: 10%
- 交易手續費: 0.1%

使用方式:
    pm = PortfolioManager(config)
    trade = pm.open_position('AAPL', 150.0, datetime.now())
    pm.close_position('AAPL', 165.0, datetime.now(), 'agent')
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class Position:
    """持倉資訊"""
    symbol: str
    buy_date: datetime
    buy_price: float
    shares: int
    cost_basis: float  # 含手續費的買入成本
    
    @property
    def current_value(self) -> float:
        """當前市值 (需外部更新價格)"""
        return self._current_price * self.shares if hasattr(self, '_current_price') else 0
    
    def update_price(self, price: float):
        """更新當前價格"""
        self._current_price = price
    
    def get_return(self, current_price: float) -> float:
        """計算報酬率"""
        return (current_price - self.buy_price) / self.buy_price
    
    def get_holding_days(self, current_date: datetime) -> int:
        """計算持有天數"""
        return (current_date - self.buy_date).days
    
    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            'symbol': self.symbol,
            'buy_date': self.buy_date.isoformat(),
            'buy_price': self.buy_price,
            'shares': self.shares,
            'cost_basis': self.cost_basis
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        """從字典建立"""
        return cls(
            symbol=data['symbol'],
            buy_date=datetime.fromisoformat(data['buy_date']),
            buy_price=data['buy_price'],
            shares=data['shares'],
            cost_basis=data['cost_basis']
        )


@dataclass
class Trade:
    """交易紀錄"""
    symbol: str
    action: str  # 'BUY' / 'SELL'
    date: datetime
    price: float
    shares: int
    fee: float
    total_value: float
    reason: Optional[str] = None
    return_pct: Optional[float] = None
    holding_days: Optional[int] = None
    
    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'date': self.date.isoformat(),
            'price': self.price,
            'shares': self.shares,
            'fee': self.fee,
            'total_value': self.total_value,
            'reason': self.reason,
            'return_pct': self.return_pct,
            'holding_days': self.holding_days
        }


class PortfolioManager:
    """
    投資組合管理器
    
    負責:
    - 持倉追蹤與管理
    - 資金分配 (論文: 最多 10 檔，單檔 ≤10%)
    - 交易成本計算 (0.1%)
    - 權益曲線記錄
    
    使用方式:
        pm = PortfolioManager(config)
        trade = pm.open_position('AAPL', 150.0, datetime.now())
        pm.close_position('AAPL', 165.0, datetime.now(), 'agent')
    """
    
    def __init__(self, config: dict = None):
        """
        初始化投資組合管理器
        
        Args:
            config: 設定字典
                - initial_capital: 初始資金 (預設 10000)
                - max_positions: 最大持倉數 (預設 10)
                - max_position_pct: 單檔最大比例 (預設 0.10)
                - trading_fee: 交易手續費 (預設 0.001)
                - state_file: 狀態儲存檔案路徑
        """
        config = config or {}
        
        # 投資組合參數 (論文設定)
        self.initial_capital = config.get('initial_capital', 10000)
        self.max_positions = config.get('max_positions', 10)
        self.max_position_pct = config.get('max_position_pct', 0.10)
        self.trading_fee = config.get('trading_fee', 0.001)
        
        # 狀態儲存
        self.state_file = config.get('state_file', 'data/portfolio_state.json')
        
        # 內部狀態
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        logger.info(f"PortfolioManager 初始化 - 初始資金: ${self.initial_capital:,}, "
                   f"最大持倉: {self.max_positions}, 單檔上限: {self.max_position_pct:.0%}")
    
    def open_position(self, symbol: str, price: float, date: datetime,
                      size_override: float = None) -> Optional[Trade]:
        """
        開立新倉位
        
        Args:
            symbol: 股票代碼
            price: 買入價格
            date: 買入日期
            size_override: 自訂倉位大小 (覆蓋預設計算)
            
        Returns:
            Trade 物件，若無法開倉則返回 None
        """
        # 檢查是否已持有
        if symbol in self.positions:
            logger.warning(f"已持有 {symbol}，無法重複開倉")
            return None
        
        # 檢查持倉數量限制
        if len(self.positions) >= self.max_positions:
            logger.warning(f"已達最大持倉數 {self.max_positions}")
            return None
        
        # 計算倉位大小
        if size_override:
            position_value = size_override
        else:
            total_equity = self.get_equity({})
            position_value = total_equity * self.max_position_pct
        
        # 計算可買股數
        shares = int(position_value / price)
        if shares <= 0:
            logger.warning(f"資金不足以購買 {symbol}")
            return None
        
        # 計算交易成本
        gross_value = price * shares
        fee = gross_value * self.trading_fee
        total_cost = gross_value + fee
        
        # 檢查現金是否足夠
        if total_cost > self.cash:
            # 調整股數
            shares = int((self.cash / (1 + self.trading_fee)) / price)
            if shares <= 0:
                logger.warning(f"現金不足以購買 {symbol}")
                return None
            gross_value = price * shares
            fee = gross_value * self.trading_fee
            total_cost = gross_value + fee
        
        # 扣除現金
        self.cash -= total_cost
        
        # 建立持倉
        position = Position(
            symbol=symbol,
            buy_date=date,
            buy_price=price,
            shares=shares,
            cost_basis=total_cost
        )
        self.positions[symbol] = position
        
        # 記錄交易
        trade = Trade(
            symbol=symbol,
            action='BUY',
            date=date,
            price=price,
            shares=shares,
            fee=fee,
            total_value=total_cost,
            reason='signal'
        )
        self.trade_history.append(trade)
        
        logger.info(f"開倉 {symbol}: {shares} 股 @ ${price:.2f}, 總成本 ${total_cost:.2f}")
        
        return trade
    
    def close_position(self, symbol: str, price: float, date: datetime,
                       reason: str = 'agent') -> Optional[Trade]:
        """
        平倉
        
        Args:
            symbol: 股票代碼
            price: 賣出價格
            date: 賣出日期
            reason: 賣出原因 ('agent', 'stop_loss_dip', 'stop_loss_sideways', 'max_holding')
            
        Returns:
            Trade 物件，若無持倉則返回 None
        """
        if symbol not in self.positions:
            logger.warning(f"未持有 {symbol}，無法平倉")
            return None
        
        position = self.positions[symbol]
        
        # 計算交易金額
        gross_value = price * position.shares
        fee = gross_value * self.trading_fee
        net_value = gross_value - fee
        
        # 計算報酬率和持有天數
        return_pct = position.get_return(price)
        holding_days = position.get_holding_days(date)
        
        # 增加現金
        self.cash += net_value
        
        # 移除持倉
        del self.positions[symbol]
        
        # 記錄交易
        trade = Trade(
            symbol=symbol,
            action='SELL',
            date=date,
            price=price,
            shares=position.shares,
            fee=fee,
            total_value=net_value,
            reason=reason,
            return_pct=return_pct,
            holding_days=holding_days
        )
        self.trade_history.append(trade)
        
        logger.info(f"平倉 {symbol}: {position.shares} 股 @ ${price:.2f}, "
                   f"報酬 {return_pct:.2%}, 持有 {holding_days} 天, 原因: {reason}")
        
        return trade
    
    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """
        計算總權益
        
        Args:
            current_prices: 當前價格字典 {symbol: price}
            
        Returns:
            總權益 (現金 + 持倉市值)
        """
        equity = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                equity += current_prices[symbol] * position.shares
            else:
                # 若無當前價格，使用買入價格估算
                equity += position.buy_price * position.shares
        
        return equity
    
    def record_equity(self, date: datetime, current_prices: Dict[str, float]):
        """記錄權益曲線"""
        equity = self.get_equity(current_prices)
        self.equity_history.append((date, equity))
    
    def get_available_cash(self) -> float:
        """取得可用現金"""
        return self.cash
    
    def get_position_count(self) -> int:
        """取得持倉數量"""
        return len(self.positions)
    
    def get_positions(self) -> Dict[str, Position]:
        """取得所有持倉"""
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """取得特定持倉"""
        return self.positions.get(symbol)
    
    def can_open_position(self) -> bool:
        """檢查是否可以開新倉"""
        return len(self.positions) < self.max_positions
    
    def get_suggested_position_size(self) -> float:
        """取得建議的倉位大小"""
        total_equity = self.get_equity({})
        return total_equity * self.max_position_pct
    
    def get_trade_history(self) -> List[Trade]:
        """取得交易歷史"""
        return self.trade_history.copy()
    
    def get_equity_curve(self) -> pd.Series:
        """取得權益曲線 DataFrame"""
        if not self.equity_history:
            return pd.Series(dtype=float)
        
        dates, values = zip(*self.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(dates))
    
    def get_statistics(self) -> dict:
        """計算投資組合統計"""
        buy_trades = [t for t in self.trade_history if t.action == 'BUY']
        sell_trades = [t for t in self.trade_history if t.action == 'SELL']
        
        winning_trades = [t for t in sell_trades if t.return_pct and t.return_pct > 0]
        
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'total_equity': self.get_equity({}),
            'position_count': len(self.positions),
            'total_trades': len(self.trade_history),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(sell_trades) if sell_trades else 0,
            'total_fees': sum(t.fee for t in self.trade_history)
        }
    
    def reset(self):
        """重置投資組合"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_history = []
        logger.info("投資組合已重置")
    
    def save_state(self, filepath: str = None):
        """儲存狀態"""
        filepath = filepath or self.state_file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'cash': self.cash,
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'trade_history': [t.to_dict() for t in self.trade_history],
            'config': {
                'initial_capital': self.initial_capital,
                'max_positions': self.max_positions,
                'max_position_pct': self.max_position_pct,
                'trading_fee': self.trading_fee
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"投資組合狀態已儲存至: {filepath}")
    
    def load_state(self, filepath: str = None):
        """載入狀態"""
        filepath = filepath or self.state_file
        
        if not os.path.exists(filepath):
            logger.warning(f"狀態檔案不存在: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.cash = state['cash']
        self.positions = {s: Position.from_dict(p) for s, p in state['positions'].items()}
        
        logger.info(f"投資組合狀態已載入: {filepath}")


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    from datetime import datetime
    
    print("=== PortfolioManager 測試 ===")
    
    # 初始化
    pm = PortfolioManager({
        'initial_capital': 10000,
        'max_positions': 10,
        'max_position_pct': 0.10,
        'trading_fee': 0.001
    })
    
    # 開倉測試
    trade1 = pm.open_position('AAPL', 150.0, datetime(2023, 1, 1))
    trade2 = pm.open_position('MSFT', 300.0, datetime(2023, 1, 1))
    
    print(f"\n持倉數: {pm.get_position_count()}")
    print(f"現金: ${pm.get_available_cash():.2f}")
    
    # 平倉測試
    trade3 = pm.close_position('AAPL', 165.0, datetime(2023, 3, 1), 'agent')
    
    # 統計
    stats = pm.get_statistics()
    print(f"\n統計: {stats}")
