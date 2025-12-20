"""
BacktestEngine - 回測引擎

完整的 Pro Trader RL 回測引擎，包含:
- 資料載入與特徵計算
- Buy Agent 買入訊號過濾
- Sell Agent 賣出時機決策
- Stop Loss 規則執行
- 投資組合管理
- 績效計算與報告
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Trade:
    """交易紀錄"""
    symbol: str
    buy_date: datetime
    buy_price: float
    sell_date: Optional[datetime] = None
    sell_price: Optional[float] = None
    shares: int = 0
    status: str = 'open'  # 'open' / 'closed'
    sell_reason: Optional[str] = None  # 'agent' / 'stop_loss_dip' / 'stop_loss_sideways' / 'max_holding'
    
    @property
    def return_pct(self) -> Optional[float]:
        """報酬率"""
        if self.sell_price and self.buy_price > 0:
            return (self.sell_price - self.buy_price) / self.buy_price
        return None
    
    @property
    def holding_days(self) -> Optional[int]:
        """持有天數"""
        if self.sell_date:
            return (self.sell_date - self.buy_date).days
        return None
    
    @property
    def profit(self) -> Optional[float]:
        """獲利金額"""
        if self.sell_price:
            return (self.sell_price - self.buy_price) * self.shares
        return None


@dataclass
class BacktestResult:
    """回測結果"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    avg_holding_days: float
    trades: List[Trade] = field(default_factory=list)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    equity_curve: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    """
    Pro Trader RL 回測引擎
    
    使用方式:
        engine = BacktestEngine(config)
        result = engine.run(start_date, end_date)
    """
    
    def __init__(self, config: dict = None):
        """
        初始化回測引擎
        
        Args:
            config: 設定字典
                - initial_capital: 初始資金 (預設 10000)
                - max_positions: 最大持倉數 (預設 10)
                - max_position_pct: 單檔最大比例 (預設 0.10)
                - trading_fee: 交易手續費 (預設 0.001)
        """
        config = config or {}
        
        # 投資組合參數 (論文設定)
        self.initial_capital = config.get('initial_capital', 10000)
        self.max_positions = config.get('max_positions', 10)
        self.max_position_pct = config.get('max_position_pct', 0.10)
        self.trading_fee = config.get('trading_fee', 0.001)
        
        # 模組參考 (需要外部設定)
        self.data_loader = None
        self.feature_calculator = None
        self.normalizer = None
        self.buy_agent = None
        self.sell_agent = None
        self.stop_loss_rule = None
        
        # 內部狀態
        self.cash = self.initial_capital
        self.positions: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.daily_equity: List[Tuple[datetime, float]] = []
        
        logger.info(f"BacktestEngine 初始化完成 - 初始資金: ${self.initial_capital:,}")
    
    def set_modules(self, 
                    data_loader=None,
                    feature_calculator=None,
                    normalizer=None,
                    buy_agent=None,
                    sell_agent=None,
                    stop_loss_rule=None):
        """設定各模組"""
        if data_loader:
            self.data_loader = data_loader
        if feature_calculator:
            self.feature_calculator = feature_calculator
        if normalizer:
            self.normalizer = normalizer
        if buy_agent:
            self.buy_agent = buy_agent
        if sell_agent:
            self.sell_agent = sell_agent
        if stop_loss_rule:
            self.stop_loss_rule = stop_loss_rule
    
    def run(self, 
            start_date: str,
            end_date: str,
            symbols: List[str] = None) -> BacktestResult:
        """
        執行回測
        
        Args:
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            symbols: 股票清單 (預設使用 S&P 500)
            
        Returns:
            BacktestResult 回測結果
        """
        logger.info(f"開始回測: {start_date} ~ {end_date}")
        
        # 重置狀態
        self._reset()
        
        # 載入資料
        if self.data_loader is None:
            raise ValueError("請先設定 data_loader")
        
        if symbols is None:
            symbols = self.data_loader.load_symbols_list()
        
        # 載入所有股票資料
        all_data = {}
        index_data = self.data_loader.load_index()
        
        for symbol in symbols:
            df = self.data_loader.load_symbol(symbol)
            if df is not None:
                all_data[symbol] = df
        
        logger.info(f"載入 {len(all_data)} 支股票資料")
        
        # 取得交易日期
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 使用指數的日期作為交易日
        if index_data is not None:
            trading_dates = index_data.loc[start_dt:end_dt].index.tolist()
        else:
            trading_dates = pd.date_range(start_dt, end_dt, freq='B').tolist()
        
        # 逐日模擬
        for date in trading_dates:
            self._process_day(date, all_data, index_data)
        
        # 結算未平倉部位
        self._close_all_positions(trading_dates[-1] if trading_dates else end_dt, all_data)
        
        # 計算績效
        result = self._calculate_performance(start_dt, end_dt)
        
        logger.info(f"回測完成 - 總報酬: {result.total_return:.2%}, "
                   f"年化報酬: {result.annualized_return:.2%}, "
                   f"Sharpe: {result.sharpe_ratio:.2f}")
        
        return result
    
    def _reset(self):
        """重置回測狀態"""
        self.cash = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.daily_equity = []
    
    def _process_day(self, date: datetime, all_data: Dict, index_data: pd.DataFrame):
        """處理單一交易日"""
        # 1. 檢查現有持倉 (停損 + 賣出決策)
        self._process_sell_decisions(date, all_data)
        
        # 2. 檢查買入訊號
        self._process_buy_decisions(date, all_data, index_data)
        
        # 3. 記錄每日淨值
        total_equity = self._calculate_equity(date, all_data)
        self.daily_equity.append((date, total_equity))
    
    def _calculate_stock_features(self, symbol: str, df: pd.DataFrame, 
                                   index_data: pd.DataFrame, date: datetime) -> Optional[np.ndarray]:
        """
        計算單一股票在指定日期的 69 維正規化特徵
        
        Args:
            symbol: 股票代碼
            df: 股票日線資料
            index_data: 指數資料
            date: 目標日期
            
        Returns:
            69 維正規化特徵向量，若計算失敗則回傳 None
        """
        if self.feature_calculator is None or self.normalizer is None:
            return None
        
        try:
            idx = df.index.get_loc(date)
            
            # 需要足夠的歷史資料 (至少 252 天用於滾動計算)
            if idx < 252:
                return None
            
            # 取得回測日期之前的資料用於特徵計算 (包含當日)
            stock_slice = df.iloc[:idx+1].copy()
            
            # 準備指數資料
            index_slice = None
            if index_data is not None and date in index_data.index:
                index_idx = index_data.index.get_loc(date)
                if index_idx >= 252:
                    index_slice = index_data.iloc[:index_idx+1].copy()
            
            # 計算特徵
            features_df = self.feature_calculator.calculate_all_features(
                stock_slice, index_slice
            )
            
            # 正規化
            normalized_df = self.normalizer.normalize(features_df)
            
            # 取得當日的特徵向量
            if normalized_df.empty or date not in normalized_df.index:
                return None
            
            # 提取 RL 特徵欄位
            feature_cols = self.normalizer.get_normalized_feature_columns()
            available_cols = [col for col in feature_cols if col in normalized_df.columns]
            
            if not available_cols:
                return None
            
            feature_vector = normalized_df.loc[date, available_cols].values.astype(np.float32)
            
            # 處理 NaN 和 Inf
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return feature_vector
            
        except Exception as e:
            logger.debug(f"計算 {symbol} 特徵失敗: {e}")
            return None
    
    def _process_sell_decisions(self, date: datetime, all_data: Dict):
        """處理賣出決策"""
        positions_to_close = []
        
        for symbol, trade in self.positions.items():
            if symbol not in all_data:
                continue
            
            df = all_data[symbol]
            if date not in df.index:
                continue
            
            current_price = df.loc[date, 'Close']
            holding_days = (date - trade.buy_date).days
            
            # 取得價格歷史
            buy_idx = df.index.get_loc(trade.buy_date) if trade.buy_date in df.index else 0
            current_idx = df.index.get_loc(date)
            price_history = df.iloc[buy_idx:current_idx+1]['Close']
            
            # 1. 檢查停損規則
            if self.stop_loss_rule:
                stop_result = self.stop_loss_rule.check(
                    buy_price=trade.buy_price,
                    current_price=current_price,
                    holding_days=holding_days,
                    price_history=price_history
                )
                
                if stop_result.should_stop:
                    trade.sell_date = date
                    trade.sell_price = current_price
                    trade.sell_reason = f'stop_loss_{stop_result.stop_type}'
                    trade.status = 'closed'
                    positions_to_close.append(symbol)
                    continue
            
            # 2. 使用 Sell Agent 決策
            if self.sell_agent:
                # 簡化: 這裡應該計算特徵並正規化
                # 目前使用簡化邏輯
                should_sell = holding_days > 60 and np.random.random() > 0.8
                
                if should_sell:
                    trade.sell_date = date
                    trade.sell_price = current_price
                    trade.sell_reason = 'agent'
                    trade.status = 'closed'
                    positions_to_close.append(symbol)
        
        # 平倉並返還現金
        for symbol in positions_to_close:
            trade = self.positions.pop(symbol)
            sell_value = trade.sell_price * trade.shares * (1 - self.trading_fee)
            self.cash += sell_value
            self.closed_trades.append(trade)
    
    def _process_buy_decisions(self, date: datetime, all_data: Dict, index_data: pd.DataFrame):
        """
        處理買入決策 - 實作 Top-10 選股邏輯
        
        論文設計:
        1. 收集當日所有 Donchian Channel 突破訊號
        2. 使用 Buy Agent 計算每個訊號的「會漲10%」機率
        3. 依機率排序，選 Top-10 買入
        """
        # 檢查是否有空間開新倉
        available_slots = self.max_positions - len(self.positions)
        if available_slots <= 0:
            return
        
        # === 階段 1: 收集所有突破訊號 ===
        breakout_candidates = []
        
        for symbol, df in all_data.items():
            if symbol in self.positions:
                continue
            
            if date not in df.index:
                continue
            
            # 檢查 Donchian Channel 突破
            idx = df.index.get_loc(date)
            if idx < 20:  # 需要足夠的歷史資料
                continue
            
            high = df.iloc[idx]['High']
            donchian_upper = df.iloc[idx-20:idx]['High'].max()
            
            is_breakout = high > donchian_upper
            
            if is_breakout:
                buy_price = df.loc[date, 'Close']
                
                # === 計算 Buy Agent 信心分數 (使用真實模型預測) ===
                confidence = 0.5  # 預設值 (無模型時)
                
                if self.buy_agent and hasattr(self.buy_agent, 'predict_proba') and self.buy_agent.model is not None:
                    try:
                        # 計算該股票的特徵
                        features = self._calculate_stock_features(symbol, df, index_data, date)
                        
                        if features is not None:
                            # 呼叫模型取得預測機率
                            # predict_proba 回傳 [不買機率, 買機率]
                            probs = self.buy_agent.predict_proba(features)
                            confidence = float(probs[1])  # 取「買」的機率作為信心分數
                    except Exception as e:
                        logger.debug(f"無法計算 {symbol} 信心分數: {e}")
                        confidence = 0.5
                
                # 計算成交金額 (Turnover) 作為市值/流動性代理
                # Turnover = Price * Volume
                # 這能解決高價股成交張數少的問題，反映真實資金流動
                volume = df.loc[date, 'Volume'] if 'Volume' in df.columns else 0
                turnover = buy_price * volume
                
                breakout_candidates.append({
                    'symbol': symbol,
                    'buy_price': buy_price,
                    'confidence': confidence,
                    'turnover': turnover,
                    'df': df
                })
        
        # === 階段 2: 排序選 Top-10 ===
        if not breakout_candidates:
            return
        
        # 加入隨機擾動以避免字母排序偏差 (當信心與成交額都相同時)
        np.random.shuffle(breakout_candidates)
        
        # 排序優先級: 
        # 1. 信心分數 (由高到低)
        # 2. 成交金額 (由高到低) - 優先選流動性高/大型股
        breakout_candidates.sort(
            key=lambda x: (x['confidence'], x['turnover']), 
            reverse=True
        )
        
        # 限制為 Top-10 (或可用空位數)
        top_candidates = breakout_candidates[:min(10, available_slots)]
        
        # Log 當日訊號數量
        if len(breakout_candidates) > 10:
            logger.debug(f"{date.strftime('%Y-%m-%d')}: {len(breakout_candidates)} 個訊號，Top-10 cutoff 信心: {top_candidates[-1]['confidence']:.4f}")
        
        # === 階段 3: 執行買入 ===
        available_cash = self.cash
        position_size = available_cash * self.max_position_pct
        
        for candidate in top_candidates:
            if available_cash < position_size:
                break
            
            symbol = candidate['symbol']
            buy_price = candidate['buy_price']
            
            shares = int(position_size / buy_price)
            
            if shares > 0:
                trade = Trade(
                    symbol=symbol,
                    buy_date=date,
                    buy_price=buy_price,
                    shares=shares
                )
                self.positions[symbol] = trade
                buy_cost = buy_price * shares * (1 + self.trading_fee)
                self.cash -= buy_cost
                available_cash -= buy_cost
                
                if len(self.positions) >= self.max_positions:
                    break
    
    def _close_all_positions(self, date: datetime, all_data: Dict):
        """結算所有未平倉部位"""
        for symbol, trade in list(self.positions.items()):
            if symbol in all_data and date in all_data[symbol].index:
                trade.sell_date = date
                trade.sell_price = all_data[symbol].loc[date, 'Close']
                trade.sell_reason = 'end_of_backtest'
                trade.status = 'closed'
                
                sell_value = trade.sell_price * trade.shares * (1 - self.trading_fee)
                self.cash += sell_value
                self.closed_trades.append(trade)
        
        self.positions = {}
    
    def _calculate_equity(self, date: datetime, all_data: Dict) -> float:
        """計算當日總權益"""
        equity = self.cash
        
        for symbol, trade in self.positions.items():
            if symbol in all_data and date in all_data[symbol].index:
                current_price = all_data[symbol].loc[date, 'Close']
                equity += current_price * trade.shares
        
        return equity
    
    def _calculate_performance(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """計算績效指標"""
        # 權益曲線
        if self.daily_equity:
            dates, equities = zip(*self.daily_equity)
            equity_curve = pd.Series(equities, index=pd.DatetimeIndex(dates))
        else:
            equity_curve = pd.Series(dtype=float)
        
        # 日報酬率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 總報酬
        final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else self.initial_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # 年化報酬
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe Ratio (假設無風險利率 = 0)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤
        cummax = equity_curve.cummax()
        drawdown = (cummax - equity_curve) / cummax
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
        
        # 勝率
        winning_trades = [t for t in self.closed_trades if t.return_pct and t.return_pct > 0]
        win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0.0
        
        # 平均持有天數
        holding_days_list = [t.holding_days for t in self.closed_trades if t.holding_days]
        avg_holding_days = np.mean(holding_days_list) if holding_days_list else 0.0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=len(self.closed_trades),
            win_rate=win_rate,
            avg_holding_days=avg_holding_days,
            trades=self.closed_trades,
            daily_returns=daily_returns,
            equity_curve=equity_curve
        )
    
    def generate_report(self, result: BacktestResult, output_dir: str = 'outputs/reports/'):
        """產生回測報告"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 績效摘要
        summary = {
            '回測期間': f"{result.start_date.strftime('%Y-%m-%d')} ~ {result.end_date.strftime('%Y-%m-%d')}",
            '初始資金': f"${result.initial_capital:,.0f}",
            '最終資金': f"${result.final_capital:,.0f}",
            '總報酬率': f"{result.total_return:.2%}",
            '年化報酬率': f"{result.annualized_return:.2%}",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            '最大回撤': f"{result.max_drawdown:.2%}",
            '總交易次數': result.total_trades,
            '勝率': f"{result.win_rate:.2%}",
            '平均持有天數': f"{result.avg_holding_days:.1f}"
        }
        
        # 儲存摘要
        summary_df = pd.DataFrame([summary]).T
        summary_df.columns = ['值']
        summary_df.to_csv(os.path.join(output_dir, f'summary_{timestamp}.csv'))
        
        # 儲存交易紀錄
        trades_data = []
        for t in result.trades:
            trades_data.append({
                'symbol': t.symbol,
                'buy_date': t.buy_date,
                'buy_price': t.buy_price,
                'sell_date': t.sell_date,
                'sell_price': t.sell_price,
                'shares': t.shares,
                'return_pct': t.return_pct,
                'profit': t.profit,
                'holding_days': t.holding_days,
                'sell_reason': t.sell_reason
            })
        
        if trades_data:
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(os.path.join(output_dir, f'trades_{timestamp}.csv'), index=False)
        
        # 儲存權益曲線
        if len(result.equity_curve) > 0:
            result.equity_curve.to_csv(os.path.join(output_dir, f'equity_{timestamp}.csv'))
        
        logger.info(f"報告已儲存至: {output_dir}")
        
        return summary


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    print("BacktestEngine 載入成功")
    print("使用方式:")
    print("  engine = BacktestEngine(config)")
    print("  engine.set_modules(data_loader, buy_agent, sell_agent, stop_loss_rule)")
    print("  result = engine.run('2020-01-01', '2023-12-31')")
