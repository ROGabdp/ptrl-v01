"""
StrategyOrchestrator - 策略協調器

整合所有模組的頂層控制器:
- 每日訊號掃描流程
- 持倉檢查流程
- 報告生成

使用方式:
    orchestrator = StrategyOrchestrator(config)
    report = orchestrator.run_daily(date)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from .portfolio_manager import PortfolioManager, Trade
from .trade_executor import TradeExecutor


@dataclass
class DailyReport:
    """每日運營報告"""
    date: datetime
    buy_signals: List[Dict] = field(default_factory=list)
    executed_buys: List[Trade] = field(default_factory=list)
    sell_signals: List[Dict] = field(default_factory=list)
    executed_sells: List[Trade] = field(default_factory=list)
    portfolio_value: float = 0.0
    cash: float = 0.0
    position_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            'date': self.date.isoformat(),
            'buy_signals': len(self.buy_signals),
            'executed_buys': len(self.executed_buys),
            'sell_signals': len(self.sell_signals),
            'executed_sells': len(self.executed_sells),
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_count': self.position_count
        }


class StrategyOrchestrator:
    """
    策略協調器
    
    整合所有模組的頂層控制器:
    - 資料載入
    - 特徵計算
    - 訊號掃描
    - 持倉管理
    - 報告生成
    
    使用方式:
        orchestrator = StrategyOrchestrator(config)
        orchestrator.set_modules(data_loader, feature_calc, normalizer, buy_agent, sell_agent, stop_loss)
        report = orchestrator.run_daily(date)
    """
    
    def __init__(self, config: dict = None):
        """
        初始化策略協調器
        
        Args:
            config: 設定字典
        """
        config = config or {}
        
        # 投資組合設定
        portfolio_config = config.get('portfolio', {})
        self.portfolio = PortfolioManager(portfolio_config)
        
        # 交易執行器設定
        executor_config = config.get('executor', {})
        self.executor = TradeExecutor(self.portfolio, executor_config)
        
        # 特徵設定
        feature_config = config.get('features', {})
        self.donchian_period = feature_config.get('donchian_period', 20)
        
        # 外部模組參考
        self.data_loader = None
        self.feature_calculator = None
        self.normalizer = None
        
        # 日誌設定
        self.output_dir = config.get('output_dir', 'outputs/daily/')
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("StrategyOrchestrator 初始化完成")
    
    def set_modules(self, data_loader=None, feature_calculator=None,
                    normalizer=None, buy_agent=None, sell_agent=None,
                    stop_loss_rule=None):
        """設定各模組"""
        if data_loader:
            self.data_loader = data_loader
        if feature_calculator:
            self.feature_calculator = feature_calculator
        if normalizer:
            self.normalizer = normalizer
        if buy_agent or sell_agent:
            self.executor.set_agents(buy_agent, sell_agent)
        if stop_loss_rule:
            self.executor.set_stop_loss(stop_loss_rule)
    
    def scan_buy_signals(self, date: datetime) -> List[Dict]:
        """
        掃描買入訊號
        
        Args:
            date: 目標日期
            
        Returns:
            買入訊號列表
        """
        if self.data_loader is None:
            logger.error("未設定 data_loader")
            return []
        
        signals = []
        symbols = self.data_loader.load_symbols_list()
        index_data = self.data_loader.load_index()
        
        for symbol in symbols:
            df = self.data_loader.load_symbol(symbol)
            if df is None or date not in df.index:
                continue
            
            try:
                idx = df.index.get_loc(date)
                if idx < self.donchian_period:
                    continue
                
                # 檢查 Donchian Channel 突破
                high = df.iloc[idx]['High']
                donchian_upper = df.iloc[idx-self.donchian_period:idx]['High'].max()
                
                if high > donchian_upper:
                    # 計算特徵
                    if self.feature_calculator and self.normalizer:
                        df_features = self.feature_calculator.calculate_all_features(df, index_data)
                        df_normalized = self.normalizer.normalize(df_features)
                        
                        feature_cols = self.normalizer.get_normalized_feature_columns()
                        available_cols = [c for c in feature_cols if c in df_normalized.columns]
                        features = df_normalized.loc[date, available_cols].values.astype(np.float32)
                        features = np.nan_to_num(features, nan=0.0)
                    else:
                        features = np.zeros(69, dtype=np.float32)
                    
                    price = df.loc[date, 'Close']
                    
                    signals.append({
                        'symbol': symbol,
                        'date': date,
                        'price': price,
                        'donchian_upper': donchian_upper,
                        'features': features
                    })
                    
            except Exception as e:
                logger.debug(f"處理 {symbol} 時錯誤: {e}")
                continue
        
        logger.info(f"掃描到 {len(signals)} 個買入訊號")
        return signals
    
    def check_positions(self, date: datetime) -> List[Dict]:
        """
        檢查持倉的賣出條件
        
        Args:
            date: 目標日期
            
        Returns:
            賣出建議列表
        """
        if self.data_loader is None:
            logger.error("未設定 data_loader")
            return []
        
        sell_signals = []
        index_data = self.data_loader.load_index()
        
        for symbol, position in self.portfolio.get_positions().items():
            df = self.data_loader.load_symbol(symbol)
            if df is None or date not in df.index:
                continue
            
            try:
                current_price = df.loc[date, 'Close']
                holding_days = position.get_holding_days(date)
                current_return = position.get_return(current_price)
                
                # 取得價格歷史
                buy_idx = df.index.get_loc(position.buy_date) if position.buy_date in df.index else 0
                current_idx = df.index.get_loc(date)
                price_history = df.iloc[buy_idx:current_idx+1]['Close']
                
                # 計算特徵
                if self.feature_calculator and self.normalizer:
                    df_features = self.feature_calculator.calculate_all_features(df, index_data)
                    df_normalized = self.normalizer.normalize(df_features)
                    
                    feature_cols = self.normalizer.get_normalized_feature_columns()
                    available_cols = [c for c in feature_cols if c in df_normalized.columns]
                    features = df_normalized.loc[date, available_cols].values.astype(np.float32)
                    features = np.nan_to_num(features, nan=0.0)
                else:
                    features = np.zeros(69, dtype=np.float32)
                
                sell_signals.append({
                    'symbol': symbol,
                    'date': date,
                    'price': current_price,
                    'features': features,
                    'price_history': price_history,
                    'holding_days': holding_days,
                    'current_return': current_return
                })
                
            except Exception as e:
                logger.warning(f"檢查 {symbol} 持倉時錯誤: {e}")
                continue
        
        return sell_signals
    
    def run_daily(self, date: datetime, update_data: bool = False) -> DailyReport:
        """
        執行每日運營流程
        
        Args:
            date: 目標日期
            update_data: 是否更新資料
            
        Returns:
            DailyReport 每日報告
        """
        logger.info(f"=== 每日運營: {date.strftime('%Y-%m-%d')} ===")
        
        report = DailyReport(date=date)
        
        # 更新資料 (可選)
        if update_data and self.data_loader:
            symbols = self.data_loader.load_symbols_list()
            for symbol in symbols[:50]:  # 限制更新數量
                try:
                    self.data_loader.update_symbol(symbol)
                except:
                    pass
        
        # 1. 檢查持倉 (優先處理賣出)
        sell_signals = self.check_positions(date)
        report.sell_signals = sell_signals
        
        for signal in sell_signals:
            trade = self.executor.process_sell_decision(
                symbol=signal['symbol'],
                features=signal['features'],
                current_price=signal['price'],
                current_date=date,
                price_history=signal['price_history']
            )
            if trade:
                report.executed_sells.append(trade)
        
        # 2. 掃描買入訊號
        buy_signals = self.scan_buy_signals(date)
        report.buy_signals = buy_signals
        
        # 加入候選並執行
        self.executor.reset_daily()
        for signal in buy_signals:
            self.executor.add_buy_candidate(
                symbol=signal['symbol'],
                features=signal['features'],
                price=signal['price'],
                date=date
            )
        
        executed_buys = self.executor.execute_daily_buys()
        report.executed_buys = executed_buys
        
        # 3. 記錄投資組合狀態
        current_prices = {}
        for symbol in self.portfolio.get_positions():
            df = self.data_loader.load_symbol(symbol) if self.data_loader else None
            if df is not None and date in df.index:
                current_prices[symbol] = df.loc[date, 'Close']
        
        self.portfolio.record_equity(date, current_prices)
        
        report.portfolio_value = self.portfolio.get_equity(current_prices)
        report.cash = self.portfolio.get_available_cash()
        report.position_count = self.portfolio.get_position_count()
        
        # 輸出摘要
        logger.info(f"買入訊號: {len(buy_signals)}, 執行買入: {len(executed_buys)}")
        logger.info(f"賣出檢查: {len(sell_signals)}, 執行賣出: {len(report.executed_sells)}")
        logger.info(f"投資組合價值: ${report.portfolio_value:,.2f}")
        
        return report
    
    def generate_report(self, daily_report: DailyReport):
        """產生每日報告檔案"""
        date_str = daily_report.date.strftime('%Y%m%d')
        
        # 買入報告
        if daily_report.buy_signals:
            buy_df = pd.DataFrame([{
                'symbol': s['symbol'],
                'price': s['price'],
                'donchian_upper': s.get('donchian_upper', 0)
            } for s in daily_report.buy_signals])
            buy_df.to_csv(
                os.path.join(self.output_dir, f'buy_signals_{date_str}.csv'),
                index=False
            )
        
        # 賣出報告
        if daily_report.executed_sells:
            sell_df = pd.DataFrame([t.to_dict() for t in daily_report.executed_sells])
            sell_df.to_csv(
                os.path.join(self.output_dir, f'sell_trades_{date_str}.csv'),
                index=False
            )
        
        # 摘要報告
        with open(os.path.join(self.output_dir, f'daily_report_{date_str}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Pro Trader RL 每日報告\n")
            f.write(f"日期: {daily_report.date.strftime('%Y-%m-%d')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"投資組合價值: ${daily_report.portfolio_value:,.2f}\n")
            f.write(f"現金: ${daily_report.cash:,.2f}\n")
            f.write(f"持倉數: {daily_report.position_count}\n\n")
            f.write(f"買入訊號: {len(daily_report.buy_signals)} 個\n")
            f.write(f"執行買入: {len(daily_report.executed_buys)} 筆\n")
            f.write(f"執行賣出: {len(daily_report.executed_sells)} 筆\n")
        
        logger.info(f"報告已儲存至: {self.output_dir}")
    
    def reset(self):
        """重置策略"""
        self.portfolio.reset()
        self.executor.reset_daily()
        logger.info("策略已重置")


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    print("=== StrategyOrchestrator 測試 ===")
    
    config = {
        'portfolio': {
            'initial_capital': 10000,
            'max_positions': 10
        }
    }
    
    orchestrator = StrategyOrchestrator(config)
    print("StrategyOrchestrator 初始化成功")
    print(f"投資組合初始資金: ${orchestrator.portfolio.initial_capital:,}")
