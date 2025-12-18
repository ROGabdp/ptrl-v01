"""
FeatureCalculator - 論文定義的 69 個技術特徵計算模組

根據論文 "Pro Trader RL" 的 Table 1-4，計算以下特徵:
- 基本變數 (9 個): Open, High, Low, Close, Volume, HA系列
- 技術指標 (21 個): Return, ATR, Stock(N), Super Trend, MFI, RSI, Donchian, AVG Stock
- 股價指數變數 (13 個): DJI ATR, Index(N)
- 相對強度變數 (26 個): RS, RS AVG, RS Rate, RS Rate(N), Up Stock, Down Stock
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger


class FeatureCalculator:
    """
    計算論文定義的 69 個技術特徵
    
    使用方式:
        calc = FeatureCalculator(config)
        df_features = calc.calculate_all_features(df_stock, df_index)
    """
    
    def __init__(self, config: dict = None):
        """
        初始化特徵計算器
        
        Args:
            config: 設定字典，包含各項指標參數
        """
        config = config or {}
        
        # ATR 相關參數
        self.atr_period = config.get('atr_period', 10)
        
        # Donchian Channel 參數
        self.donchian_period = config.get('donchian_period', 20)
        
        # RSI / MFI 參數
        self.rsi_period = config.get('rsi_period', 14)
        self.mfi_period = config.get('mfi_period', 14)
        
        # Super Trend 參數
        self.super_trend_periods = config.get('super_trend_periods', [14, 21])
        self.super_trend_multipliers = config.get('super_trend_multipliers', [2, 1])
        
        # 相對強度期間 (1-12 個月)
        self.rs_periods = config.get('rs_periods', list(range(1, 13)))
        
        # RS Rate 移動平均期間
        self.rs_rate_periods = config.get('rs_rate_periods', [5, 10, 20, 40])
        
        logger.info("FeatureCalculator 初始化完成")
    
    # =========================================================================
    # 主要計算函數
    # =========================================================================
    
    def calculate_all_features(self, df: pd.DataFrame, 
                                index_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        計算所有 69 個特徵
        
        Args:
            df: 股票 OHLCV DataFrame (需包含 Open, High, Low, Close, Volume)
            index_df: 指數 OHLCV DataFrame (可選，用於計算相對強度)
            
        Returns:
            包含所有特徵的 DataFrame
        """
        logger.info("開始計算 69 個特徵...")
        
        # 確保有日期索引
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        # 複製資料以避免修改原始資料
        result = df.copy()
        
        # 1. 計算基本變數 (9 個)
        result = self.calculate_basic_features(result)
        
        # 2. 計算技術指標 (21 個)
        result = self.calculate_technical_indicators(result)
        
        # 3. 如果有指數資料，計算指數變數和相對強度
        if index_df is not None:
            if 'Date' in index_df.columns:
                index_df = index_df.set_index('Date')
            
            # 計算指數特徵 (13 個)
            result = self.calculate_index_features(result, index_df)
            
            # 計算相對強度 (26 個)
            result = self.calculate_relative_strength(result, index_df)
        
        logger.info(f"特徵計算完成，共 {len(result.columns)} 個欄位")
        return result
    
    # =========================================================================
    # 基本變數 (9 個)
    # =========================================================================
    
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 9 個基本變數
        
        包含:
        - Open, High, Low, Close, Volume (5 個)
        - HA_Open, HA_High, HA_Low, HA_Close (4 個 Heikin Ashi)
        """
        result = df.copy()
        
        # Heikin Ashi 計算
        result['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # HA_Open: 第一根 = (Open + Close) / 2，後續 = 前一根 HA 的 (HA_Open + HA_Close) / 2
        ha_open = [(df['Open'].iloc[0] + df['Close'].iloc[0]) / 2]
        for i in range(1, len(df)):
            ha_open.append((ha_open[i-1] + result['HA_Close'].iloc[i-1]) / 2)
        result['HA_Open'] = ha_open
        
        result['HA_High'] = result[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        result['HA_Low'] = result[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
        
        logger.debug("基本變數計算完成 (9 個)")
        return result
    
    # =========================================================================
    # 技術指標 (21 個)
    # =========================================================================
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 21 個技術指標
        
        包含:
        - Return (1 個): 日報酬率
        - ATR (1 個): 平均真實區間
        - Stock_N (12 個): 波動性比較 (N = 1-12 個月)
        - Super Trend (2 個): 趨勢指標
        - MFI (1 個): 資金流量指標
        - RSI (1 個): 相對強弱指標
        - Donchian (2 個): 唐奇安通道上下軌
        - AVG_Stock (1 個): Stock(N) 平均
        """
        result = df.copy()
        
        # 1. Return (日報酬率)
        result['Return'] = result['Close'].pct_change()
        
        # 2. ATR (Average True Range)
        result['ATR'] = self._calculate_atr(result, self.atr_period)
        
        # 3. Stock(N) - 當前 ATR 與 N 個月前的比較
        trading_days_per_month = 21
        for n in range(1, 13):
            shift_days = n * trading_days_per_month
            result[f'Stock_{n}'] = result['ATR'] / result['ATR'].shift(shift_days)
        
        # 4. AVG_Stock - Stock(N) 的平均值
        stock_cols = [f'Stock_{n}' for n in [1, 3, 6, 12]]
        result['AVG_Stock'] = result[stock_cols].mean(axis=1)
        
        # 5. Super Trend
        for period, mult in zip(self.super_trend_periods, self.super_trend_multipliers):
            result[f'SuperTrend_{period}'] = self._calculate_supertrend(result, period, mult)
        
        # 6. MFI (Money Flow Index)
        result['MFI'] = self._calculate_mfi(result, self.mfi_period)
        
        # 7. RSI (Relative Strength Index)
        result['RSI'] = self._calculate_rsi(result, self.rsi_period)
        
        # 8. Donchian Channel
        result['Donchian_Upper'] = result['High'].rolling(window=self.donchian_period).max()
        result['Donchian_Lower'] = result['Low'].rolling(window=self.donchian_period).min()
        
        logger.debug("技術指標計算完成 (21 個)")
        return result
    
    # =========================================================================
    # 指數變數 (13 個)
    # =========================================================================
    
    def calculate_index_features(self, df: pd.DataFrame, 
                                  index_df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 14 個指數變數 (論文 13 個 + 1 個 Index_Return)
        
        包含:
        - Index_ATR (1 個): 指數 ATR
        - Index_N (12 個): 指數波動性比較 (N = 1-12 個月)
        - Index_Return (1 個): 指數日報酬率
        """
        result = df.copy()
        
        # 確保指數資料與股票資料日期對齊
        index_aligned = index_df.reindex(df.index)
        
        # 1. Index_ATR (指數的 ATR)
        index_atr = self._calculate_atr(index_aligned, self.atr_period)
        result['Index_ATR'] = index_atr
        
        # 2. Index(N) - 指數 ATR 與 N 個月前的比較
        trading_days_per_month = 21
        for n in range(1, 13):
            shift_days = n * trading_days_per_month
            result[f'Index_{n}'] = index_atr / index_atr.shift(shift_days)
        
        # 3. Index_Return - 指數日報酬率 (額外補充)
        result['Index_Return'] = index_aligned['Close'].pct_change()
        
        logger.debug("指數變數計算完成 (14 個)")
        return result
    
    # =========================================================================
    # 相對強度變數 (26 個)
    # =========================================================================
    
    def calculate_relative_strength(self, df: pd.DataFrame,
                                     index_df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 26 個相對強度變數
        
        包含:
        - RS_N (12 個): Stock(N) / Index(N)
        - RS_AVG (1 個): RS 平均 (1,3,6,12 個月)
        - AVG_Index (1 個): Index(N) 平均
        - RS_AVG_Short (1 個): RS 短期平均 (1,3 個月)
        - RS_AVG_Long (1 個): RS 長期平均 (6,12 個月)  
        - RS_Rate (1 個): RS 轉換為 0-100
        - RS_Rate_N (4 個): RS_Rate 的移動平均
        - RS_Momentum (1 個): RS 動能
        - RS_Trend (1 個): RS 趨勢
        - Up_Stock (1 個): 上漲家數
        - Down_Stock (1 個): 下跌家數
        
        總計: 12 + 1 + 1 + 1 + 1 + 1 + 4 + 1 + 1 + 1 + 1 + 1 = 26 個
        
        注意: Up_Stock 和 Down_Stock 需要多股票資料，這裡先預留欄位
        """
        result = df.copy()
        
        # 1. RS(N) = Stock(N) / Index(N)  [12 個]
        for n in range(1, 13):
            stock_col = f'Stock_{n}'
            index_col = f'Index_{n}'
            if stock_col in result.columns and index_col in result.columns:
                result[f'RS_{n}'] = result[stock_col] / result[index_col]
        
        # 2. RS_AVG - RS 的平均 (1, 3, 6, 12 個月)  [1 個]
        rs_cols = [f'RS_{n}' for n in [1, 3, 6, 12] if f'RS_{n}' in result.columns]
        if rs_cols:
            result['RS_AVG'] = result[rs_cols].mean(axis=1)
        
        # 3. AVG_Index - Index(N) 的平均  [1 個]
        index_cols = [f'Index_{n}' for n in [1, 3, 6, 12] if f'Index_{n}' in result.columns]
        if index_cols:
            result['AVG_Index'] = result[index_cols].mean(axis=1)
        
        # 4. RS_AVG_Short - RS 短期平均 (1, 3 個月)  [1 個]
        rs_short_cols = [f'RS_{n}' for n in [1, 3] if f'RS_{n}' in result.columns]
        if rs_short_cols:
            result['RS_AVG_Short'] = result[rs_short_cols].mean(axis=1)
        
        # 5. RS_AVG_Long - RS 長期平均 (6, 12 個月)  [1 個]
        rs_long_cols = [f'RS_{n}' for n in [6, 12] if f'RS_{n}' in result.columns]
        if rs_long_cols:
            result['RS_AVG_Long'] = result[rs_long_cols].mean(axis=1)
        
        # 6. RS_Rate - 將 RS 轉換為 0-100 範圍  [1 個]
        if 'RS_AVG' in result.columns:
            result['RS_Rate'] = result['RS_AVG'].rolling(window=252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
            )
        
        # 7. RS_Rate(N) - RS_Rate 的移動平均  [4 個]
        if 'RS_Rate' in result.columns:
            for period in self.rs_rate_periods:
                result[f'RS_Rate_{period}'] = result['RS_Rate'].rolling(window=period).mean()
        
        # 8. RS_Momentum - RS 動能 (短期 vs 長期)  [1 個]
        if 'RS_AVG_Short' in result.columns and 'RS_AVG_Long' in result.columns:
            result['RS_Momentum'] = result['RS_AVG_Short'] / result['RS_AVG_Long']
        
        # 9. RS_Trend - RS 趨勢 (當前 vs N日前)  [1 個]
        if 'RS_AVG' in result.columns:
            result['RS_Trend'] = result['RS_AVG'] / result['RS_AVG'].shift(21)
        
        # 10. Up_Stock / Down_Stock - 需要多股票資料  [2 個]
        # 這些欄位會在 StrategyOrchestrator 中根據全市場資料計算
        result['Up_Stock'] = np.nan
        result['Down_Stock'] = np.nan
        
        logger.debug("相對強度變數計算完成 (26 個)")
        return result
    
    # =========================================================================
    # 輔助計算函數
    # =========================================================================
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """計算 ATR (Average True Range)"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR = TR 的指數移動平均
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算 RSI (Relative Strength Index)"""
        delta = df['Close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算 MFI (Money Flow Index)"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        
        # 區分正負資金流
        delta = typical_price.diff()
        positive_flow = raw_money_flow.where(delta > 0, 0)
        negative_flow = raw_money_flow.where(delta < 0, 0)
        
        # 計算 MFI
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_supertrend(self, df: pd.DataFrame, 
                               period: int = 14, multiplier: float = 2) -> pd.Series:
        """
        計算 Super Trend 指標
        
        Returns:
            1 表示上升趨勢，-1 表示下降趨勢
        """
        atr = self._calculate_atr(df, period)
        hl2 = (df['High'] + df['Low']) / 2
        
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif df['Close'].iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1] if not np.isnan(direction.iloc[i-1]) else 1
            
            supertrend.iloc[i] = direction.iloc[i]
        
        return supertrend
    
    # =========================================================================
    # Donchian Channel 買入訊號
    # =========================================================================
    
    def generate_buy_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        根據 Donchian Channel 策略產生買入訊號
        
        買入條件:
        - 當前最高價 > 上通道 (過去 20 天最高價)
        - 且前一天沒有買入訊號
        """
        if 'Donchian_Upper' not in df.columns:
            df['Donchian_Upper'] = df['High'].rolling(window=self.donchian_period).max()
        
        # 買入訊號: 突破上通道
        signal = (df['High'] > df['Donchian_Upper'].shift(1))
        
        # 移除連續訊號 (只保留第一天)
        signal = signal & (~signal.shift(1).fillna(False))
        
        return signal.astype(int)
    
    def generate_sell_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        根據 Donchian Channel 策略產生賣出訊號
        
        賣出條件:
        - 當前最低價 < 下通道 (過去 20 天最低價)
        """
        if 'Donchian_Lower' not in df.columns:
            df['Donchian_Lower'] = df['Low'].rolling(window=self.donchian_period).min()
        
        # 賣出訊號: 跌破下通道
        signal = (df['Low'] < df['Donchian_Lower'].shift(1))
        
        return signal.astype(int)


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'd:/000-github-repositories/ptrl-v01')
    
    from src.data.data_loader import DataLoader
    
    # 設定
    config = {
        'data_dir': 'data/raw/',
        'index_symbol': '^GSPC',
        'symbols_file': 'config/sp500_symbols.txt'
    }
    
    # 載入資料
    loader = DataLoader(config)
    df_stock = loader.load_symbol('AAPL')
    df_index = loader.load_index()
    
    if df_stock is not None and df_index is not None:
        # 計算特徵
        calc = FeatureCalculator()
        df_features = calc.calculate_all_features(df_stock, df_index)
        
        print(f"計算完成，共 {len(df_features.columns)} 個特徵")
        print("\n特徵清單:")
        print(df_features.columns.tolist())
        print("\n前 5 筆資料:")
        print(df_features.head())
    else:
        print("請先執行 DataLoader 下載資料")
