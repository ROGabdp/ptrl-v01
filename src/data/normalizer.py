"""
DataNormalizer - 論文定義的 18 個正規化公式實作

根據論文 "Pro Trader RL" 的公式 (1)-(18)：

公式類別 1: 價格比率正規化 (1-8) - 修正版
- 除以當日 High 來正規化價格相關變數

公式類別 2: 時間變化比率 (9-10)
- ATR 今日 vs 昨日的變化率

公式類別 3: 滾動式 MinMax (11-15)
- 使用過去 12 個月的 min/max 來正規化

公式類別 4: 百分比縮放 (16-18)
- 將 0-100 的指標縮放到 0-1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger


class DataNormalizer:
    """
    實作論文定義的 18 個正規化公式
    
    使用方式:
        normalizer = DataNormalizer()
        df_normalized = normalizer.normalize(df_features)
    
    注意:
        - 論文公式 (3)-(8) 有筆誤，本實作採用修正版
        - 論文公式 (19) 報酬率採用標準金融公式 (Sell-Buy)/Buy
    """
    
    # 定義各類別的特徵欄位
    # 公式 1-8: 價格比率正規化
    PRICE_RATIO_COLS = [
        'Donchian_Upper', 'Donchian_Lower',  # 公式 1-2
        'Close', 'Low', 'High',               # 公式 3-5 (修正版)
        'HA_Close', 'HA_Low', 'HA_High'        # 公式 6-8 (修正版)
    ]
    
    # 公式 9-10: 時間變化比率
    TEMPORAL_RATIO_COLS = ['Index_ATR', 'ATR']
    
    # 公式 11-15: MinMax 正規化 (使用 12 個月滾動窗口)
    MINMAX_COLS = [
        'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Index_5', 'Index_6',
        'Index_7', 'Index_8', 'Index_9', 'Index_10', 'Index_11', 'Index_12',
        'Stock_1', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5', 'Stock_6',
        'Stock_7', 'Stock_8', 'Stock_9', 'Stock_10', 'Stock_11', 'Stock_12',
        'AVG_Stock', 'AVG_Index',
        'RS_1', 'RS_2', 'RS_3', 'RS_4', 'RS_5', 'RS_6',
        'RS_7', 'RS_8', 'RS_9', 'RS_10', 'RS_11', 'RS_12',
        'RS_AVG', 'RS_AVG_Short', 'RS_AVG_Long'
    ]
    
    # 公式 16-18: 百分比縮放
    PERCENTAGE_COLS = ['RS_Rate', 'MFI', 'RSI']
    
    # 不需正規化的欄位 (已經是標準化數值)
    NO_NORMALIZE_COLS = [
        'SuperTrend_14', 'SuperTrend_21',  # 已經是 1/-1
        'Return', 'Index_Return',           # 報酬率已經是小數
        'Up_Stock', 'Down_Stock',           # 市場廣度 (已是 0~1 比例)
        'RS_Rate_5', 'RS_Rate_10', 'RS_Rate_20', 'RS_Rate_40',  # 已經是 0-100
        'RS_Momentum', 'RS_Trend'           # 已經是比率
    ]
    
    def __init__(self, config: dict = None):
        """初始化正規化器"""
        config = config or {}
        self.rolling_window = config.get('rolling_window', 252)  # 約 12 個月
        logger.info("DataNormalizer 初始化完成")
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        執行所有正規化
        
        Args:
            df: 包含所有特徵的 DataFrame
            
        Returns:
            正規化後的 DataFrame
        """
        logger.info("開始執行正規化...")
        
        result = df.copy()
        
        # 1. 價格比率正規化 (公式 1-8)
        result = self.normalize_price_ratios(result)
        
        # 2. 時間變化比率 (公式 9-10)
        result = self.normalize_temporal_ratios(result)
        
        # 3. MinMax 正規化 (公式 11-15)
        result = self.normalize_minmax(result)
        
        # 4. 百分比縮放 (公式 16-18)
        result = self.normalize_percentages(result)
        
        # 5. 正規化 Open 和 HA_Open (額外)
        result = self.normalize_open_prices(result)
        
        # 6. Volume 正規化 - V3 移除 (對齊論文)
        # 論文明確排除 Volume 特徵
        # result = self.normalize_volume(result)
        
        logger.info("正規化完成 (Volume 已排除)")
        return result
    
    def normalize_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        價格比率正規化 (公式 1-8) - 修正版
        
        論文原始公式有筆誤，修正版邏輯:
        - 公式 1: Donchian_Upper_new = Donchian_Upper / High
        - 公式 2: Donchian_Lower_new = Donchian_Lower / Low
        - 公式 3-5: Close/Low/High 除以 High
        - 公式 6-8: HA_Close/HA_Low/HA_High 除以 HA_High
        """
        result = df.copy()
        
        # 公式 1: Donchian_Upper / High
        if 'Donchian_Upper' in result.columns and 'High' in result.columns:
            result['Donchian_Upper_norm'] = result['Donchian_Upper'] / result['High']
        
        # 公式 2: Donchian_Lower / Low
        if 'Donchian_Lower' in result.columns and 'Low' in result.columns:
            result['Donchian_Lower_norm'] = result['Donchian_Lower'] / result['Low']
        
        # 公式 3-5: 價格除以 High (修正版)
        if 'High' in result.columns:
            if 'Close' in result.columns:
                result['Close_norm'] = result['Close'] / result['High']
            if 'Low' in result.columns:
                result['Low_norm'] = result['Low'] / result['High']
            # High / High = 1，保留原始值
            result['High_norm'] = 1.0
        
        # 公式 6-8: HA 價格除以 HA_High (修正版)
        if 'HA_High' in result.columns:
            if 'HA_Close' in result.columns:
                result['HA_Close_norm'] = result['HA_Close'] / result['HA_High']
            if 'HA_Low' in result.columns:
                result['HA_Low_norm'] = result['HA_Low'] / result['HA_High']
            result['HA_High_norm'] = 1.0
        
        logger.debug("價格比率正規化完成 (公式 1-8)")
        return result
    
    def normalize_temporal_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時間變化比率正規化 (公式 9-10)
        
        - 公式 9: Index_ATR_new = Index_ATR(t) / Index_ATR(t-1)
        - 公式 10: ATR_new = ATR(t) / ATR(t-1)
        """
        result = df.copy()
        
        # 公式 9: Index_ATR 今日/昨日
        if 'Index_ATR' in result.columns:
            result['Index_ATR_norm'] = result['Index_ATR'] / result['Index_ATR'].shift(1)
        
        # 公式 10: ATR 今日/昨日
        if 'ATR' in result.columns:
            result['ATR_norm'] = result['ATR'] / result['ATR'].shift(1)
        
        logger.debug("時間變化比率正規化完成 (公式 9-10)")
        return result
    
    def normalize_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        滾動式 MinMax 正規化 (公式 11-15)
        
        使用過去 12 個月 (~252 天) 的 min/max 進行正規化:
        
        - 公式 11: Index_new = (Index - Index_min) / (Index_max - Index_min)
        - 公式 12: Stock_new = (Stock - Stock_min) / (Stock_max - Stock_min)
        - 公式 13: AVG_Stock_new = (AVG_Stock - Stock_min) / (Stock_max - Stock_min)
        - 公式 14: RS_new = (RS - RS_min) / (RS_max - RS_min)
        - 公式 15: RS_AVG_new = (RS_AVG - RS_min) / (RS_max - RS_min)
        """
        result = df.copy()
        
        # 對每個 MinMax 類別的欄位進行正規化
        for col in self.MINMAX_COLS:
            if col in result.columns:
                rolling_min = result[col].rolling(window=self.rolling_window, min_periods=1).min()
                rolling_max = result[col].rolling(window=self.rolling_window, min_periods=1).max()
                
                # 避免除以零
                range_val = rolling_max - rolling_min
                range_val = range_val.replace(0, 1)  # 如果 range 為 0，設為 1 避免除零
                
                result[f'{col}_norm'] = (result[col] - rolling_min) / range_val
        
        logger.debug("MinMax 正規化完成 (公式 11-15)")
        return result
    
    def normalize_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        百分比縮放正規化 (公式 16-18)
        
        將 0-100 的百分比指標縮放到 0-1:
        - 公式 16: RS_Rate_new = RS_Rate * 0.01
        - 公式 17: MFI_new = MFI * 0.01
        - 公式 18: RSI_new = RSI * 0.01
        """
        result = df.copy()
        
        for col in self.PERCENTAGE_COLS:
            if col in result.columns:
                result[f'{col}_norm'] = result[col] * 0.01
        
        logger.debug("百分比縮放正規化完成 (公式 16-18)")
        return result
    
    def normalize_open_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        正規化 Open 和 HA_Open (額外處理)
        
        論文提到 Open 和 HA_Open 用於標準化參考，
        這裡我們將它們除以 High 進行正規化
        """
        result = df.copy()
        
        if 'Open' in result.columns and 'High' in result.columns:
            result['Open_norm'] = result['Open'] / result['High']
        
        if 'HA_Open' in result.columns and 'HA_High' in result.columns:
            result['HA_Open_norm'] = result['HA_Open'] / result['HA_High']
        
        return result
    
    def normalize_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        正規化 Volume (論文排除，但我們用 log + 滾動 z-score)
        
        論文因為 Volume 正規化複雜而排除，
        這裡使用 log 變換 + 滾動 z-score 來處理
        """
        result = df.copy()
        
        if 'Volume' in result.columns:
            # Log 變換
            log_volume = np.log1p(result['Volume'])
            
            # 滾動 z-score
            rolling_mean = log_volume.rolling(window=self.rolling_window, min_periods=1).mean()
            rolling_std = log_volume.rolling(window=self.rolling_window, min_periods=1).std()
            
            # 避免除以零
            rolling_std = rolling_std.replace(0, 1)
            
            result['Volume_norm'] = (log_volume - rolling_mean) / rolling_std
        
        return result
    
    def get_normalized_feature_columns(self) -> List[str]:
        """
        取得正規化後應用於 RL 的 69 個特徵欄位名稱
        
        Returns:
            69 個正規化特徵的欄位名稱清單
        """
        # Buy Agent 使用的 69 個正規化特徵
        normalized_cols = [
            # 基本變數 (9 個) - 正規化版
            'Open_norm', 'High_norm', 'Low_norm', 'Close_norm', 'Volume_norm',
            'HA_Open_norm', 'HA_High_norm', 'HA_Low_norm', 'HA_Close_norm',
            
            # 技術指標 (21 個)
            'Return',  # 報酬率已是小數
            'ATR_norm',  # 公式 10
            'Stock_1_norm', 'Stock_2_norm', 'Stock_3_norm', 'Stock_4_norm',
            'Stock_5_norm', 'Stock_6_norm', 'Stock_7_norm', 'Stock_8_norm',
            'Stock_9_norm', 'Stock_10_norm', 'Stock_11_norm', 'Stock_12_norm',
            'AVG_Stock_norm',
            'SuperTrend_14', 'SuperTrend_21',  # 已是 1/-1
            'MFI_norm', 'RSI_norm',  # 公式 17-18
            'Donchian_Upper_norm', 'Donchian_Lower_norm',  # 公式 1-2
            
            # 指數變數 (14 個)
            'Index_ATR_norm',  # 公式 9
            'Index_1_norm', 'Index_2_norm', 'Index_3_norm', 'Index_4_norm',
            'Index_5_norm', 'Index_6_norm', 'Index_7_norm', 'Index_8_norm',
            'Index_9_norm', 'Index_10_norm', 'Index_11_norm', 'Index_12_norm',
            'Index_Return',  # 已是小數
            
            # 相對強度 (25 個)
            'RS_1_norm', 'RS_2_norm', 'RS_3_norm', 'RS_4_norm',
            'RS_5_norm', 'RS_6_norm', 'RS_7_norm', 'RS_8_norm',
            'RS_9_norm', 'RS_10_norm', 'RS_11_norm', 'RS_12_norm',
            'RS_AVG_norm', 'AVG_Index_norm',
            'RS_AVG_Short_norm', 'RS_AVG_Long_norm',
            'RS_Rate_norm',  # 公式 16
            'RS_Rate_5', 'RS_Rate_10', 'RS_Rate_20', 'RS_Rate_40',  # 已是百分比
            'RS_Momentum', 'RS_Trend',  # 已是比率
            'Up_Stock', 'Down_Stock'  # 將在後續計算
        ]
        
        return normalized_cols
    
    def extract_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        從正規化後的 DataFrame 提取 69 個 RL 特徵
        
        Args:
            df: 經過 normalize() 處理後的 DataFrame
            
        Returns:
            只包含 69 個正規化特徵的 DataFrame
        """
        cols = self.get_normalized_feature_columns()
        
        # 檢查哪些欄位存在
        available_cols = [c for c in cols if c in df.columns]
        missing_cols = [c for c in cols if c not in df.columns]
        
        if missing_cols:
            logger.warning(f"缺少 {len(missing_cols)} 個欄位: {missing_cols[:5]}...")
        
        return df[available_cols]


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'd:/000-github-repositories/ptrl-v01')
    
    from src.data.data_loader import DataLoader
    from src.data.feature_calculator import FeatureCalculator
    
    # 設定
    config = {'data_dir': 'data/raw/'}
    
    # 載入資料
    loader = DataLoader(config)
    df_stock = loader.load_symbol('AAPL')
    df_index = loader.load_index()
    
    if df_stock is not None and df_index is not None:
        # 計算特徵
        calc = FeatureCalculator()
        df_features = calc.calculate_all_features(df_stock, df_index)
        
        print(f"原始特徵: {len(df_features.columns)} 個")
        
        # 正規化
        normalizer = DataNormalizer()
        df_normalized = normalizer.normalize(df_features)
        
        print(f"正規化後: {len(df_normalized.columns)} 個欄位")
        
        # 提取 RL 特徵
        df_rl = normalizer.extract_normalized_features(df_normalized)
        print(f"RL 特徵: {len(df_rl.columns)} 個")
        print("\n特徵欄位:")
        print(df_rl.columns.tolist())
    else:
        print("請先執行 DataLoader 下載資料")
