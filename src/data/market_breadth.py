"""
MarketBreadth - 市場廣度指標計算模組

計算論文定義的 Up_Stock / Down_Stock:
- Up_Stock: 當日上漲股票數量 (正規化為比例)
- Down_Stock: 當日下跌股票數量 (正規化為比例)

使用方式:
    from src.data.market_breadth import MarketBreadthCalculator
    
    calculator = MarketBreadthCalculator()
    breadth_df = calculator.calculate(all_stocks_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger


class MarketBreadthCalculator:
    """
    計算市場廣度指標 (Up_Stock / Down_Stock)
    
    論文定義:
    - Up_Stock: 當日收盤價 > 前日收盤價 的股票數量
    - Down_Stock: 當日收盤價 < 前日收盤價 的股票數量
    
    正規化方式:
    - 使用比例法: Up_Stock / Total_Stocks (範圍 0~1)
    """
    
    def __init__(self, normalize: bool = True):
        """
        初始化市場廣度計算器
        
        Args:
            normalize: 是否正規化為比例 (預設 True)
        """
        self.normalize = normalize
        logger.info("MarketBreadthCalculator 初始化完成")
    
    def calculate(self, all_stocks_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        計算每日市場廣度
        
        Args:
            all_stocks_data: {symbol: DataFrame} 所有股票資料
                            每個 DataFrame 需包含 'Close' 欄位，索引為日期
        
        Returns:
            DataFrame with columns ['Up_Stock', 'Down_Stock', 'Total_Stocks']
            索引為日期
        """
        logger.info(f"開始計算市場廣度，共 {len(all_stocks_data)} 支股票")
        
        # 收集所有股票的日報酬率
        daily_returns = {}
        
        for symbol, df in all_stocks_data.items():
            if df is None or len(df) == 0:
                continue
            if 'Close' not in df.columns:
                continue
            
            # 確保索引是日期格式
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                else:
                    continue
            
            # 計算日報酬率
            returns = df['Close'].pct_change()
            daily_returns[symbol] = returns
        
        if len(daily_returns) == 0:
            logger.warning("沒有有效的股票資料來計算市場廣度")
            return pd.DataFrame()
        
        # 合併為一個大 DataFrame (每列是一天，每欄是一支股票)
        returns_df = pd.DataFrame(daily_returns)
        
        logger.info(f"報酬率矩陣大小: {returns_df.shape}")
        
        # 計算每日漲跌家數
        up_count = (returns_df > 0).sum(axis=1)
        down_count = (returns_df < 0).sum(axis=1)
        total_count = returns_df.notna().sum(axis=1)  # 當日有資料的股票數
        
        # 建立結果 DataFrame
        breadth = pd.DataFrame({
            'Up_Stock': up_count,
            'Down_Stock': down_count,
            'Total_Stocks': total_count
        })
        
        # 正規化為比例 (0~1)
        if self.normalize:
            # 避免除以零
            breadth['Up_Stock'] = np.where(
                breadth['Total_Stocks'] > 0,
                breadth['Up_Stock'] / breadth['Total_Stocks'],
                0.5  # 無資料時設為 0.5 (中性)
            )
            breadth['Down_Stock'] = np.where(
                breadth['Total_Stocks'] > 0,
                breadth['Down_Stock'] / breadth['Total_Stocks'],
                0.5
            )
            logger.info("市場廣度已正規化為比例 (0~1)")
        
        logger.info(f"市場廣度計算完成，共 {len(breadth)} 天")
        
        return breadth
    
    def calculate_and_cache(self, 
                            all_stocks_data: Dict[str, pd.DataFrame],
                            cache_path: str = 'data/processed/market_breadth.csv') -> pd.DataFrame:
        """
        計算市場廣度並快取到檔案
        
        Args:
            all_stocks_data: 所有股票資料
            cache_path: 快取檔案路徑
            
        Returns:
            市場廣度 DataFrame
        """
        import os
        
        breadth = self.calculate(all_stocks_data)
        
        if len(breadth) > 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            breadth.to_csv(cache_path)
            logger.info(f"市場廣度已快取至: {cache_path}")
        
        return breadth
    
    def load_cached(self, cache_path: str = 'data/processed/market_breadth.csv') -> Optional[pd.DataFrame]:
        """
        載入快取的市場廣度資料
        
        Args:
            cache_path: 快取檔案路徑
            
        Returns:
            市場廣度 DataFrame，若檔案不存在則返回 None
        """
        import os
        
        if os.path.exists(cache_path):
            breadth = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"載入快取市場廣度: {cache_path} ({len(breadth)} 天)")
            return breadth
        
        logger.warning(f"快取檔案不存在: {cache_path}")
        return None


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    print("MarketBreadthCalculator 模組載入成功")
    print("使用方式:")
    print("  calculator = MarketBreadthCalculator()")
    print("  breadth = calculator.calculate(all_stocks_data)")
    print("  # breadth 包含 Up_Stock, Down_Stock (0~1 比例)")
