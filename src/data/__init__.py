# 資料模組
"""
資料下載、特徵計算、正規化

Classes:
- DataLoader: 下載並管理 S&P 500 股票資料
- FeatureCalculator: 計算論文定義的 69 個技術特徵
- DataNormalizer: 實作論文的 18 個正規化公式
- MarketBreadthCalculator: 計算 Up_Stock / Down_Stock 市場廣度指標
"""

from .data_loader import DataLoader
from .feature_calculator import FeatureCalculator
from .normalizer import DataNormalizer
from .market_breadth import MarketBreadthCalculator

__all__ = ['DataLoader', 'FeatureCalculator', 'DataNormalizer', 'MarketBreadthCalculator']

