# 規則模組
"""
交易規則定義

Classes:
- DonchianChannel: 唐奇安通道策略 (買入訊號生成)
- StopLossRule: 停損規則 (跌幅停損 + 盤整停損)
"""

from .stop_loss import StopLossRule, StopLossResult, DonchianChannel

__all__ = ['StopLossRule', 'StopLossResult', 'DonchianChannel']
