"""
Visualizer - 視覺化工具

產生各種回測和績效圖表:
- 權益曲線圖
- 回撤圖
- 月度報酬熱力圖
- 交易分布圖

使用方式:
    viz = Visualizer()
    fig = viz.plot_equity_curve(equity_curve)
    viz.create_backtest_report(result, output_path)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

# 設定中文字體 (如果可用)
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


class Visualizer:
    """
    視覺化工具
    
    產生各種回測和績效圖表:
    - 權益曲線圖
    - 回撤圖
    - 月度報酬熱力圖
    - 交易分布圖
    
    使用方式:
        viz = Visualizer()
        fig = viz.plot_equity_curve(equity_curve)
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', figsize: Tuple[int, int] = (12, 6)):
        """
        初始化視覺化工具
        
        Args:
            style: matplotlib 樣式
            figsize: 預設圖表大小
        """
        self.figsize = figsize
        
        # 嘗試設定樣式
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass
        
        # 顏色方案
        self.colors = {
            'primary': '#2E86AB',      # 藍色
            'secondary': '#A23B72',    # 紫紅色
            'positive': '#28A745',     # 綠色
            'negative': '#DC3545',     # 紅色
            'neutral': '#6C757D',      # 灰色
            'benchmark': '#FFC107'     # 黃色
        }
    
    def plot_equity_curve(self, equity_curve: pd.Series, 
                          benchmark: pd.Series = None,
                          title: str = 'Equity Curve',
                          show_drawdown: bool = True) -> plt.Figure:
        """
        繪製權益曲線圖
        
        Args:
            equity_curve: 策略權益曲線
            benchmark: 基準曲線 (可選)
            title: 圖表標題
            show_drawdown: 是否顯示回撤
            
        Returns:
            matplotlib Figure
        """
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5),
                                           gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
        
        # 繪製權益曲線
        ax1.plot(equity_curve.index, equity_curve.values, 
                 color=self.colors['primary'], linewidth=1.5, label='Strategy')
        
        if benchmark is not None:
            # 正規化基準到相同起點
            normalized_benchmark = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
            ax1.plot(normalized_benchmark.index, normalized_benchmark.values,
                    color=self.colors['benchmark'], linewidth=1, alpha=0.7, 
                    linestyle='--', label='Benchmark')
            ax1.legend(loc='upper left')
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 繪製回撤
        if show_drawdown:
            cummax = equity_curve.cummax()
            drawdown = (cummax - equity_curve) / cummax * 100
            
            ax2.fill_between(drawdown.index, 0, drawdown.values, 
                            color=self.colors['negative'], alpha=0.4)
            ax2.plot(drawdown.index, drawdown.values, 
                    color=self.colors['negative'], linewidth=0.5)
            
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_returns(self, equity_curve: pd.Series,
                             title: str = 'Monthly Returns') -> plt.Figure:
        """
        繪製月度報酬熱力圖
        
        Args:
            equity_curve: 權益曲線
            title: 圖表標題
            
        Returns:
            matplotlib Figure
        """
        # 計算月度報酬
        monthly_returns = equity_curve.resample('ME').last().pct_change().dropna() * 100
        
        # 重組為年/月矩陣
        years = monthly_returns.index.year.unique()
        months = range(1, 13)
        
        data = pd.DataFrame(index=years, columns=months)
        for date, ret in monthly_returns.items():
            data.loc[date.year, date.month] = ret
        
        data = data.astype(float)
        
        # 繪製熱力圖
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 建立色彩映射
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(vmin=-10, vmax=10)
        
        im = ax.imshow(data.values, cmap=cmap, norm=norm, aspect='auto')
        
        # 設定軸標籤
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        
        # 加入數值標籤
        for i in range(len(years)):
            for j in range(12):
                value = data.iloc[i, j]
                if pd.notna(value):
                    color = 'white' if abs(value) > 5 else 'black'
                    ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                           color=color, fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Return (%)')
        
        plt.tight_layout()
        return fig
    
    def plot_trade_distribution(self, trades: List,
                                title: str = 'Trade Distribution') -> plt.Figure:
        """
        繪製交易分布圖
        
        Args:
            trades: 交易列表
            title: 圖表標題
            
        Returns:
            matplotlib Figure
        """
        returns = [t.return_pct * 100 for t in trades 
                   if hasattr(t, 'return_pct') and t.return_pct is not None]
        
        if not returns:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center')
            return fig
        
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # 報酬分布直方圖
        ax1 = axes[0]
        colors = [self.colors['positive'] if r > 0 else self.colors['negative'] 
                  for r in returns]
        
        n, bins, patches = ax1.hist(returns, bins=30, edgecolor='white', alpha=0.7)
        
        # 為每個柱子著色
        for i, patch in enumerate(patches):
            if bins[i] >= 0:
                patch.set_facecolor(self.colors['positive'])
            else:
                patch.set_facecolor(self.colors['negative'])
        
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.axvline(x=np.mean(returns), color=self.colors['primary'], 
                   linestyle='-', linewidth=2, label=f'Mean: {np.mean(returns):.1f}%')
        ax1.set_xlabel('Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Return Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 累積報酬圖
        ax2 = axes[1]
        cumulative = np.cumsum(returns)
        colors = [self.colors['positive'] if c >= 0 else self.colors['negative'] 
                  for c in cumulative]
        ax2.bar(range(len(cumulative)), cumulative, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Trade #')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.set_title('Cumulative Trade Returns')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_performance_summary(self, metrics: dict,
                                 title: str = 'Performance Summary') -> plt.Figure:
        """
        繪製績效摘要圖
        
        Args:
            metrics: 績效指標字典
            title: 圖表標題
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        # 格式化表格內容
        table_data = [
            ['Total Return', f"{metrics.get('total_return', 0):.2%}"],
            ['Annualized Return', f"{metrics.get('annualized_return', 0):.2%}"],
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"],
            ['Volatility', f"{metrics.get('volatility', 0):.2%}"],
            ['', ''],
            ['Total Trades', str(metrics.get('total_trades', 0))],
            ['Win Rate', f"{metrics.get('win_rate', 0):.2%}"],
            ['Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"],
            ['Avg Holding Days', f"{metrics.get('avg_holding_days', 0):.1f}"]
        ]
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='left',
            colWidths=[0.5, 0.3]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # 設定標題顏色
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(self.colors['primary'])
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#F8F9FA')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def create_backtest_report(self, equity_curve: pd.Series,
                               trades: List,
                               metrics: dict,
                               benchmark: pd.Series = None,
                               output_path: str = 'outputs/reports/') -> str:
        """
        建立完整的回測報告
        
        Args:
            equity_curve: 權益曲線
            trades: 交易列表
            metrics: 績效指標
            benchmark: 基準曲線 (可選)
            output_path: 輸出路徑
            
        Returns:
            報告儲存路徑
        """
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 建立多頁報告
        fig = plt.figure(figsize=(16, 20))
        
        # 1. 權益曲線
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(equity_curve.index, equity_curve.values, 
                color=self.colors['primary'], linewidth=1.5)
        if benchmark is not None:
            normalized = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
            ax1.plot(normalized.index, normalized.values,
                    color=self.colors['benchmark'], linestyle='--', alpha=0.7)
        ax1.set_title('Equity Curve', fontweight='bold')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 回撤
        ax2 = fig.add_subplot(3, 2, 2)
        cummax = equity_curve.cummax()
        drawdown = (cummax - equity_curve) / cummax * 100
        ax2.fill_between(drawdown.index, 0, drawdown.values, 
                        color=self.colors['negative'], alpha=0.4)
        ax2.set_title('Drawdown', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        # 3. 月度報酬
        ax3 = fig.add_subplot(3, 2, 3)
        monthly = equity_curve.resample('ME').last().pct_change().dropna() * 100
        colors = [self.colors['positive'] if r > 0 else self.colors['negative'] for r in monthly]
        ax3.bar(monthly.index, monthly.values, color=colors, width=20)
        ax3.axhline(y=0, color='black', linestyle='--')
        ax3.set_title('Monthly Returns', fontweight='bold')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 交易分布
        if trades:
            ax4 = fig.add_subplot(3, 2, 4)
            returns = [t.return_pct * 100 for t in trades 
                      if hasattr(t, 'return_pct') and t.return_pct is not None]
            if returns:
                ax4.hist(returns, bins=20, color=self.colors['primary'], 
                        edgecolor='white', alpha=0.7)
                ax4.axvline(x=0, color='black', linestyle='--')
                ax4.axvline(x=np.mean(returns), color=self.colors['secondary'],
                           linestyle='-', linewidth=2)
            ax4.set_title('Trade Return Distribution', fontweight='bold')
            ax4.set_xlabel('Return (%)')
            ax4.grid(True, alpha=0.3)
        
        # 5-6. 績效摘要
        ax5 = fig.add_subplot(3, 2, (5, 6))
        ax5.axis('off')
        
        summary_text = f"""
        Performance Summary
        ═══════════════════════════════════════════════
        
        Total Return:       {metrics.get('total_return', 0):.2%}
        Annualized Return:  {metrics.get('annualized_return', 0):.2%}
        Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):.2f}
        Max Drawdown:       {metrics.get('max_drawdown', 0):.2%}
        
        Total Trades:       {metrics.get('total_trades', 0)}
        Win Rate:           {metrics.get('win_rate', 0):.2%}
        Avg Holding Days:   {metrics.get('avg_holding_days', 0):.1f}
        """
        
        ax5.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
                verticalalignment='center', transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Pro Trader RL Backtest Report', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 儲存
        filepath = os.path.join(output_path, f'backtest_report_{timestamp}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"回測報告已儲存至: {filepath}")
        return filepath
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                    output_path: str = 'outputs/plots/'):
        """儲存圖表"""
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"圖表已儲存至: {filepath}")
        return filepath


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    print("=== Visualizer 測試 ===")
    
    # 建立模擬資料
    dates = pd.date_range('2020-01-01', periods=504, freq='B')
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, len(dates))
    equity = 10000 * (1 + returns).cumprod()
    equity_curve = pd.Series(equity, index=dates)
    
    # 建立視覺化工具
    viz = Visualizer()
    
    # 繪製權益曲線
    fig1 = viz.plot_equity_curve(equity_curve, title='Test Strategy')
    viz.save_figure(fig1, 'test_equity_curve.png')
    
    # 繪製月度報酬
    fig2 = viz.plot_monthly_returns(equity_curve)
    viz.save_figure(fig2, 'test_monthly_returns.png')
    
    print("圖表已產生!")
