# Pro Trader RL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-60%20passed-brightgreen.svg)](#測試)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基於論文 **"Pro Trader RL: Reinforcement learning framework for generating trading knowledge by mimicking the decision-making patterns of professional traders"** (Expert Systems with Applications, 2024) 的完整 Python 實作。

---

## 📖 目錄

- [專案概述](#專案概述)
- [系統架構](#系統架構)
- [目錄結構](#目錄結構)
- [安裝指南](#安裝指南)
- [快速開始](#快速開始)
- [核心模組詳解](#核心模組詳解)
- [腳本使用說明](#腳本使用說明)
- [設定檔說明](#設定檔說明)
- [測試](#測試)
- [論文規格對照](#論文規格對照)
- [參考文獻](#參考文獻)

---

## 專案概述

Pro Trader RL 是一個**模組化的強化學習交易框架**，透過模仿專業交易員的決策模式（買入時機、賣出時機、停損策略）來進行股票交易。

### 核心設計理念

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pro Trader RL System                         │
├─────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  Donchian   │──▶│  Buy Agent  │──▶│   過濾後    │          │
│   │  Channel    │   │  (PPO)      │   │   買入訊號   │          │
│   │  買入訊號   │   │  69維特徵   │   │   Top 10    │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                              │                  │
│                                              ▼                  │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │  Stop Loss  │◀──│ Sell Agent  │◀──│   持倉管理   │          │
│   │  Rules      │   │  (PPO)      │   │             │          │
│   │ -10%/20天   │   │  70維特徵   │   │             │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 四大核心模組

| 模組 | 功能 | 關鍵技術 |
|------|------|----------|
| **Data Preprocessing** | 資料下載、69維特徵計算、18個正規化公式 | yfinance, pandas |
| **Buy Knowledge RL** | 過濾 Donchian 突破訊號，預測報酬 ≥10% | PPO (69→40→2) |
| **Sell Knowledge RL** | 在 120 天內找最佳賣點 | PPO (70→40→2) |
| **Stop Loss Rules** | 硬規則停損，優先於 Agent 決策 | -10% 跌幅, 20 天盤整 |

---

## 系統架構

### 模組關係圖

```
src/
├── data/                    # 資料處理層
│   ├── DataLoader           # 資料下載與快取 (yfinance)
│   ├── FeatureCalculator    # 69 維特徵計算
│   └── DataNormalizer       # 18 個正規化公式
│
├── environments/            # RL 環境層
│   ├── BuyEnv               # Buy Agent 訓練環境 (69維)
│   └── SellEnv              # Sell Agent 訓練環境 (70維)
│
├── agents/                  # RL Agent 層
│   ├── BuyAgent             # Buy Agent (PPO + SB3)
│   └── SellAgent            # Sell Agent (PPO + SB3)
│
├── rules/                   # 規則層
│   ├── DonchianChannel      # 唐奇安通道買入訊號
│   └── StopLossRule         # 停損規則 (跌幅/盤整/最長持有)
│
├── trading/                 # 交易執行層
│   ├── PortfolioManager     # 投資組合管理
│   ├── TradeExecutor        # 交易執行器
│   └── StrategyOrchestrator # 策略協調器
│
├── evaluation/              # 評估層
│   ├── PerformanceEvaluator # 績效指標計算
│   └── Visualizer           # 圖表生成
│
└── backtest/                # 回測層
    └── BacktestEngine       # 回測引擎
```

### 資料流程

```
[Yahoo Finance] ──▶ [DataLoader] ──▶ [FeatureCalculator] ──▶ [DataNormalizer]
        │                                     │                     │
        │              69 維特徵              │      正規化特徵      │
        ▼                                     ▼                     ▼
   原始 OHLCV ─────▶ 基礎變數 (9)      ─────▶ 技術指標 (21)  ─────▶ RL 輸入
                     指數變數 (13)            相對強度 (26)
```

---

## 目錄結構

```
ptrl-v01/
├── config/                     # 設定檔目錄
│   ├── default_config.yaml     # 主設定檔 (PPO參數、特徵設定等)
│   └── sp500_symbols.txt       # S&P 500 股票列表
│
├── data/                       # 資料目錄
│   ├── raw/                    # 原始 OHLCV 資料 (CSV)
│   ├── processed/              # 處理後特徵資料
│   └── cache/                  # 快取檔案 (PKL)
│
├── models/                     # 模型儲存目錄
│   ├── buy_agent/              # Buy Agent 模型
│   │   └── best_model.zip      # 最佳模型
│   ├── sell_agent/             # Sell Agent 模型
│   │   └── best_model.zip      # 最佳模型
│   └── checkpoints/            # 訓練檢查點
│
├── logs/                       # 日誌目錄
│   ├── training/               # TensorBoard 訓練日誌
│   ├── backtest/               # 回測日誌
│   └── daily_ops/              # 每日營運日誌
│
├── outputs/                    # 輸出目錄
│   ├── reports/                # 報告 (TXT, JSON)
│   ├── signals/                # 交易訊號 (CSV)
│   └── plots/                  # 圖表 (PNG)
│
├── src/                        # 核心原始碼
│   ├── data/                   # 資料處理模組
│   │   ├── data_loader.py      # DataLoader 類別
│   │   ├── feature_calculator.py # FeatureCalculator 類別
│   │   └── normalizer.py       # DataNormalizer 類別
│   │
│   ├── environments/           # RL 環境模組
│   │   ├── buy_env.py          # BuyEnv 類別
│   │   └── sell_env.py         # SellEnv, SellEnvSimple 類別
│   │
│   ├── agents/                 # RL Agent 模組
│   │   ├── buy_agent.py        # BuyAgent 類別
│   │   └── sell_agent.py       # SellAgent 類別
│   │
│   ├── rules/                  # 交易規則模組
│   │   └── stop_loss.py        # StopLossRule, DonchianChannel 類別
│   │
│   ├── trading/                # 交易系統模組
│   │   ├── portfolio_manager.py    # PortfolioManager 類別
│   │   ├── trade_executor.py       # TradeExecutor 類別
│   │   └── strategy_orchestrator.py # StrategyOrchestrator 類別
│   │
│   ├── evaluation/             # 評估模組
│   │   ├── performance_evaluator.py # PerformanceEvaluator 類別
│   │   └── visualizer.py       # Visualizer 類別
│   │
│   ├── backtest/               # 回測模組
│   │   └── backtest_engine.py  # BacktestEngine 類別
│   │
│   └── utils/                  # 工具模組
│       └── __init__.py
│
├── scripts/                    # 執行腳本
│   ├── train.py                # 訓練腳本
│   ├── backtest.py             # 回測腳本
│   ├── daily_ops.py            # 每日營運腳本
│   ├── data_update.py          # 資料更新腳本
│   ├── generate_signals.py     # 訊號產生腳本
│   └── generate_report.py      # 報告產生腳本
│
├── tests/                      # 測試目錄
│   ├── test_trading.py         # 交易模組測試 (19 tests)
│   ├── test_evaluation.py      # 評估模組測試 (16 tests)
│   └── test_paper_verification.py # 論文規格驗證 (25 tests)
│
├── requirements.txt            # Python 依賴
├── .gitignore                  # Git 忽略檔案
└── README.md                   # 本文件
```

---

## 安裝指南

### 系統需求

- Python 3.10 或更高版本
- Windows / Linux / macOS
- 建議 8GB+ RAM

### 安裝步驟

```bash
# 1. 克隆專案
git clone https://github.com/ROGabdp/ptrl-v01.git
cd ptrl-v01

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 驗證安裝
python -m pytest tests/ -v
```

### 依賴套件

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.2.0
tensorflow>=2.12.0
stable-baselines3>=2.0.0
gymnasium>=0.28.0
yfinance>=0.2.18
PyYAML>=6.0
loguru>=0.7.0
pytest>=7.3.0
```

---

## 快速開始

### 1. 下載資料並訓練模型

```bash
# 下載 S&P 500 資料並計算特徵
python scripts/data_update.py --all --features

# 訓練 Buy Agent 和 Sell Agent
python scripts/train.py

# 或分開訓練
python scripts/train.py --buy-only
python scripts/train.py --sell-only
```

### 2. 執行回測

```bash
# 回測 2022-2023 年
python scripts/backtest.py --start 2022-01-01 --end 2023-12-31

# 指定股票回測
python scripts/backtest.py --symbols AAPL MSFT GOOGL
```

### 3. 每日營運

```bash
# 執行今日營運 (更新資料、掃描訊號、檢查持倉)
python scripts/daily_ops.py

# 指定日期
python scripts/daily_ops.py --date 2023-12-18

# 載入現有持倉
python scripts/daily_ops.py --positions positions.csv
```

### 4. 監控訓練

```bash
# 啟動 TensorBoard
tensorboard --logdir=logs/training/

# 瀏覽器訪問 http://localhost:6006
```

---

## 核心模組詳解

### 1. DataLoader (`src/data/data_loader.py`)

負責資料下載、快取與管理。

```python
from src.data import DataLoader

loader = DataLoader({
    'cache_dir': 'data/raw/',
    'index_symbol': '^GSPC'
})

# 下載單支股票
df = loader.download_symbol('AAPL', '2020-01-01', '2023-12-31')

# 載入已快取資料
df = loader.load_symbol('AAPL')

# 載入指數資料
index_df = loader.load_index()

# 取得 S&P 500 成分股列表
symbols = loader.get_sp500_symbols()
```

### 2. FeatureCalculator (`src/data/feature_calculator.py`)

計算論文定義的 69 維特徵。

```python
from src.data import FeatureCalculator

calc = FeatureCalculator({
    'atr_period': 14,
    'donchian_period': 20,
    'rsi_period': 14,
    'mfi_period': 14
})

# 計算所有特徵
features = calc.calculate_all_features(stock_df, index_df)

# features.columns 包含:
# - 基礎變數 (9): Open, High, Low, Close, Volume, HA_*
# - 技術指標 (21): Return, ATR, Stock_1~12, SuperTrend, MFI, RSI, Donchian
# - 指數變數 (13): Index_*, Index_Return
# - 相對強度 (26): RS_*
```

**69 維特徵明細**:

| 類別 | 數量 | 說明 |
|------|------|------|
| 基礎變數 | 9 | Open, High, Low, Close, Volume, HA_Open, HA_High, HA_Low, HA_Close |
| 技術指標 | 21 | Return, ATR, Stock(1-12), AVG_Stock, SuperTrend_14, SuperTrend_21, MFI, RSI, Donchian_Upper, Donchian_Lower |
| 指數變數 | 13 | Index 版本的基礎變數 + Index_Return |
| 相對強度 | 26 | RS_* (股票相對指數的各項指標) |

### 3. DataNormalizer (`src/data/normalizer.py`)

實作論文的 18 個正規化公式 (Eq. 1-18)。

```python
from src.data import DataNormalizer

normalizer = DataNormalizer()

# 正規化特徵
normalized = normalizer.normalize(features)

# 提取 RL 用的正規化特徵 (69 維)
rl_features = normalizer.extract_normalized_features(normalized)

# 取得正規化特徵欄位名稱
feature_cols = normalizer.get_normalized_feature_columns()
```

**正規化公式對照**:

| 公式 | 變數 | 正規化方法 |
|------|------|------------|
| Eq. 1-2 | Donchian | X / High 或 X / Low |
| Eq. 3-8 | OHLC, HA_OHLC | X / High |
| Eq. 9-10 | SuperTrend | X / Close |
| Eq. 11 | Return | tanh(X) |
| Eq. 12 | ATR | X / Close |
| Eq. 13-15 | Stock(N), AVG_Stock | tanh(X) |
| Eq. 16-18 | RS_Rate, MFI, RSI | X * 0.01 |

### 4. BuyEnv (`src/environments/buy_env.py`)

Buy Agent 的訓練環境。

```python
from src.environments import BuyEnv

# 建立環境
env = BuyEnv(training_data, config={
    'success_threshold': 0.10,  # 成功定義: ≥10% 報酬
    'balance_samples': True     # 平衡正負樣本 1:1
})

# 狀態空間: 69 維
# 動作空間: 2 (0=不買, 1=買)
# 獎勵: +1 (正確預測成功), -1 (錯誤預測)
```

### 5. SellEnv (`src/environments/sell_env.py`)

Sell Agent 的訓練環境。

```python
from src.environments import SellEnv

# 建立環境
env = SellEnv(trade_data, config={
    'max_holding_days': 120,    # 最長持有天數
    'reward_type': 'ranking'    # 排名獎勵
})

# 狀態空間: 70 維 (69 特徵 + SellReturn)
# 動作空間: 2 (0=持有, 1=賣出)
# 獎勵: 基於賣出時機的相對排名 (-1 到 +2)
```

### 6. BuyAgent / SellAgent (`src/agents/`)

基於 PPO 的 RL Agent。

```python
from src.agents import BuyAgent, SellAgent

# 初始化 Agent
buy_agent = BuyAgent({
    'learning_rate': 0.0001,
    'batch_size': 64,
    'n_steps': 2048,
    'hidden_size': 40
})

# 訓練
buy_agent.train(env, total_timesteps=500000)

# 儲存/載入
buy_agent.save('models/buy_agent/best_model.zip')
buy_agent.load_best_model()

# 預測
action = buy_agent.predict(observation)
probs = buy_agent.predict_proba(observation)  # [hold_prob, buy_prob]
```

**網路架構** (論文 Table 6):

```
Buy Agent:  Input(69) → Dense(40, ReLU) → Output(2, Softmax)
Sell Agent: Input(70) → Dense(40, ReLU) → Output(2, Softmax)
```

### 7. StopLossRule (`src/rules/stop_loss.py`)

實作論文的停損規則。

```python
from src.rules import StopLossRule

rule = StopLossRule({
    'dip_threshold': -0.10,     # 跌幅停損: -10%
    'sideways_days': 20,        # 盤整停損: 連續 20 天
    'sideways_threshold': 0.10, # 盤整閾值: <10% 報酬
    'max_holding_days': 120     # 最長持有: 120 天
})

# 檢查停損
result = rule.check(
    buy_price=100,
    current_price=88,    # -12%
    holding_days=15,
    price_history=prices
)

if result.should_stop:
    print(f"停損類型: {result.stop_type}")  # 'dip', 'sideways', 'max_holding'
```

### 8. DonchianChannel (`src/rules/stop_loss.py`)

產生買入訊號。

```python
from src.rules import DonchianChannel

dc = DonchianChannel(period=20)

# 計算通道
result = dc.calculate(stock_df)
# result['Donchian_Upper'], result['Donchian_Lower']

# 產生訊號
buy_signals = dc.generate_buy_signals(stock_df)  # 1=買入, 0=等待
sell_signals = dc.generate_sell_signals(stock_df)
```

### 9. PortfolioManager (`src/trading/portfolio_manager.py`)

管理投資組合。

```python
from src.trading import PortfolioManager

pm = PortfolioManager({
    'initial_capital': 10000,   # 初始資金 $10,000
    'max_positions': 10,        # 最大持倉 10 檔
    'max_position_pct': 0.10,   # 單檔上限 10%
    'trading_fee': 0.001        # 手續費 0.1%
})

# 開倉
trade = pm.open_position('AAPL', price=150.0, date=datetime.now())

# 平倉
trade = pm.close_position('AAPL', price=165.0, date=datetime.now(), reason='agent')

# 取得資訊
equity = pm.get_equity({'AAPL': 165.0})  # 總權益
positions = pm.get_positions()            # 所有持倉
stats = pm.get_statistics()               # 統計資料

# 儲存/載入狀態
pm.save_state('data/portfolio_state.json')
pm.load_state('data/portfolio_state.json')
```

### 10. TradeExecutor (`src/trading/trade_executor.py`)

整合 Agent 和規則執行交易。

```python
from src.trading import TradeExecutor

executor = TradeExecutor(portfolio_manager, {
    'buy_confidence_threshold': 0.5,
    'sell_prob_threshold': 0.85,     # 論文: |sell-hold| > 0.85
    'use_top_n': True,
    'top_n': 10
})

# 設定 Agent
executor.set_agents(buy_agent, sell_agent)
executor.set_stop_loss(stop_loss_rule)

# 處理買入訊號
executor.add_buy_candidate('AAPL', features, price=150, date=today)
executor.add_buy_candidate('MSFT', features, price=300, date=today)
trades = executor.execute_daily_buys()  # 執行 Top N 買入

# 處理賣出決策 (優先檢查停損)
trade = executor.process_sell_decision('AAPL', features, price=145, date=today)
```

### 11. PerformanceEvaluator (`src/evaluation/performance_evaluator.py`)

計算績效指標。

```python
from src.evaluation import PerformanceEvaluator

evaluator = PerformanceEvaluator(risk_free_rate=0.0, trading_days=252)

# 計算所有指標
metrics = evaluator.calculate_all(equity_curve, trades)

print(f"總報酬: {metrics.total_return:.2%}")
print(f"年化報酬: {metrics.annualized_return:.2%}")
print(f"夏普比率: {metrics.sharpe_ratio:.2f}")
print(f"最大回撤: {metrics.max_drawdown:.2%}")
print(f"勝率: {metrics.win_rate:.2%}")
print(f"獲利因子: {metrics.profit_factor:.2f}")

# 與基準比較
comparison = evaluator.compare_with_benchmark(strategy_curve, benchmark_curve)
print(f"Alpha: {comparison['alpha']:.2%}")
```

### 12. Visualizer (`src/evaluation/visualizer.py`)

產生圖表。

```python
from src.evaluation import Visualizer

viz = Visualizer(figsize=(12, 6))

# 權益曲線 (含回撤)
fig = viz.plot_equity_curve(equity_curve, benchmark=index_curve, show_drawdown=True)

# 月度報酬熱力圖
fig = viz.plot_monthly_returns(equity_curve)

# 交易分布
fig = viz.plot_trade_distribution(trades)

# 績效摘要
fig = viz.plot_performance_summary(metrics.to_dict())

# 完整回測報告
report_path = viz.create_backtest_report(
    equity_curve, trades, metrics.to_dict(),
    benchmark=index_curve,
    output_path='outputs/reports/'
)
```

---

## 腳本使用說明

### 1. `train.py` - 訓練腳本

訓練 Buy Agent 和 Sell Agent。

```bash
# 完整訓練 (Buy + Sell)
python scripts/train.py

# 只訓練 Buy Agent
python scripts/train.py --buy-only

# 只訓練 Sell Agent
python scripts/train.py --sell-only

# 從檢查點恢復訓練
python scripts/train.py --resume

# 指定設定檔
python scripts/train.py --config config/custom_config.yaml
```

**參數說明**:

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 設定檔路徑 | `config/default_config.yaml` |
| `--buy-only` | 只訓練 Buy Agent | False |
| `--sell-only` | 只訓練 Sell Agent | False |
| `--resume` | 從檢查點恢復訓練 | False |

**輸出**:
- `models/buy_agent/best_model.zip` - 最佳 Buy Agent
- `models/sell_agent/best_model.zip` - 最佳 Sell Agent
- `logs/training/` - TensorBoard 日誌

---

### 2. `backtest.py` - 回測腳本

使用訓練好的模型執行回測。

```bash
# 基本回測
python scripts/backtest.py --start 2022-01-01 --end 2023-12-31

# 指定股票
python scripts/backtest.py --symbols AAPL MSFT GOOGL NVDA

# 指定輸出目錄
python scripts/backtest.py --output outputs/my_backtest/
```

**參數說明**:

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 設定檔路徑 | `config/default_config.yaml` |
| `--start` | 回測起始日期 | `2022-01-01` |
| `--end` | 回測結束日期 | `2023-12-31` |
| `--symbols` | 股票列表 (空格分隔) | 全部 S&P 500 |
| `--output` | 輸出目錄 | `outputs/reports/` |

**輸出**:
- 權益曲線圖 (`equity_curve_*.png`)
- 回撤圖 (`drawdown_*.png`)
- 月度報酬圖 (`monthly_returns_*.png`)
- 績效報告 (終端輸出)

---

### 3. `daily_ops.py` - 每日營運腳本

執行完整的每日營運流程。

```bash
# 執行今日營運
python scripts/daily_ops.py

# 指定日期
python scripts/daily_ops.py --date 2023-12-18

# 不更新資料 (使用快取)
python scripts/daily_ops.py --no-update

# 載入現有持倉
python scripts/daily_ops.py --positions data/positions.csv
```

**參數說明**:

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 設定檔路徑 | `config/default_config.yaml` |
| `--date` | 執行日期 (YYYY-MM-DD) | 今天 |
| `--no-update` | 不更新資料 | False |
| `--positions` | 持倉 CSV 檔案路徑 | None |

**持倉 CSV 格式**:
```csv
symbol,buy_date,buy_price,shares
AAPL,2023-10-15,175.50,10
MSFT,2023-11-01,330.00,5
```

**流程**:
1. 更新股票資料
2. 掃描 Donchian 買入訊號
3. 使用 Buy Agent 過濾
4. 檢查持倉的停損條件
5. 使用 Sell Agent 判斷賣出
6. 產生每日報告

---

### 4. `data_update.py` - 資料更新腳本

獨立的資料更新腳本。

```bash
# 更新所有 S&P 500 股票
python scripts/data_update.py --all

# 更新指定股票
python scripts/data_update.py --symbols AAPL MSFT GOOGL

# 更新並計算特徵
python scripts/data_update.py --all --features

# 更新股票列表
python scripts/data_update.py --update-list

# 指定日期範圍
python scripts/data_update.py --start 2020-01-01 --end 2023-12-31
```

**參數說明**:

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 設定檔路徑 | `config/default_config.yaml` |
| `--symbols` | 股票列表 | 已儲存列表 |
| `--all` | 更新所有股票 | False |
| `--start` | 起始日期 | 30 天前 |
| `--end` | 結束日期 | 今天 |
| `--features` | 同時計算特徵 | False |
| `--update-list` | 更新 S&P 500 列表 | False |

**輸出**:
- `data/raw/*.csv` - 原始資料
- `data/cache/*.pkl` - 特徵快取 (若 `--features`)
- `config/sp500_symbols.txt` - 股票列表 (若 `--update-list`)

---

### 5. `generate_signals.py` - 訊號產生腳本

產生買賣訊號。

```bash
# 產生今日訊號
python scripts/generate_signals.py

# 指定日期
python scripts/generate_signals.py --date 2023-12-18

# 載入持倉檢查賣出條件
python scripts/generate_signals.py --positions data/positions.csv

# 指定輸出目錄
python scripts/generate_signals.py --output outputs/signals/
```

**參數說明**:

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 設定檔路徑 | `config/default_config.yaml` |
| `--date` | 目標日期 | 今天 |
| `--positions` | 持倉 CSV | None |
| `--output` | 輸出目錄 | `outputs/signals/` |

**輸出**:
- `buy_signals_YYYYMMDD.csv` - 買入訊號 (Top 10)
- `sell_signals_YYYYMMDD.csv` - 賣出建議

**買入訊號 CSV 格式**:
```csv
symbol,date,price,donchian_upper,confidence,recommendation
NVDA,2023-12-18,485.50,480.00,0.92,BUY
AAPL,2023-12-18,197.20,195.00,0.87,BUY
```

---

### 6. `generate_report.py` - 報告產生腳本

產生績效報告和圖表。

```bash
# 從交易紀錄產生報告
python scripts/generate_report.py --trades outputs/trades.csv

# 從投資組合狀態產生
python scripts/generate_report.py --portfolio data/portfolio_state.json

# 從權益曲線產生
python scripts/generate_report.py --equity outputs/equity_curve.csv

# 不產生圖表 (只文字)
python scripts/generate_report.py --trades trades.csv --no-visual
```

**參數說明**:

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--config` | 設定檔路徑 | `config/default_config.yaml` |
| `--trades` | 交易紀錄 CSV | None |
| `--equity` | 權益曲線 CSV | None |
| `--portfolio` | 投資組合 JSON | None |
| `--output` | 輸出目錄 | `outputs/reports/` |
| `--no-visual` | 不產生圖表 | False |

**輸出**:
- `report_YYYYMMDD_HHMMSS.txt` - 文字報告
- `metrics_YYYYMMDD_HHMMSS.csv` - 指標 CSV
- `trades_YYYYMMDD_HHMMSS.csv` - 交易明細
- `backtest_report_*.png` - 視覺化報告

---

## 設定檔說明

### `config/default_config.yaml`

```yaml
# 資料設定
data:
  cache_dir: "data/raw/"
  index_symbol: "^GSPC"

# 特徵計算設定
features:
  atr_period: 14
  donchian_period: 20
  rsi_period: 14
  mfi_period: 14
  supertrend_periods: [14, 21]
  supertrend_multiplier: 3.0

# Buy Agent 設定
buy_agent:
  learning_rate: 0.0001
  batch_size: 64
  n_steps: 2048
  hidden_size: 40
  total_timesteps: 500000
  success_threshold: 0.10
  model_dir: "models/buy_agent/"

# Sell Agent 設定
sell_agent:
  learning_rate: 0.0001
  batch_size: 64
  n_steps: 2048
  hidden_size: 40
  total_timesteps: 500000
  sell_prob_threshold: 0.85
  max_holding_days: 120
  model_dir: "models/sell_agent/"

# 停損設定
stop_loss:
  dip_threshold: -0.10
  sideways_days: 20
  sideways_threshold: 0.10
  max_holding_days: 120

# 投資組合設定
portfolio:
  initial_capital: 10000
  max_positions: 10
  max_position_pct: 0.10
  trading_fee: 0.001

# 訓練設定
training:
  checkpoint_dir: "models/checkpoints/"
  tensorboard_dir: "logs/training/"
  save_frequency: 10000
  eval_frequency: 5000

# 回測設定
backtest:
  start_date: "2022-01-01"
  end_date: "2023-12-31"

# 日誌設定
logging:
  level: "INFO"
  dir: "logs/"
```

---

## 測試

### 執行全部測試

```bash
python -m pytest tests/ -v
```

### 執行特定測試

```bash
# 交易模組測試
python -m pytest tests/test_trading.py -v

# 評估模組測試
python -m pytest tests/test_evaluation.py -v

# 論文規格驗證
python -m pytest tests/test_paper_verification.py -v
```

### 測試覆蓋

| 測試檔案 | 測試數量 | 說明 |
|----------|----------|------|
| `test_trading.py` | 19 | Position, PortfolioManager, TradeExecutor |
| `test_evaluation.py` | 16 | PerformanceMetrics, PerformanceEvaluator |
| `test_paper_verification.py` | 25 | 論文全部規格驗證 |
| **總計** | **60** | **全部通過** |

---

## 論文規格對照

### Table 1-4: 69 維特徵 ✅

| 類別 | 論文數量 | 實作數量 | 狀態 |
|------|----------|----------|------|
| 基礎變數 | 9 | 9 | ✅ |
| 技術指標 | 21 | 21 | ✅ |
| 指數變數 | 13 | 13 | ✅ |
| 相對強度 | 26 | 26 | ✅ |

### Eq. 1-18: 正規化公式 ✅

全部 18 個公式已實作於 `DataNormalizer.normalize()` 方法。

### Section 3.4: 停損規則 ✅

| 規則 | 論文值 | 實作值 | 狀態 |
|------|--------|--------|------|
| 跌幅停損 | -10% | -10% | ✅ |
| 盤整停損 | 20 天 | 20 天 | ✅ |
| 盤整閾值 | <10% | <10% | ✅ |
| 最長持有 | 120 天 | 120 天 | ✅ |

### Table 6: PPO 超參數 ✅

| 參數 | 論文值 | 實作值 | 狀態 |
|------|--------|--------|------|
| Learning Rate | 0.0001 | 0.0001 | ✅ |
| Batch Size | 64 | 64 | ✅ |
| N Steps | 2048 | 2048 | ✅ |
| Hidden Size | 40 | 40 | ✅ |

### Section 4.2: 投資組合限制 ✅

| 參數 | 論文值 | 實作值 | 狀態 |
|------|--------|--------|------|
| 初始資金 | $10,000 | $10,000 | ✅ |
| 最大持倉 | 10 | 10 | ✅ |
| 單檔上限 | 10% | 10% | ✅ |
| 手續費 | 0.1% | 0.1% | ✅ |

---

## TensorBoard 監控

訓練過程可透過 TensorBoard 即時監控。

```bash
tensorboard --logdir=logs/training/

# 瀏覽器訪問 http://localhost:6006
```

**監控指標**:
- Episode Reward (訓練獎勵)
- Policy Loss (策略損失)
- Value Loss (價值損失)
- Explained Variance (解釋變異)
- Entropy (熵值)

---

## 參考文獻

```
Jeong, D. W., & Gu, Y. H. (2024). 
Pro Trader RL: Reinforcement learning framework for generating trading knowledge 
by mimicking the decision-making patterns of professional traders. 
Expert Systems with Applications, 252, 124124.
https://doi.org/10.1016/j.eswa.2024.124124
```

---

## License

MIT License

---

## 作者

基於論文實作，供研究與學習使用。

如有問題或建議，歡迎提出 Issue 或 Pull Request。
