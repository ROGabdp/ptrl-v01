# Pro Trader RL

基於論文 ***"Pro Trader RL: Reinforcement learning framework for generating trading knowledge by mimicking the decision-making patterns of professional traders"*** 的 Python 實作。

## 專案概述

Pro Trader RL 是一個模組化的強化學習交易框架，模仿專業交易員的決策模式。

### 四大核心模組

1. **Data Preprocessing** - 資料預處理與特徵工程
2. **Buy Knowledge RL** - 買入決策 Agent (PPO)
3. **Sell Knowledge RL** - 賣出決策 Agent (PPO)
4. **Stop Loss Rules** - 停損規則 (跌幅/盤整)

## 專案結構

```
ptrl-v01/
├── config/                 # 設定檔
│   └── default_config.yaml
├── data/                   # 資料目錄
│   ├── raw/                # 原始 OHLCV
│   └── processed/          # 特徵資料
├── models/                 # 訓練模型
│   ├── buy_agent/
│   ├── sell_agent/
│   └── checkpoints/
├── logs/                   # 日誌
│   └── training/           # TensorBoard
├── outputs/                # 輸出結果
├── src/                    # 核心程式碼
│   ├── data/               # 資料模組
│   ├── environments/       # RL 環境
│   ├── agents/             # RL Agent
│   ├── rules/              # 交易規則
│   ├── trading/            # 交易系統
│   ├── backtest/           # 回測引擎
│   └── utils/              # 工具函數
├── scripts/                # 執行腳本
├── tests/                  # 測試
└── requirements.txt
```

## 安裝

```bash
# 安裝依賴
pip install -r requirements.txt
```

## 使用方式

```bash
# 訓練 Buy Agent
python scripts/train_buy_agent.py

# 訓練 Sell Agent
python scripts/train_sell_agent.py

# 執行回測
python scripts/run_backtest.py

# 每日營運
python scripts/daily_ops.py
```

## TensorBoard 監控

```bash
tensorboard --logdir=logs/training/
# 瀏覽器訪問 http://localhost:6006
```

## 論文參數設定

| 參數 | 值 |
|------|-----|
| PPO Learning Rate | 0.0001 |
| Batch Size | 64 |
| N Steps | 2048 |
| 成功交易閾值 | ≥10% |
| 賣出信心閾值 | 0.85 |
| 跌幅停損 | -10% |
| 盤整停損 | 20天 |

## 參考文獻

```
Jeong, D. W., & Gu, Y. H. (2024). Pro Trader RL: Reinforcement learning framework 
for generating trading knowledge by mimicking the decision-making patterns of 
professional traders. Expert Systems with Applications.
```
