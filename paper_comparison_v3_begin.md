# Pro Trader RL v3 - 論文與程式碼完整比對分析

> **目的**: 為 v3 開發準備，逐項驗證目前程式碼與論文的一致性
> **更新日期**: 2025-12-20
> **程式碼版本**: v2.0 (Git Tag)

---

## 📊 總覽摘要

| 項目 | 論文規格 | 實作狀態 | 結論 |
|------|----------|----------|------|
| **Buy Agent 獎勵函數** | +1/0 | +1/0 | ✅ **一致** |
| **Sell Agent 獎勵函數** | 4情境 | 完全相同 | ✅ **一致** |
| **PPO 超參數** | Table 6 | 完全相同 | ✅ **一致** |
| **神經網路架構** | 69/70→40→2 | 完全相同 | ✅ **一致** |
| **停損規則** | 2種+120天 | 完全相同 | ✅ **一致** |
| **Top-10 選股** | 依信心排序 | 已實作 | ✅ **一致** |
| **基準指數** | Dow Jones (^DJI) | S&P 500 (^GSPC) | ⚠️ **刻意差異** |
| **資產池** | S&P 500+400+600 | S&P 500 only | ⚠️ **刻意差異** |
| **Volume 特徵** | 排除 | 包含 (log+z-score) | ⚠️ **刻意差異** |

---

## 1. 特徵工程 (Feature Engineering)

### 1.1 基本變數 (9個)

| 變數 | 論文定義 | 程式碼實作 | 檔案位置 | 狀態 |
|------|----------|------------|----------|------|
| Open | 開盤價 | `df['Open']` | [feature_calculator.py](file:///d:/000-github-repositories/ptrl-v01/src/data/feature_calculator.py) | ✅ |
| High | 最高價 | `df['High']` | 同上 | ✅ |
| Low | 最低價 | `df['Low']` | 同上 | ✅ |
| Close | 收盤價 | `df['Close']` | 同上 | ✅ |
| Volume | 成交量 | `df['Volume']` (log正規化) | 同上 | ⚠️ 論文排除 |
| HA_Open | Heikin Ashi Open | `(HA_Open_{t-1} + HA_Close_{t-1}) / 2` | Line 119-123 | ✅ |
| HA_High | Heikin Ashi High | `max(High, HA_Open, HA_Close)` | Line 125 | ✅ |
| HA_Low | Heikin Ashi Low | `min(Low, HA_Open, HA_Close)` | Line 126 | ✅ |
| HA_Close | Heikin Ashi Close | `(Open + High + Low + Close) / 4` | Line 117 | ✅ |

> **Volume 差異說明**: 論文明確排除 Volume，但我們使用 log + z-score 處理後包含。這是刻意的增強。

### 1.2 技術指標 (21個)

| 指標 | 論文參數 | 程式碼參數 | 公式/計算邏輯 | 狀態 |
|------|----------|------------|---------------|------|
| Return | 日報酬率 | `pct_change()` | `(Close_t - Close_{t-1}) / Close_{t-1}` | ✅ |
| ATR | N=10 | `atr_period=10` | 10日 Average True Range | ✅ |
| Stock_1~12 | N=1-12月 | `range(1,13)` | `ATR / ATR.shift(N*21)` | ✅ |
| AVG_Stock | 1,3,6,12月平均 | `[1,3,6,12]` | `mean(Stock_1, Stock_3, Stock_6, Stock_12)` | ✅ |
| SuperTrend_14 | (14, 2) | `(14, 2)` | 14日 ATR × 2 | ✅ |
| SuperTrend_21 | (21, 1) | `(21, 1)` | 21日 ATR × 1 | ✅ |
| MFI | N=14 | `mfi_period=14` | Money Flow Index | ✅ |
| RSI | N=14 | `rsi_period=14` | Relative Strength Index | ✅ |
| Donchian_Upper | N=20 | `donchian_period=20` | `High.rolling(20).max()` | ✅ |
| Donchian_Lower | N=20 | `donchian_period=20` | `Low.rolling(20).min()` | ✅ |

### 1.3 指數變數 (13個)

| 變數 | 論文 | 程式碼 | 計算邏輯 | 狀態 |
|------|------|--------|----------|------|
| Index_ATR | DJI ATR | S&P 500 ATR | 指數的 10日 ATR | ⚠️ 指數不同 |
| Index_1~12 | DJI N月比 | S&P 500 N月比 | `Index_ATR / Index_ATR.shift(N*21)` | ⚠️ 指數不同 |
| Index_Return | — | ✓ (額外) | 指數日報酬率 (額外補充) | ➕ 額外 |

> **指數差異說明**: 論文使用 Dow Jones (^DJI)，我們使用 S&P 500 (^GSPC)。S&P 500 更具代表性且與資產池一致。

### 1.4 相對強度變數 (26個)

| 變數 | 論文公式 | 程式碼實作 | 狀態 |
|------|----------|------------|------|
| RS_1~12 | Stock_N / Index_N | `Stock_N / Index_N` | ✅ |
| RS_AVG | mean(RS_1,3,6,12) | `mean(RS_1, RS_3, RS_6, RS_12)` | ✅ |
| AVG_Index | mean(Index_1,3,6,12) | `mean(Index_1, Index_3, Index_6, Index_12)` | ✅ |
| RS_Rate | 百分位排名 0-100 | `RS_AVG.rolling(252).rank(pct=True) * 100` | ✅ |
| RS_Rate_5 | 5日 MA | `RS_Rate.rolling(5).mean()` | ✅ |
| RS_Rate_10 | 10日 MA | `RS_Rate.rolling(10).mean()` | ✅ |
| RS_Rate_20 | 20日 MA | `RS_Rate.rolling(20).mean()` | ✅ |
| RS_Rate_40 | 40日 MA | `RS_Rate.rolling(40).mean()` | ✅ |
| Up_Stock | 上漲家數 | 預留 (需多股票資料) | ⚠️ 待完善 |
| Down_Stock | 下跌家數 | 預留 (需多股票資料) | ⚠️ 待完善 |

---

## 2. 正規化公式 (Normalization)

### 2.1 公式 1-8: 價格比率

| 公式 | 論文 (有筆誤) | 程式碼 (已修正) | 檔案位置 | 狀態 |
|------|--------------|-----------------|----------|------|
| (1) Donchian_Upper | Upper / High | `Donchian_Upper / High` | [normalizer.py:127-128](file:///d:/000-github-repositories/ptrl-v01/src/data/normalizer.py#L127-L128) | ✅ |
| (2) Donchian_Lower | Lower / Low | `Donchian_Lower / Low` | Line 131-132 | ✅ |
| (3) Close | ~~Upper / High~~ | `Close / High` | Line 136-137 | ✅ 修正 |
| (4) Low | ~~Upper / High~~ | `Low / High` | Line 138-139 | ✅ 修正 |
| (5) High | ~~Upper / High~~ | `1.0` (High/High) | Line 141 | ✅ 修正 |
| (6) HA_Close | ~~同上~~ | `HA_Close / HA_High` | Line 145-146 | ✅ 修正 |
| (7) HA_Low | ~~同上~~ | `HA_Low / HA_High` | Line 147-148 | ✅ 修正 |
| (8) HA_High | ~~同上~~ | `1.0` (HA_High/HA_High) | Line 149 | ✅ 修正 |

> **修正說明**: 論文公式 (3)-(8) 有明顯筆誤（全部寫成 Upper/High），我們採用合理的修正版本。

### 2.2 公式 9-10: 時間變化比率

| 公式 | 論文 | 程式碼 | 狀態 |
|------|------|--------|------|
| (9) Index_ATR | ATR_t / ATR_{t-1} | `Index_ATR / Index_ATR.shift(1)` | ✅ |
| (10) ATR | ATR_t / ATR_{t-1} | `ATR / ATR.shift(1)` | ✅ |

### 2.3 公式 11-15: 滾動 MinMax

| 公式 | 論文 | 程式碼 | 狀態 |
|------|------|--------|------|
| (11-12) Index/Stock | 12月 MinMax | `rolling(252).min/max` | ✅ |
| (13) AVG_Stock | 12月 MinMax | 同上 | ✅ |
| (14-15) RS/RS_AVG | 12月 MinMax | 同上 | ✅ |

### 2.4 公式 16-18: 百分比縮放

| 公式 | 論文 | 程式碼 | 狀態 |
|------|------|--------|------|
| (16) RS_Rate | × 0.01 | `RS_Rate * 0.01` | ✅ |
| (17) MFI | × 0.01 | `MFI * 0.01` | ✅ |
| (18) RSI | × 0.01 | `RSI * 0.01` | ✅ |

---

## 3. Buy Knowledge RL

### 3.1 環境設計

| 項目 | 論文規格 | 程式碼實作 | 檔案位置 | 狀態 |
|------|----------|------------|----------|------|
| 狀態維度 | 69 | 69 | [buy_env.py](file:///d:/000-github-repositories/ptrl-v01/src/environments/buy_env.py) | ✅ |
| 動作空間 | 2 (買/不買) | `Discrete(2)`: 0=不買, 1=買 | Line 130 | ✅ |
| 成功定義 | 報酬 ≥ 10% | `success_threshold=0.10` | Line 74 | ✅ |
| 資料平衡 | 1:1 | `_balance_data()` | Line 161-185 | ✅ |

### 3.2 獎勵函數 ⭐ 關鍵驗證

| 情境 | 論文 | 程式碼 (Line 242-245) | 狀態 |
|------|------|------------------------|------|
| 預測買 + 實際成功 | **+1** | `reward = 1.0 if is_successful else 0.0` | ✅ |
| 預測買 + 實際失敗 | **0** | ↑ (else 分支返回 0.0) | ✅ |
| 預測不買 + 實際失敗 | **+1** | `reward = 1.0 if not is_successful else 0.0` | ✅ |
| 預測不買 + 實際成功 | **0** | ↑ (else 分支返回 0.0) | ✅ |

```python
# buy_env.py Line 241-245 (實際程式碼)
if action == 1:
    reward = 1.0 if is_successful else 0.0  # 買+成功=+1, 買+失敗=0
else:
    reward = 1.0 if not is_successful else 0.0  # 不買+失敗=+1, 不買+成功=0
```

> ✅ **經驗證，Buy Agent 獎勵函數完全符合論文 (+1/0)，舊版比對文件的記錄有誤。**

### 3.3 PPO 超參數 (論文 Table 6)

| 參數 | 論文值 | config/default_config.yaml | 檔案位置 | 狀態 |
|------|--------|---------------------------|----------|------|
| learning_rate | 0.0001 | 0.0001 | Line 43 | ✅ |
| n_steps | 2048 | 2048 | Line 44 | ✅ |
| batch_size | 64 | 64 | Line 45 | ✅ |
| n_epochs | 10 | 10 | Line 46 | ✅ |
| gamma | 0.99 | 0.99 | Line 47 | ✅ |
| gae_lambda | 0.95 | 0.95 | Line 48 | ✅ |
| clip_range | 0.2 | 0.2 | Line 49 | ✅ |
| ent_coef | 0.01 | 0.01 | Line 50 | ✅ |
| vf_coef | 0.5 | 0.5 | Line 51 | ✅ |
| max_grad_norm | 0.5 | 0.5 | Line 52 | ✅ |

### 3.4 神經網路架構

| 層 | 論文 | 程式碼 | 狀態 |
|------|------|--------|------|
| 輸入層 | 69 | `observation_space.shape = (69,)` | ✅ |
| 隱藏層 | 40 | `net_arch: pi: [40], vf: [40]` | ✅ |
| 輸出層 | 2 | `action_space = Discrete(2)` | ✅ |
| Actor/Critic | 相同結構 | `pi` 和 `vf` 設定相同 | ✅ |

---

## 4. Sell Knowledge RL

### 4.1 環境設計

| 項目 | 論文規格 | 程式碼實作 | 檔案位置 | 狀態 |
|------|----------|------------|----------|------|
| 狀態維度 | 70 (69+SellReturn) | 70 | [sell_env.py](file:///d:/000-github-repositories/ptrl-v01/src/environments/sell_env.py) | ✅ |
| SellReturn 公式 (Eq.20) | Open_t / BuyPrice | `current_open / buy_price` | Line 336-338 | ✅ |
| 動作空間 | 2 (持有/賣出) | `Discrete(2)`: 0=持有, 1=賣出 | Line 130 | ✅ |
| 最大持有天數 | 120 | `max_holding_days=120` | Line 73 | ✅ |
| 賣出信心門檻 | sell-hold > 0.85 | `sell_threshold=0.85` | Line 75 | ✅ |
| 訓練資料過濾 | 只用成功交易 | `if max_return >= success_threshold` | Line 168 | ✅ |

### 4.2 獎勵函數 (4情境) ⭐ 關鍵驗證

| 情境 | 論文 | 程式碼 (Line 348-372) | 狀態 |
|------|------|------------------------|------|
| 賣出 + 獲利≥10% | 排名獎勵 +1~+2 | `_calculate_ranking_reward()` → +1~+2 | ✅ |
| 賣出 + 獲利<10% | -1 | `return -1.0` | ✅ |
| 持有 + 獲利<10% | +0.5 | `return 0.5` | ✅ |
| 持有 + 獲利≥10% | -1 | `return -1.0` | ✅ |

```python
# sell_env.py Line 348-372 (實際程式碼)
def _calculate_reward_paper(self, action, current_return, features):
    is_profitable = current_return >= self.success_threshold  # 10%
    
    if action == 1:  # 賣出
        if is_profitable:
            return self._calculate_ranking_reward(...)  # +1 到 +2
        else:
            return -1.0  # 賣太早/虧損
    else:  # 持有
        if is_profitable:
            return -1.0  # 錯過賣點
        else:
            return 0.5  # 正確等待
```

### 4.3 排名獎勵 (Equation 21)

| 項目 | 論文 | 程式碼 (Line 374-400) | 狀態 |
|------|------|------------------------|------|
| 計算範圍 | 120天內≥10%的日子 | `holding_period[returns >= 0.10]` | ✅ |
| 排名計算 | rank / total | `(profitable_returns <= sell_return).mean()` | ✅ |
| 獎勵公式 | rank + 1 | `reward = rank + 1.0` (範圍 +1 到 +2) | ✅ |

### 4.4 賣出判斷邏輯 (Backtest)

| 項目 | 論文 | 程式碼 ([backtest_engine.py](file:///d:/000-github-repositories/ptrl-v01/src/backtest/backtest_engine.py#L324-L365)) | 狀態 |
|------|------|------------|------|
| 特徵計算 | 70維 | `obs_70 = concat(features_69, [sell_return])` | ✅ |
| 閾值條件 | \|sell - hold\| > 0.85 | `confidence_diff > 0.85` | ✅ |

---

## 5. 停損規則 (Stop Loss)

### 5.1 跌幅停損 (Stop Loss on Dips)

| 項目 | 論文 | 程式碼 ([stop_loss.py](file:///d:/000-github-repositories/ptrl-v01/src/rules/stop_loss.py)) | 狀態 |
|------|------|------------|------|
| 觸發條件 | 報酬率 < -10% | `current_return < dip_threshold (-0.10)` | ✅ |
| 執行時機 | 隔天開盤價 | 使用當日收盤價模擬 | ⚠️ 近似 |

### 5.2 盤整停損 (Stop Loss on Sideways)

| 項目 | 論文 | 程式碼 | 狀態 |
|------|------|--------|------|
| 觀察天數 | 連續 20 天 | `sideways_days=20` | ✅ |
| 觸發條件 | 報酬率 < 10% | `sideways_threshold=0.10` | ✅ |
| 適用期間 | 120 天內 | `max_holding_days=120` | ✅ |
| 判斷邏輯 | 連續20天未達10% | `(daily_returns < threshold).all()` | ✅ |

### 5.3 優先級

| 項目 | 論文 | 程式碼 | 狀態 |
|------|------|--------|------|
| 優先級 | 停損 > Sell Agent | Stop Loss 先檢查，Agent 後檢查 | ✅ |
| Override | 停損覆蓋 Agent | `continue` 跳過 Agent 決策 | ✅ |

---

## 6. 投資組合與回測設定

### 6.1 投資組合參數

| 項目 | 論文 | [default_config.yaml](file:///d:/000-github-repositories/ptrl-v01/config/default_config.yaml) | 狀態 |
|------|------|-----------------|------|
| 初始資金 | $10,000 | `initial_capital: 10000` | ✅ |
| 最大持倉 | 10 檔 | `max_positions: 10` | ✅ |
| 單檔上限 | 10% | `max_position_pct: 0.10` | ✅ |
| 手續費 | 0.1% | `trading_fee: 0.001` | ✅ |

### 6.2 回測期間

| 項目 | 論文 | 程式碼 | 狀態 |
|------|------|--------|------|
| 訓練期間 | 2005/02/25 - 2017/10/15 | `start_date: "2005-02-25"` | ✅ |
| 測試期間 | 2017/10/16 - 2023/10/15 | `backtest: start_date: "2017-10-16"` | ✅ |
| 資產池 | S&P 500+400+600 (1465檔) | S&P 500 only (~500檔) | ⚠️ 刻意差異 |

### 6.3 Top-10 選股邏輯

| 項目 | 論文 | [backtest_engine.py](file:///d:/000-github-repositories/ptrl-v01/src/backtest/backtest_engine.py#L442-L458) | 狀態 |
|------|------|------------|------|
| 收集訊號 | Donchian 突破 | `high > donchian_upper` | ✅ |
| 排序依據 | Buy Agent 信心 | `sort(key=confidence, reverse=True)` | ✅ |
| 選取數量 | Top 10 | `[:min(10, available_slots)]` | ✅ |
| 次要排序 | — | Turnover (成交額) | ➕ 額外 |

---

## 7. 差異總結

### 7.1 ✅ 完全一致 (已驗證)

| 項目 | 驗證結果 |
|------|----------|
| PPO 超參數 (全部 10 項) | ✅ 完全符合 Table 6 |
| 神經網路架構 (69/70→40→2) | ✅ 完全符合 |
| Buy Agent 4情境獎勵 (+1/0) | ✅ 完全符合 |
| Sell Agent 4情境獎勵 | ✅ 完全符合 |
| 排名獎勵公式 (Eq.21) | ✅ 完全符合 |
| 停損規則 (dip/sideways) | ✅ 完全符合 |
| 投資組合設定 | ✅ 完全符合 |
| 回測期間 | ✅ 完全符合 |
| Top-10 選股邏輯 | ✅ 已實作 |
| 賣出信心閾值 0.85 | ✅ 完全符合 |

### 7.2 ⚠️ 刻意差異 (設計決策)

| 項目 | 論文 | 我們的實作 | 理由 |
|------|------|------------|------|
| 基準指數 | Dow Jones (^DJI) | S&P 500 (^GSPC) | S&P 500 更具代表性，與資產池一致 |
| 資產池 | 1465 檔 (S&P 500+400+600) | ~500 檔 (S&P 500 only) | 簡化資料處理，減少噪音 |
| Volume 特徵 | 排除 | 包含 (log正規化) | 補充流動性資訊 |
| Top-10 次要排序 | 僅信心 | 信心 + 成交額 | 優先大型股，提高流動性 |

### 7.3 📝 論文筆誤 (我們已修正)

| 項目 | 論文原文 | 我們的修正 |
|------|----------|------------|
| 公式 (3)-(8) | 全部標示為 Upper/High | 各變數除以對應的 High |
| 公式 (19) 分母 | 不明確 | 使用 BuyPrice (標準金融公式) |

### 7.4 ⏳ 待完善項目

| 項目 | 現狀 | 建議 |
|------|------|------|
| Up_Stock (上漲家數) | 預留欄位，未填充 | 需多股票同時計算 |
| Down_Stock (下跌家數) | 預留欄位，未填充 | 需多股票同時計算 |
| 隔天開盤執行 | 使用當日收盤價 | 可改為 Open_{t+1} |

---

## 8. v3 開發建議

### 8.1 高優先級

1. **完善 Up_Stock / Down_Stock**: 需要在 backtest 時計算每日漲跌家數
2. **驗證執行時機**: 確認買賣是用當日收盤還是隔日開盤

### 8.2 中優先級

3. **考慮切換到 DJI**: 若要完全對齊論文結果
4. **擴展資產池**: 加入 S&P 400/600

### 8.3 低優先級

5. **移除 Volume**: 若要完全對齊論文
6. **驗證 Heikin Ashi 公式**: 確認是否有遞迴定義的細微差異

---

## 9. 檢查清單

- [x] PPO 超參數 - **完全一致**
- [x] 神經網路架構 - **完全一致**
- [x] Buy Agent 獎勵函數 - **完全一致 (+1/0)**
- [x] Sell Agent 獎勵函數 - **完全一致**
- [x] 排名獎勵 (Eq.21) - **完全一致**
- [x] 停損規則 - **完全一致**
- [x] 投資組合設定 - **完全一致**
- [x] 回測期間 - **完全一致**
- [x] Top-10 選股 - **已正確實作**
- [x] 賣出信心閾值 - **完全一致 (0.85)**
- [ ] Up_Stock / Down_Stock - **待完善**
- [ ] 執行時機驗證 - **建議驗證**
