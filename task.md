# Pro Trader RL - Data Restoration & Retraining

- [x] 下載完整歷史資料 (2004-01-01 至今)
- [x] 計算全量特徵
- [/] 實作多核並行訓練優化 (SubprocVecEnv)
- [x] 執行初始模型訓練 (各 1M steps)
  - [x] Buy Agent 訓練 (100萬步完成)
  - [x] Sell Agent 訓練 (100萬步完成)
- [x] 執行強化模型訓練 (至 10M steps, 接續訓練)
  - [x] Buy Agent 強化訓練 (達到 10.1M)
  - [x] Sell Agent 強化訓練 (達到 5M)
- [/] 執行完整期間回測 (2017-2023)
- [/] 驗證績效指標是否接近論文結果


## 論文對齊修正
- [x] 修正 SellEnv 獎勵函數 (4 情境邏輯)
- [x] 修正 train.py Sell 過濾門檻 (5% → 10%)
- [x] 增加訓練步數至 10M
- [/] 重新訓練並驗證
