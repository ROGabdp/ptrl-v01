# Pro Trader RL - Data Restoration & Retraining

- [x] 下載完整歷史資料 (2004-01-01 至今)
- [x] 計算全量特徵
- [x] Debug `EOFError` with `SubprocVecEnv` on Windows
    - [x] Refactor nested `make_env` to top-level `EnvCreator` class (fix pickling)
    - [x] Add pickling verification step before `SubprocVecEnv` creation
    - [x] Implement robust fallback to `DummyVecEnv` for Windows
- [x] Debug Feature Mismatch (127 vs 128/69)
    - [x] Align `BuyEnv` feature columns with `paper` config (69 features)
    - [x] Integrate `DataNormalizer.get_normalized_feature_columns()`
- [x] Optimize Imports for Subprocesses
    - [x] Move heavy imports (Agents, Rules) to local scope to prevent TF loading in workers
- [x] Execute Multicore Training (Buy Agent)
    - [x] Verify efficient data loading with Shared Memory
    - [x] Confirm FPS improvement (Achieved ~7000 FPS)
  - [x] Sell Agent 強化訓練 (達到 5M)
- [/] 執行完整期間回測 (2017-2023)
- [/] 驗證績效指標是否接近論文結果


## 論文對齊修正
- [x] 修正 SellEnv 獎勵函數 (4 情境邏輯)
- [x] 修正 train.py Sell 過濾門檻 (5% → 10%)
- [x] 增加訓練步數至 10M
- [/] 重新訓練並驗證
