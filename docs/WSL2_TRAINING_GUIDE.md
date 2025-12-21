# WSL2 多核訓練指南

## 快速開始 (複製貼上)

```bash
# 1. 進入 WSL
wsl

# 2. 進入專案目錄
cd /mnt/d/000-github-repositories/ptrl-v01

# 3. 安裝依賴 (首次需要)
pip3 install -r requirements.txt --break-system-packages

# 4. 訓練 (Buy + Sell 連續執行)
python3 scripts/train_multicore_wsl.py --agent buy --timesteps 12000000 --n-envs 12 && python3 scripts/train_multicore_wsl.py --agent sell --timesteps 12000000 --n-envs 4
```

---

## 監控訓練 (PowerShell)

```powershell
cd D:\000-github-repositories\ptrl-v01
tensorboard --logdir=logs/training/
# 開啟瀏覽器 http://localhost:6006
```

---

## 訓練指令說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--agent` | buy 或 sell | 必填 |
| `--timesteps` | 訓練步數 | 12,000,000 |
| `--n-envs` | CPU 核心數 | 自動 (CPU-2) |
| `--resume` | 從 checkpoint 恢復 | 否 |

---

## 輸出檔案

| 類型 | 路徑 |
|------|------|
| 模型 | `models/{buy,sell}_agent/final_model.zip` |
| Checkpoints | `models/checkpoints/{buy,sell}_agent/` |
| 日誌 | `logs/training/{buy,sell}_agent/` |

---

## 常見問題

### pip3 權限錯誤
```bash
pip3 install -r requirements.txt --break-system-packages
```

### python 找不到
```bash
sudo apt install python-is-python3 -y
```
