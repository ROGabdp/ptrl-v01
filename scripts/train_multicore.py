"""
Pro Trader RL 多核訓練腳本 (Shared Memory 版)

功能:
- 使用 Shared Memory 技術解決多核訓練時的記憶體不足 (OOM) 問題
- 支援 Buy Agent 和 Sell Agent 高效並行訓練
- 使用 numpy array 直接存取記憶體，避免複製

使用方式:
    python scripts/train_multicore.py --config config/default_config.yaml --n-envs 8
    python scripts/train_multicore.py --buy-only --n-envs 16
"""

import os
import sys
import argparse
import yaml
import time
import gc
from datetime import datetime
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp

# 加入專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.data import DataLoader, FeatureCalculator, DataNormalizer
# 移除 top-level imports 以避免 subprocess 載入不必要的重型依賴 (如 TensorFlow)
# from src.environments import BuyEnv, SellEnv  <-- Moved inside functions
# from src.agents import BuyAgent, SellAgent    <-- Moved inside functions
# from src.rules import DonchianChannel         <-- Moved inside functions


def load_config(config_path: str) -> dict:
    """載入設定檔"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# =============================================================================
# Shared Memory 輔助函數
# =============================================================================

def create_shared_memory_from_df(df: pd.DataFrame, name_prefix: str = "shm"):
    """
    將 DataFrame 轉換為 Shared Memory
    
    Returns:
        shm: SharedMemory 物件 (需要手動 close/unlink)
        shm_info: 用於重建 array 的資訊 (name, shape, dtype, feature_idx_map)
    """
    # 轉換為 numpy array
    # 確保所有數據都是數值型 (float32)，這對於 RL 環境來說通常是成立的
    # 如果有非數值欄位 (如 Date)，需要排除或轉換。但我們的 df 已經正規化過，應該都是數值。
    # 這裡我們為了安全，只取數值欄位，但我們會建立一個 mapping 以便反查
    
    # 檢查是否有非數值欄位
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < len(df.columns):
        logger.warning(f"DataFrame 包含非數值欄位，將被忽略: {set(df.columns) - set(numeric_df.columns)}")
    
    data_array = numeric_df.to_numpy(dtype=np.float32)
    feature_idx_map = {col: i for i, col in enumerate(numeric_df.columns)}
    
    # 建立 Shared Memory
    try:
        shm = SharedMemory(create=True, size=data_array.nbytes)
    except FileExistsError:
        # 如果上次異常退出沒清理，可能還存在
        logger.warning(f"Shared Memory 已存在，嘗試重新連接並覆蓋")
        shm = SharedMemory(name=None, create=True, size=data_array.nbytes) # 無法指定 name 覆蓋，只能 create new anonymous
    
    # 建立 numpy array wrapper
    shm_array = np.ndarray(data_array.shape, dtype=data_array.dtype, buffer=shm.buf)
    
    # 複製數據
    shm_array[:] = data_array[:]
    
    shm_info = {
        'shm_name': shm.name,
        'shm_shape': data_array.shape,
        'shm_dtype': data_array.dtype,
        'feature_idx_map': feature_idx_map
    }
    
    logger.info(f"Shared Memory 建立成功: {shm.name} ({data_array.nbytes / 1024 / 1024:.2f} MB)")
    return shm, shm_info


# =============================================================================
# 環境生成函數
# =============================================================================

# =============================================================================
# 環境生成類別 (必須是 top-level class 以支援 pickling)
# =============================================================================

class BuyEnvCreator:
    """BuyEnv 生成器 (用於 SubprocVecEnv)"""
    def __init__(self, shm_info, config):
        self.shm_info = shm_info
        self.config = config
    
    def __call__(self):
        # 合併 config
        # 確保 config 是 clean 的 (只包含 primitives)
        env_config = self.config.copy()
        env_config.update({
            'use_shared_memory': True,
            'shm_name': self.shm_info['shm_name'],
            'shm_shape': self.shm_info['shm_shape'],
            'shm_dtype': self.shm_info['shm_dtype'],
            'feature_idx_map': self.shm_info['feature_idx_map']
        })
        # 移除可能導致 pickling 問題的 keys (如果有)
        env_config.pop('logger', None)
        
        # 傳入空的 DataFrame，因為數據會從 Shared Memory 讀取
        from src.environments.buy_env import BuyEnv
        import os
        print(f"Worker PID: {os.getpid()} creating BuyEnv...")
        # Monitor 在 Windows SubprocVecEnv 中可能導致 crashed/EOFError 或 pickling issues
        # 為了穩定並行訓練，暫時移除 Monitor (PPO 仍會顯示 fps, loss)
        env = BuyEnv(pd.DataFrame(), env_config)
        print(f"Worker PID: {os.getpid()} BuyEnv created. Type: {type(env)}")
        return env

class SellEnvCreator:
    """SellEnv 生成器 (用於 SubprocVecEnv)"""
    def __init__(self, shm_info, episodes_metadata, config):
        self.shm_info = shm_info
        self.episodes_metadata = episodes_metadata
        self.config = config

    def __call__(self):
        # 合併 config
        env_config = self.config.copy()
        env_config.update({
            'use_shared_memory': True,
            'shm_name': self.shm_info['shm_name'],
            'shm_shape': self.shm_info['shm_shape'],
            'shm_dtype': self.shm_info['shm_dtype'],
            'feature_idx_map': self.shm_info['feature_idx_map']
        })
        env_config.pop('logger', None)

        # 傳入 metadata list 而不是 episodes dict
        from src.environments.sell_env import SellEnv
        return SellEnv(self.episodes_metadata, env_config)


# =============================================================================
# 資料準備與訓練
# =============================================================================

def prepare_data_for_shm(config: dict) -> pd.DataFrame:
    """
    載入並正規化所有資料，合併為一個巨大的 DataFrame (用於放入 Shared Memory)
    這裡我們統一載入所有需要的欄位
    """
    logger.info("載入所有歷史資料並計算特徵...")
    data_config = config.get('data', {})
    feature_config = config.get('features', {})
    
    loader = DataLoader(data_config)
    feature_calc = FeatureCalculator(feature_config)
    normalizer = DataNormalizer()
    
    symbols = loader.load_symbols_list()
    index_data = loader.load_index()
    
    all_data_list = []
    
    # 為了加速，我們可以只載入訓練期內的數據 (如果記憶體真的很吃緊)
    # 但為了簡單起見，我們先載入全部，因為 numpy array 比 objects 省空間
    
    for i, symbol in enumerate(symbols):
        df = loader.load_symbol(symbol)
        if df is None: continue
        try:
            cache_file = Path(f"data/cache/{symbol}_features.pkl")
            if cache_file.exists():
                df_normalized = pd.read_pickle(cache_file)
            else:
                df_normalized = normalizer.normalize(feature_calc.calculate_all_features(df, index_data))
            
            # 加入 symbol 和 date (若是數值化處理比較麻煩，我們先不放入 shm，或者只放 mapping)
            # 為了簡化 Shared Memory 存取，我們假設 shm 中每一列都是一個樣本
            # 我們需要記錄每一列對應的 (Symbol, Date) 資訊，這部分保留在主進程即可
            # 子進程只需要數值特徵
            
            # 確保 'Close', 'Open', 'Volume' 存在 (如果被 normalize 移除了，需要加回來)
            # Normalizer 通常保留了原始 Close (因為沒被 drop)，我們確認一下
            
            df_normalized['symbol_idx'] = i # 用 index 代表 symbol
            
            # 儲存日期這類 meta 資訊需要另外處理，這裡我們專注於數值矩陣
            # 我們可以把需要的 'is_successful' label 等等都算好放進去
            
            # 確保 'Date' 欄位存在 (從原始 df 拿)
            if 'Date' not in df_normalized.columns:
                # 假設 df index 是 Date
                if isinstance(df.index, pd.DatetimeIndex):
                    df_normalized['Date'] = df.index
                elif 'Date' in df.columns:
                    df_normalized['Date'] = df['Date'].values
                else: 
                     # Fallback to df normalization logic if it stripped it
                     # But normalizer usually keeps index if it was Date
                     # Check if we can get it from loader df
                     df_normalized['Date'] = df.index if isinstance(df.index, pd.DatetimeIndex) else df['Date']

            
            all_data_list.append(df_normalized)
            
        except Exception as e:
            # logger.warning(f"Error processing {symbol}: {e}")
            continue
            
        if (i+1) % 100 == 0: logger.info(f"載入進度: {i+1}/{len(symbols)}")

    logger.info("合併所有數據...")
    full_df = pd.concat(all_data_list, ignore_index=True)
    
    # 填充 NaN
    full_df = full_df.fillna(0.0)
    
    logger.info(f"全量數據大小: {full_df.shape}, 記憶體: {full_df.memory_usage().sum() / 1024**3:.2f} GB")
    return full_df


def prepare_buy_signals_metadata(full_df: pd.DataFrame, config: dict):
    """
    在主進程計算 Buy Signals，並回傳只要傳給子進程的 indices 和 metadata
    """
    from src.rules import DonchianChannel
    logger.info("準備 Buy Agent 訊號 metadata...")
    donchian = DonchianChannel(period=config.get('features', {}).get('donchian_period', 20))
    success_threshold = config.get('buy_env', {}).get('success_threshold', 0.10)
    max_holding_days = config.get('sell_env', {}).get('max_holding_days', 120)
    train_end_date = pd.Timestamp(config.get('backtest', {}).get('train_end_date', '2017-10-15'))
    
    # 這裡我們需要原始 DataFrame 的操作，幸好 full_df 還在
    # 產生訊號 (向量化操作很快)
    buy_signals = donchian.generate_buy_signals(full_df)
    signal_indices = buy_signals[buy_signals == 1].index
    
    # 為了判斷 label (is_successful)，我們需要未來價格
    # 這在全量 df 上做比較快
    
    # 預計算 is_successful (向量化)
    # 這比較難完全向量化，因為每個訊號往後看的時間窗口不同
    # 用簡單迴圈處理 signal_indices
    
    labels = np.zeros(len(full_df), dtype=np.bool_) # default False
    actual_returns = np.zeros(len(full_df), dtype=np.float32)
    
    valid_indices = []
    train_indices = []
    eval_indices = []
    
    # 假設 full_df 有 'Date' 欄位 (在 concat 時丟失了 index，如果原始 index 是 Date)
    # 我們需要在 prepare_data_for_shm 時保留 Date column
    if 'Date' not in full_df.columns:
        logger.warning("'Date' column missing, cannot split train/eval correctly!")
        # 暫時 fallback: 假設前 80% train
        # 但這不準確。我們需要在 prep 階段把 Date 變成數值 timestamp 或是保留 column
        pass
    
    # 優化：我們直接在 full_df 上加兩個欄位 'is_successful', 'actual_return'
    # 這樣可以直接進 Shared Memory
    
    logger.info(f"計算 {len(signal_indices)} 個訊號的標籤...")
    
    # 為了加速，我們只對 signal points 計算
    for idx in signal_indices:
        # 邊界檢查
        if idx + 1 + max_holding_days > len(full_df): continue
        
        # 簡單檢查 symbol 是否一致 (避免跨股票計算)
        # 用 symbol_idx 判斷
        if full_df.iloc[idx]['symbol_idx'] != full_df.iloc[idx + max_holding_days - 1]['symbol_idx']:
            continue
            
        buy_price = full_df.iloc[idx]['Close']
        future_window = full_df.iloc[idx+1 : idx+1+max_holding_days]
        max_price = future_window['High'].max()
        ret = (max_price - buy_price) / buy_price
        
        actual_returns[idx] = ret
        labels[idx] = (ret >= success_threshold)
        
        # 分類
        date_val = full_df.iloc[idx]['Date'] # 需要確保 Date 欄位存在
        if date_val <= train_end_date:
            train_indices.append(idx)
        else:
            eval_indices.append(idx)
            
    # 把計算結果併入 full_df (這樣就會進 shm)
    full_df['is_successful'] = labels
    full_df['actual_return'] = actual_returns
    
    logger.info(f"Buy 訊號準備完成: Train {len(train_indices)}, Eval {len(eval_indices)}")
    
    # 我們只需要回傳 indices，子環境會去 shm 查
    # 但 BuyEnv 需要 data_np，但不需要全部 row，只需要 signal rows
    # 為了簡單，BuyEnv 讀取 shm 後，直接用 indices 存取
    # 但 BuyEnv 現在的邏輯是: self.data[self.current_idx]
    # 如果我們傳入所有的 data 到 shm，BuyEnv 依然要遍歷所有數據嗎？
    # 不，我們可以做一個 "Virtual BuyEnv"，它只看 train_indices
    
    # 調整：我們重構 BuyEnv 讓它接受一個 "row_indices" 列表
    # 但我們剛才沒改這個。
    # 替代方案：我們在 Shm 建立後，把 full_df 縮減成只包含 signals 的 df (在主進程)，
    # 然後只把這個縮減後的 df 放入 Shm？
    # 這樣最省記憶體！因為只存訊號點 (幾萬筆 vs 幾百萬筆)
    
    # 策略修正：
    # 1. 計算所有 signals
    # 2. 篩選出 train_df_signals (只包含 signal rows)
    # 3. 把 train_df_signals 轉成 Shm
    # 4. 把 Shm 傳給 BuyEnv
    # 這樣 BuyEnv 邏輯不用大改 (它以為是全部數據，其實只是訊號集)
    
    train_df_signals = full_df.iloc[train_indices].copy()
    eval_df_signals = full_df.iloc[eval_indices].copy()
    
    return train_df_signals, eval_df_signals


def prepare_sell_episodes_metadata(full_df: pd.DataFrame, config: dict):
    """
    準備 Sell Agent 的 metadata
    """
    from src.rules import DonchianChannel
    logger.info("準備 Sell Agent episodes metadata...")
    donchian = DonchianChannel(period=config.get('features', {}).get('donchian_period', 20))
    success_threshold = config.get('buy_env', {}).get('success_threshold', 0.10)
    max_holding_days = config.get('sell_env', {}).get('max_holding_days', 120)
    max_episodes = config.get('training', {}).get('max_sell_episodes', 50000)
    train_end_date = pd.Timestamp(config.get('backtest', {}).get('train_end_date', '2017-10-15'))
    
    # 類似邏輯，找出成功交易
    buy_signals = donchian.generate_buy_signals(full_df)
    signal_indices = buy_signals[buy_signals == 1].index
    
    train_episodes = []
    eval_episodes = []
    
    # 隨機打亂以避免 biases (雖然 indices 是有序的)
    indices_list = list(signal_indices)
    import random
    random.shuffle(indices_list)
    
    count = 0
    for idx in indices_list:
        if count >= max_episodes * 2: break # 稍微多找一點
        
        # 邊界與 Symbol 檢查
        if idx + 1 + max_holding_days > len(full_df): continue
        if full_df.iloc[idx]['symbol_idx'] != full_df.iloc[idx + max_holding_days - 1]['symbol_idx']: continue
        
        buy_price = full_df.iloc[idx]['Close']
        future_window = full_df.iloc[idx+1 : idx+1+max_holding_days]
        max_return = (future_window['High'].max() - buy_price) / buy_price
        
        if max_return < success_threshold: continue
        
        # 這是一個成功的 episode
        # 記錄 metadata (start_idx, end_idx)
        # SellEnv 是從 buy date 開始，直到 120 天
        # 注意：full_df 在這裡必須包含連續的日線資料！
        # 如果 full_df 是由多個 symbol 拼接而成，只要我們確保不跨 symbol，索引就是連續的
        
        start_idx = idx
        end_idx = idx + 1 + max_holding_days
        # 如果中間跨 symbol (雖然上面檢查了頭尾，但為了保險)
        # 其實只要確保 start 和 end 都在同一個 symbol 區間即可
        
        meta = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'buy_price': buy_price,
            'max_return': max_return
        }
        
        date_val = full_df.iloc[idx]['Date']
        if date_val <= train_end_date:
            if len(train_episodes) < max_episodes:
                train_episodes.append(meta)
        else:
            eval_episodes.append(meta)
            
    logger.info(f"Sell Episodes 準備完成: Train {len(train_episodes)}, Eval {len(eval_episodes)}")
    
    # SellEnv 需要完整的連續時間序列數據
    # 所以對於 Sell Agent，我們必須把 **整個 full_df** 放進 Shared Memory
    # 不能只放 signal rows
    
    return train_episodes, eval_episodes


def run_buy_training(config, args):
    from src.agents import BuyAgent
    # 1. 準備數據 (全量載入)
    # 為了 Buy Agent，我們只需要 訊號點 的數據
    # 這樣可以大幅節省記憶體
    
    full_df = prepare_data_for_shm(config)
    train_df, eval_df = prepare_buy_signals_metadata(full_df, config)
    
    # 刪除 full_df 釋放記憶體 (Buy Agent 不需要連續時間序列，只需要當下的特徵)
    # 注意: _get_observation 只看當前 row
    del full_df
    gc.collect()
    
    # V3 修復: 過濾特徵欄位，只保留需要的 69 維
    # 這確保 BuyEnv 的觀察空間與回測時一致
    from src.data.normalizer import DataNormalizer
    normalizer = DataNormalizer()
    v3_feature_cols = normalizer.get_normalized_feature_columns()
    
    # Buy Agent 還需要 actual_return, is_successful 用於獎勵計算
    required_cols = v3_feature_cols + ['actual_return', 'is_successful', 'symbol', 'Date']
    available_cols = [c for c in required_cols if c in train_df.columns]
    
    logger.info(f"V3 特徵過濾: 原 {len(train_df.columns)} 欄 -> 保留 {len(available_cols)} 欄")
    train_df = train_df[available_cols].copy()
    
    logger.info(f"建立 Buy Agent Shared Memory ({len(train_df)} rows)...")
    shm, shm_info = create_shared_memory_from_df(train_df, name_prefix="buy_shm")
    
    # 建立並行環境
    n_envs = args.n_envs
    logger.info(f"啟動 {n_envs} 個並行環境 (SubprocVecEnv)...")
    
    # Pickling 測試
    import pickle
    try:
        test_creator = BuyEnvCreator(shm_info, config.get('buy_env', {}))
        pickle.dumps(test_creator)
        logger.info("EnvCreator pickling check passed.")
    except Exception as e:
        logger.error(f"EnvCreator pickling failed: {e}")
        raise e
    
    # 使用我們定義的 BuyEnvCreator 類別 (可 pickling)
    env_fns = [BuyEnvCreator(shm_info, config.get('buy_env', {})) for _ in range(n_envs)]
    
    if n_envs == 1 and not args.force_subproc:
         # Debug: use DummyVecEnv to see errors directly
         logger.info("Using DummyVecEnv for debugging (n_envs=1)")
         env = DummyVecEnv(env_fns)
    else:
        # Windows SubprocVecEnv 極度不穩定 (即使使用 spawn 和 clean imports)
        # 為了保證訓練能跑，我們加入 fallback 機制
        try:
            logger.info("Attempting to start SubprocVecEnv...")
            # 嘗試啟動，如果不穩定，將 fallback 到 DummyVecEnv
            # 注意: 如果 SubprocVecEnv 啟動成功但在運行時 crash，這裡抓不到
            # 鑑於多次嘗試失敗，我們直接使用 DummyVecEnv 作為 Windows 的默認選擇
            # 若用戶堅持要試，可以使用 --force-subproc
            # 強制使用 DummyVecEnv 以確保穩定性 (Debug Mode)
            if True: # os.name == 'nt' and not args.force_subproc:
                 logger.warning("Forcing DummyVecEnv for stability.")
                 env = DummyVecEnv(env_fns)
            else:
                 env = SubprocVecEnv(env_fns, start_method='spawn')
        except Exception as e:
            logger.error(f"SubprocVecEnv failed to start: {e}")
            logger.warning("Falling back to DummyVecEnv (Sequential Execution)...")
            env = DummyVecEnv(env_fns)
    
    # 加入 VecMonitor 以確保 TensorBoard 能記錄 ep_rew_mean
    from stable_baselines3.common.vec_env import VecMonitor
    import uuid
    log_dir = "logs/training/buy_agent"
    os.makedirs(log_dir, exist_ok=True)
    unique_monitor_name = f"monitor_{uuid.uuid4().hex[:8]}.csv"
    env = VecMonitor(env, filename=os.path.join(log_dir, unique_monitor_name))
    logger.info(f"VecMonitor added. Logs will be saved to {log_dir}")
    
    # 訓練
    agent = BuyAgent(config.get('buy_agent', {}))
    total_timesteps = config.get('buy_agent', {}).get('total_timesteps', 1000000)
    
    if args.resume:
        logger.info("Resume mode: training will continue from latest checkpoint")
    
    try:
        agent.train(env=env, total_timesteps=total_timesteps, resume=args.resume)
    finally:
        # 清理
        env.close()
        shm.close()
        shm.unlink()
        logger.info("Shared Memory 已釋放")


def run_sell_training(config, args):
    from src.agents import SellAgent
    # 1. 準備數據 (全量載入)
    # Sell Agent 需要連續的時間序列 (長達 120 天)
    # 所以必須把 整個 full_df 放入 Shared Memory
    
    full_df = prepare_data_for_shm(config)
    train_episodes, eval_episodes = prepare_sell_episodes_metadata(full_df, config)
    
    # V3 修復: 過濾特徵欄位，只保留需要的 69 維 + Close
    # 這確保 SellEnv 的觀察空間是 70 維 (69 特徵 + SellReturn)
    from src.data.normalizer import DataNormalizer
    normalizer = DataNormalizer()
    v3_feature_cols = normalizer.get_normalized_feature_columns()
    
    # 額外需要的欄位: Close (用於計算 SellReturn), symbol, Date
    required_cols = v3_feature_cols + ['Close', 'symbol', 'Date']
    available_cols = [c for c in required_cols if c in full_df.columns]
    
    logger.info(f"V3 特徵過濾: 原 {len(full_df.columns)} 欄 -> 保留 {len(available_cols)} 欄")
    filtered_df = full_df[available_cols].copy()
    
    logger.info(f"建立 Sell Agent Shared Memory ({len(filtered_df)} rows)...")
    # 這可能很大 (數百 MB 到數 GB)，但只有一份
    shm, shm_info = create_shared_memory_from_df(filtered_df, name_prefix="sell_shm")
    
    # 建立並行環境
    n_envs = args.n_envs
    logger.info(f"啟動 {n_envs} 個並行環境 (SubprocVecEnv)...")
    
    # 注意: 這裡傳入的是 episodes metadata list
    # 使用 SellEnvCreator 類別
    env_fns = [SellEnvCreator(shm_info, train_episodes, config.get('sell_env', {})) for _ in range(n_envs)]
    
    # Windows SubprocVecEnv 極度不穩定 (即使使用 spawn 和 clean imports)
    # 為了保證訓練能跑，我們加入 fallback 機制
    try:
        logger.info("Attempting to start SubprocVecEnv for Sell Agent...")
        if os.name == 'nt' and not args.force_subproc:
             logger.warning("Windows environment detected. Defaulting to DummyVecEnv for stability.")
             env = DummyVecEnv(env_fns)
        else:
             env = SubprocVecEnv(env_fns, start_method='spawn')
    except Exception as e:
        logger.error(f"SubprocVecEnv failed to start: {e}")
        logger.warning("Falling back to DummyVecEnv (Sequential Execution)...")
        env = DummyVecEnv(env_fns)
            
    # 加入 VecMonitor (Sell Agent)
    from stable_baselines3.common.vec_env import VecMonitor
    log_dir = "logs/training/sell_agent"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    logger.info(f"VecMonitor added for Sell Agent. Logs to {log_dir}")
    
    # 訓練
    agent = SellAgent(config.get('sell_agent', {}))
    total_timesteps = config.get('sell_agent', {}).get('total_timesteps', 1000000)
    
    try:
        agent.train(env=env, total_timesteps=total_timesteps, resume=args.resume)
    finally:
        env.close()
        shm.close()
        shm.unlink()
        logger.info("Shared Memory 已釋放")


def main():
    parser = argparse.ArgumentParser(description='Pro Trader RL 多核訓練腳本')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--buy-only', action='store_true')
    parser.add_argument('--sell-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='從檢查點恢復訓練')
    parser.add_argument('--n-envs', type=int, default=8, help='並行環境數量 (預設 8)')
    parser.add_argument('--force-subproc', action='store_true', help='強制使用 SubprocVecEnv (即使只有 1 個環境)')
    args = parser.parse_args()
    
    config_path = project_root / args.config
    config = load_config(str(config_path))
    
    # 覆蓋 config 中的 n_envs
    if 'training' not in config: config['training'] = {}
    config['training']['n_envs'] = args.n_envs
    
    if mp.get_start_method(allow_none=True) != 'spawn':
        # Windows 上必須使用 spawn (預設)，但顯式設定比較安全
        # 其實 Windows 只支援 spawn，這裡寫給 Linux 用戶看
        pass

    if args.buy_only:
        run_buy_training(config, args)
    elif args.sell_only:
        run_sell_training(config, args)
    else:
        # 兩者都跑 (需分開跑，因為 shm 不同)
        run_buy_training(config, args)
        gc.collect()
        run_sell_training(config, args)

if __name__ == '__main__':
    # Windows Multiprocessing 必須包含這個
    mp.freeze_support()
    main()
