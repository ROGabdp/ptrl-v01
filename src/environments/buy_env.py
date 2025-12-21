"""
BuyEnv - Buy Knowledge RL 環境

根據論文設計:
- 狀態空間: 69 維正規化特徵
- 動作空間: 2 (Action 1: 不買, Action 2: 買)
- 獎勵函數: 基於預測準確性 (是否正確預測 ≥10% 報酬)
- 訓練資料: Donchian Channel 買入訊號點
- 資料平衡: 成功/失敗交易 1:1
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple
from loguru import logger


class BuyEnv(gym.Env):
    """
    Buy Knowledge RL 環境
    
    角色:
        作為 Donchian Channel 買入訊號的「過濾器」，
        判斷該訊號是否會帶來 ≥10% 的報酬率
    
    狀態 (69 維):
        正規化後的技術特徵
    
    動作 (2 個):
        - Action 0: 不執行買入 (預測會虧損)
        - Action 1: 執行買入 (預測會獲利 ≥10%)
    
    獎勵:
        - 正確預測: +1
        - 錯誤預測: -1
    
    使用方式:
        env = BuyEnv(signals_data, config)
        obs, info = env.reset()
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 signals_data: pd.DataFrame,
                 config: dict = None):
        """
        初始化 BuyEnv
        
        Args:
            signals_data: 包含以下欄位的 DataFrame:
                - 69 個正規化特徵欄位
                - 'actual_return': 實際報酬率 (買入後的結果)
                - 'is_successful': 是否達到 ≥10% 報酬
            config: 設定字典
        """
        super().__init__()
        
        config = config or {}
        self.success_threshold = config.get('success_threshold', 0.10)
        self.balance_data = config.get('balance_data', True)
        
        # Shared Memory 支援
        self.use_shared_memory = config.get('use_shared_memory', False)
        
        if self.use_shared_memory:
            from multiprocessing.shared_memory import SharedMemory
            self.shm_name = config.get('shm_name')
            self.shm_shape = config.get('shm_shape')
            self.shm_dtype = config.get('shm_dtype')
            self.feature_idx_map = config.get('feature_idx_map', {})
            
            # 連接現有的 Shared Memory
            self.existing_shm = SharedMemory(name=self.shm_name)
            # 建立 Numpy array (不複製數據，直接用 buffer)
            self.data_np = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.existing_shm.buf)
            # 為了相容性，self.data 設為 None 或包裝過的物件 (這裡主要依賴 self.data_np)
            self.data = pd.DataFrame(self.data_np, columns=list(self.feature_idx_map.keys())) # 視情況可能不需要這行，若 _get_obs 改寫的話
            
            # 為了資料平衡，我們需要在 train_multicore.py 就先平衡好，這裡只負責讀取
            # 這裡假設傳入 Shared Memory 的資料已經是平衡好的
            logger.info(f"BuyEnv (Shared Memory) 初始化完成 - {len(self.data_np)} 個訓練樣本")
            
            # 定義 feature_cols (排除非特徵欄位)
            # 注意: map keys 可能包含 Date, symbol 等
            exclude_cols = ['actual_return', 'is_successful', 'Date', 'symbol', 'label', 'symbol_idx']
            
            # 優先使用 DataNormalizer 定義的標準 69 特徵 (如果可用)
            # 這能解決與 Paper Agent Checkpoint (69 features) 的兼容性問題
            try:
                from src.data.normalizer import DataNormalizer
                standard_features = DataNormalizer().get_normalized_feature_columns()
                # 只保留存在於 shm 中的標準特徵
                self.feature_cols = [col for col in standard_features if col in self.feature_idx_map]
                logger.info(f"Using Standard 69 Features Mode ({len(self.feature_cols)} found)")
            except ImportError:
                # Fallback 到排除法
                self.feature_cols = [col for col in self.feature_idx_map.keys() if col not in exclude_cols]
                logger.warning(f"DataNormalizer not found, using exclusion mode ({len(self.feature_cols)} features)")

            logger.info(f"BuyEnv Features ({len(self.feature_cols)}): {self.feature_cols[:5]}...")
            
            # 設定特徵欄位索引 (加速讀取)
            self.feature_indices = [self.feature_idx_map[col] for col in self.feature_cols]
            
        else:
            # 傳統模式 (Pandas Copy)
            # 處理標籤: 如果有 'label' 欄位，轉換為 'is_successful'
            if 'label' in signals_data.columns and 'is_successful' not in signals_data.columns:
                signals_data = signals_data.copy()
                signals_data['is_successful'] = signals_data['label'] == 2
            
            # 特徵欄位 (排除 actual_return 和 is_successful)
            self.feature_cols = [col for col in signals_data.columns 
                                 if col not in ['actual_return', 'is_successful', 'Date', 'symbol', 'label']]
            
            # 儲存原始資料
            self._original_data = signals_data.copy()
            
            # 資料平衡 (1:1)
            if self.balance_data:
                self.data = self._balance_data(signals_data)
            else:
                self.data = signals_data.copy()
            
            logger.info(f"BuyEnv (Standard) 初始化完成 - {len(self.data)} 個訓練樣本, {len(self.feature_cols)} 維特徵")
            
        # 定義狀態空間 (69 維)
        n_features = len(self.feature_cols) if not self.use_shared_memory else len(self.feature_indices)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features,), 
            dtype=np.float32
        )
        
        # 定義動作空間 (0: 不買, 1: 買)
        self.action_space = spaces.Discrete(2)
        
        # 內部狀態
        self.current_idx = 0
        self.current_obs = None
        
        # 索引管理 (用於 Shuffle)
        if self.use_shared_memory:
            self.data_len = len(self.data_np)
        else:
            self.data_len = len(self.data)
            
        self.row_indices = np.arange(self.data_len, dtype=np.int32)

    
    def _balance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        平衡成功/失敗交易資料 (1:1)
        
        論文要求: 訓練時成功和失敗的交易樣本數量相同
        """
        if 'is_successful' not in data.columns:
            logger.warning("資料缺少 'is_successful' 欄位，跳過平衡")
            return data
        
        successful = data[data['is_successful'] == True]
        failed = data[data['is_successful'] == False]
        
        # 取較少的數量
        min_count = min(len(successful), len(failed))
        
        if min_count == 0:
            logger.warning("成功或失敗樣本數為 0")
            return data
        
        # 隨機抽樣達到平衡
        balanced_successful = successful.sample(n=min_count, random_state=42)
        balanced_failed = failed.sample(n=min_count, random_state=42)
        
        balanced_data = pd.concat([balanced_successful, balanced_failed])
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"資料平衡完成: {len(balanced_data)} 樣本 (成功:{min_count}, 失敗:{min_count})")
        return balanced_data
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        重置環境
        """
        super().reset(seed=seed)
        
        # 設定隨機種子 (如果有的話)
        if seed is not None:
            np.random.seed(seed)
        
        # 打亂索引順序 (無論是 Shared Memory 還是 Pandas 模式)
        # 這確保了每個 Episode 看到的數據順序是隨機的
        np.random.shuffle(self.row_indices)
        
        self.current_idx = 0
        
        # 取得第一個觀察 (使用映射後的真實索引)
        real_idx = self.row_indices[self.current_idx]
        self.current_obs = self._get_observation(real_idx)
        
        info = {
            'sample_idx': int(real_idx),
            'total_samples': self.data_len
        }
        
        return self.current_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        執行動作
        """
        # 取得當前真實索引
        real_idx = self.row_indices[self.current_idx]

        if self.use_shared_memory:
            # Shared Memory 模式
            current_row = self.data_np[real_idx]
            
            is_successful_idx = self.feature_idx_map.get('is_successful')
            actual_return_idx = self.feature_idx_map.get('actual_return')
            
            is_successful = bool(current_row[is_successful_idx]) if is_successful_idx is not None else False
            actual_return = float(current_row[actual_return_idx]) if actual_return_idx is not None else 0.0
            
        else:
            # 傳統模式 (Pandas) - 如果是 Pandas 模式，reset 雖然 shuffle 了 indices，
            # 但 self.data 沒有被 shuffle (我們改用 indices access)
            # 所以這裡也要用 indices
            # 注意：原來的 reset 邏輯是 shuffle self.data。現在我們統一用 indices。
            current_sample = self.data.iloc[real_idx]
            is_successful = current_sample.get('is_successful', False)
            actual_return = current_sample.get('actual_return', 0.0)
        
        # 計算獎勵
        if action == 1:
            reward = 1.0 if is_successful else 0.0
        else:
            reward = 1.0 if not is_successful else 0.0
        
        # 移動到下一個樣本
        self.current_idx += 1
        
        # 檢查是否結束
        terminated = self.current_idx >= self.data_len
        truncated = False
        
        # 取得下一個觀察
        if not terminated:
            next_real_idx = self.row_indices[self.current_idx]
            self.current_obs = self._get_observation(next_real_idx)
        
        info = {
            'action': action,
            'is_successful': is_successful,
            'actual_return': actual_return,
            'correct_prediction': (action == 1 and is_successful) or (action == 0 and not is_successful)
        }
        
        return self.current_obs, reward, terminated, truncated, info
    
    def _get_observation(self, idx: int) -> np.ndarray:
        """取得指定索引的觀察向量"""
        total_len = len(self.data_np) if self.use_shared_memory else len(self.data)
        
        if idx >= total_len:
            # 回傳零向量
            dim = len(self.feature_indices) if self.use_shared_memory else len(self.feature_cols)
            return np.zeros(dim, dtype=np.float32)
        
        if self.use_shared_memory:
            # Numpy 模式: 直接使用預先計算好的 indices 切片
            # 這裡假設 self.data_np 包含了所有欄位，我們只取特徵欄位
            # 使用 fancy indexing (注意: 這會產生 copy，但對於 69 維 float32 來說非常快)
            obs = self.data_np[idx, self.feature_indices].astype(np.float32)
        else:
            # Pandas 模式
            sample = self.data.iloc[idx]
            obs = sample[self.feature_cols].values.astype(np.float32)
        
        # 處理 NaN 和 Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def get_state_dim(self) -> int:
        """取得狀態維度"""
        return len(self.feature_cols)
    
    def render(self):
        """渲染環境 (可選)"""
        if self.current_idx < len(self.data):
            sample = self.data.iloc[self.current_idx]
            print(f"樣本 {self.current_idx}/{len(self.data)}")
            print(f"  是否成功: {sample.get('is_successful', 'N/A')}")
            print(f"  實際報酬: {sample.get('actual_return', 'N/A'):.2%}")


class BuyEnvEpisodic(BuyEnv):
    """
    分段式 BuyEnv (每個 episode 包含多個決策)
    
    與基本 BuyEnv 的差異:
    - 每個 episode 包含 N 個樣本
    - episode 結束時計算整體準確率作為獎勵
    """
    
    def __init__(self, signals_data: pd.DataFrame, config: dict = None):
        super().__init__(signals_data, config)
        config = config or {}
        self.episode_length = config.get('episode_length', 100)
        self.episode_start_idx = 0
        self.episode_rewards = []
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置 episode"""
        obs, info = super().reset(seed=seed, options=options)
        self.episode_start_idx = self.current_idx
        self.episode_rewards = []
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """執行動作"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        self.episode_rewards.append(reward)
        
        # 檢查 episode 是否結束
        steps_in_episode = self.current_idx - self.episode_start_idx
        if steps_in_episode >= self.episode_length:
            terminated = True
            # 計算 episode 總獎勵
            info['episode_accuracy'] = sum(1 for r in self.episode_rewards if r > 0) / len(self.episode_rewards)
        
        return obs, reward, terminated, truncated, info


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    import numpy as np
    
    # 創建模擬資料
    n_samples = 1000
    n_features = 69
    
    # 模擬特徵
    feature_data = np.random.randn(n_samples, n_features)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    # 模擬結果
    actual_returns = np.random.uniform(-0.2, 0.3, n_samples)
    is_successful = actual_returns >= 0.10
    
    # 建立 DataFrame
    df = pd.DataFrame(feature_data, columns=feature_cols)
    df['actual_return'] = actual_returns
    df['is_successful'] = is_successful
    
    print(f"成功率: {is_successful.mean():.2%}")
    
    # 建立環境
    env = BuyEnv(df)
    
    # 測試
    obs, info = env.reset()
    print(f"觀察維度: {obs.shape}")
    print(f"動作空間: {env.action_space}")
    
    # 執行幾步
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"步驟 {i+1}: action={action}, reward={reward}, correct={info['correct_prediction']}")
        if terminated:
            break
    
    print("\nBuyEnv 測試完成!")
