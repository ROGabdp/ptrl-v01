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
        
        # 特徵欄位 (排除 actual_return 和 is_successful)
        self.feature_cols = [col for col in signals_data.columns 
                             if col not in ['actual_return', 'is_successful', 'Date', 'symbol']]
        
        # 儲存原始資料
        self._original_data = signals_data.copy()
        
        # 資料平衡 (1:1)
        if self.balance_data:
            self.data = self._balance_data(signals_data)
        else:
            self.data = signals_data.copy()
        
        # 定義狀態空間 (69 維)
        n_features = len(self.feature_cols)
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
        
        logger.info(f"BuyEnv 初始化完成 - {len(self.data)} 個訓練樣本, {n_features} 維特徵")
    
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
        
        Returns:
            observation: 初始狀態 (69 維)
            info: 額外資訊
        """
        super().reset(seed=seed)
        
        # 打亂資料順序
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.current_idx = 0
        
        # 取得第一個觀察
        self.current_obs = self._get_observation(self.current_idx)
        
        info = {
            'sample_idx': self.current_idx,
            'total_samples': len(self.data)
        }
        
        return self.current_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        執行動作
        
        Args:
            action: 0 (不買) 或 1 (買)
            
        Returns:
            observation: 下一個狀態
            reward: 獎勵值
            terminated: 是否結束 (單步結束)
            truncated: 是否截斷
            info: 額外資訊
        """
        # 取得當前樣本的實際結果
        current_sample = self.data.iloc[self.current_idx]
        is_successful = current_sample.get('is_successful', False)
        actual_return = current_sample.get('actual_return', 0.0)
        
        # 計算獎勵
        # action=1 表示「預測會獲利」，如果實際成功則正確
        # action=0 表示「預測會虧損」，如果實際失敗則正確
        if action == 1:
            reward = 1.0 if is_successful else -1.0
        else:
            reward = 1.0 if not is_successful else -1.0
        
        # 移動到下一個樣本
        self.current_idx += 1
        
        # 檢查是否結束
        terminated = self.current_idx >= len(self.data)
        truncated = False
        
        # 取得下一個觀察
        if not terminated:
            self.current_obs = self._get_observation(self.current_idx)
        
        info = {
            'action': action,
            'is_successful': is_successful,
            'actual_return': actual_return,
            'correct_prediction': (action == 1 and is_successful) or (action == 0 and not is_successful)
        }
        
        return self.current_obs, reward, terminated, truncated, info
    
    def _get_observation(self, idx: int) -> np.ndarray:
        """取得指定索引的觀察向量"""
        if idx >= len(self.data):
            return np.zeros(len(self.feature_cols), dtype=np.float32)
        
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
