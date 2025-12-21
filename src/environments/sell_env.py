"""
SellEnv - Sell Knowledge RL 環境

根據論文設計:
- 狀態空間: 70 維 (69 個正規化特徵 + SellReturn)
- 動作空間: 2 (Action 1: 持有, Action 2: 賣出)
- 獎勵函數: 基於相對排名的獎勵 (鼓勵賣在高點)
- 訓練資料: 只使用最終達到 ≥10% 報酬的買入訊號
- 最大持有天數: 120 天
- 賣出條件: |sell_prob - hold_prob| > 0.85 且 sell_prob > hold_prob
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple
from loguru import logger


class SellEnv(gym.Env):
    """
    Sell Knowledge RL 環境
    
    角色:
        在買入後的 120 天內，決定何時賣出以最大化報酬
    
    狀態 (70 維):
        - 69 個正規化技術特徵
        - SellReturn: 當前價格 / 買入價格 (公式 20)
    
    動作 (2 個):
        - Action 0: 持有 (Hold)
        - Action 1: 賣出 (Sell)
    
    獎勵:
        基於賣出時機的相對排名獎勵:
        - 賣在最高點: +1 到 +2
        - 賣在最低點: -1 到 0
        - 持有超過 120 天: 強制賣出
    
    使用方式:
        env = SellEnv(trade_data, config)
        obs, info = env.reset()
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 trade_data: Dict[str, pd.DataFrame],
                 config: dict = None):
        """
        初始化 SellEnv
        
        Args:
            trade_data: 交易資料字典
                {
                    'symbol': {
                        'features': pd.DataFrame (每日特徵),
                        'buy_date': pd.Timestamp (買入日期),
                        'buy_price': float (買入價格)
                    },
                    ...
                }
            config: 設定字典
        """
        super().__init__()
        
        config = config or {}
        self.max_holding_days = config.get('max_holding_days', 120)
        self.success_threshold = config.get('success_threshold', 0.10)
        self.sell_threshold = config.get('sell_threshold', 0.85)
        
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
            self.data_np = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.existing_shm.buf)
            
            # 特徵欄位
            # 在 Pandas 模式下是從 DataFrame 動態取得，但在 Shared Memory 模式下，我們依賴 feature_idx_map
            # 排除非特徵欄位和特殊欄位 (Date, OHLCV, actual_return, is_successful)
            self.feature_cols = [col for col in self.feature_idx_map.keys() 
                                 if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'actual_return', 'is_successful', 'label']]
            # 確保順序一致
            self.feature_indices = [self.feature_idx_map[col] for col in self.feature_cols]
            
            # 準備 episodes (基於索引)
            # trade_data 在 Shared Memory 模式下是一個包含 episode metadata 的列表
            # [{'start_idx': 100, 'end_idx': 220, 'buy_price': 50.0}, ...]
            self.episodes = trade_data 
            logger.info(f"SellEnv (Shared Memory) 初始化完成 - {len(self.episodes)} 個 episodes")
            
        else:
            # 傳統模式 (Pandas Copy)
            # 將交易資料轉換為訓練用序列
            self.episodes = self._prepare_episodes(trade_data)
            
            # 取得特徵欄位名稱
            if self.episodes:
                first_episode = self.episodes[0]
                self.feature_cols = [col for col in first_episode['features'].columns 
                                    if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
            else:
                self.feature_cols = []
            
            logger.info(f"SellEnv (Standard) 初始化完成 - {len(self.episodes)} 個 episodes")
        
        # 定義狀態空間 (70 維 = 69 特徵 + SellReturn)
        n_features = len(self.feature_cols) + 1  # +1 for SellReturn
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features,), 
            dtype=np.float32
        )
        
        # 定義動作空間 (0: 持有, 1: 賣出)
        self.action_space = spaces.Discrete(2)
        
        # 內部狀態
        self.current_episode_idx = 0
        self.current_episode = None
        # Shared Memory 模式下的內部指標
        self.current_episode_start_idx = 0
        self.current_episode_len = 0
        
        self.current_day = 0
        self.buy_price = 0.0
        self.holding_days = 0
    
    def _prepare_episodes(self, trade_data: Dict) -> List[Dict]:
        """
        準備訓練用 episodes (Pandas 模式)
        
        每個 episode 代表一筆交易 (從買入到賣出)
        只使用最終達到 ≥10% 報酬的交易
        """
        episodes = []
        
        for key, data in trade_data.items():
            features = data.get('features')
            buy_date = data.get('buy_date')
            buy_price = data.get('buy_price')
            
            if features is None or buy_price is None:
                continue
            
            # 計算該交易的最終報酬率
            if 'Close' in features.columns:
                # 找到 120 天內的最高價
                holding_period = features.iloc[:self.max_holding_days] if len(features) > self.max_holding_days else features
                max_price = holding_period['Close'].max()
                max_return = (max_price - buy_price) / buy_price
                
                # 只使用成功的交易 (論文要求)
                if max_return >= self.success_threshold:
                    episodes.append({
                        'key': key,
                        'features': features,
                        'buy_price': buy_price,
                        'buy_date': buy_date,
                        'max_return': max_return
                    })
        
        logger.info(f"準備了 {len(episodes)} 個成功交易 episodes")
        return episodes
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        重置環境，開始新的 episode
        """
        super().reset(seed=seed)
        
        if len(self.episodes) == 0:
            raise ValueError("沒有可用的訓練 episode")
        
        # 選擇下一個 episode
        self.current_episode_idx = (self.current_episode_idx + 1) % len(self.episodes)
        self.current_episode = self.episodes[self.current_episode_idx]
        
        # 重置狀態
        self.current_day = 0
        self.holding_days = 0
        
        if self.use_shared_memory:
            # Shared Memory 模式
            # current_episode 是一個 dict: {'start_idx': ..., 'end_idx': ..., 'buy_price': ...}
            self.current_episode_start_idx = self.current_episode['start_idx']
            self.current_episode_len = self.current_episode['end_idx'] - self.current_episode['start_idx']
            self.buy_price = self.current_episode['buy_price']
            
            obs = self._get_observation()
            
            info = {
                'episode_idx': self.current_episode_idx,
                'buy_price': self.buy_price,
                'max_return': self.current_episode.get('max_return', 0.0)
            }
        else:
            # Pandas 模式
            # current_episode 是一個 dict: {'features': DataFrame, ...}
            self.buy_price = self.current_episode['buy_price']
            
            obs = self._get_observation()
            
            info = {
                'episode_idx': self.current_episode_idx,
                'buy_price': self.buy_price,
                'max_return': self.current_episode['max_return']
            }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        執行動作
        
        Args:
            action: 0 (持有) 或 1 (賣出)
        """
        # 取得當前價格
        if self.use_shared_memory:
            # Shared Memory 模式
            # 必須使用 feature_idx_map 來找到 'Close' 的索引
            close_idx = self.feature_idx_map.get('Close')
            if close_idx is None: raise ValueError("Shared Memory 中缺少 'Close' 欄位")
            
            # 安全地讀取 Close 價格
            # 若 current_day 超出範圍 (例如最後一天)，取最後一天的價格
            day_idx = min(self.current_day, self.current_episode_len - 1)
            global_idx = self.current_episode_start_idx + day_idx
            
            current_close = float(self.data_np[global_idx, close_idx])
            
            # 用於計算獎勵，這裡先用 current_return
            # 為了計算論文獎勵，我們需要傳入一些額外資訊給 _calculate_reward_paper
            # 這裡我們簡化：修改 _calculate_reward_paper 讓它能接受 current_return 和 features (若有)
            
        else:
            # Pandas 模式
            features = self.current_episode['features']
            if self.current_day < len(features):
                current_close = features.iloc[self.current_day]['Close']
            else:
                current_close = features.iloc[-1]['Close']
        
        # 計算當前報酬率
        current_return = (current_close - self.buy_price) / self.buy_price
        
        # 計算賣出時機的相對排名
        sell_return = current_return
        
        # 檢查是否達到最大持有天數
        self.holding_days += 1
        force_sell = self.holding_days >= self.max_holding_days
        
        # 決定是否結束
        terminated = (action == 1) or force_sell
        truncated = False
        
        # 計算獎勵 (論文 4 情境邏輯)
        # 注意: 我們需要傳遞 features 資訊給 _calculate_reward_paper
        # 對於 Shared Memory 模式，我們可能需要調整該方法
        if self.use_shared_memory:
            reward = self._calculate_reward_paper_shm(action, current_return, self.current_episode, self.current_day)
        else:
            reward = self._calculate_reward_paper(action, current_return, self.current_episode['features'])
        
        # 移動到下一天
        self.current_day += 1
        
        # 取得下一個觀察
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            'action': action,
            'current_return': current_return,
            'holding_days': self.holding_days,
            'force_sell': force_sell,
            'sell_return': sell_return if terminated else None
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """取得當前觀察 (70 維)"""
        if self.use_shared_memory:
            # Shared Memory 模式
            if self.current_day >= self.current_episode_len:
                return np.zeros(len(self.feature_cols) + 1, dtype=np.float32)
            
            # 從 data_np 讀取特徵
            global_idx = self.current_episode_start_idx + self.current_day
            feature_values = self.data_np[global_idx, self.feature_indices].astype(np.float32)
            
            # 計算 SellReturn
            # 取 Open 或 Close，若無 Open 則用 Close
            open_idx = self.feature_idx_map.get('Open')
            close_idx = self.feature_idx_map.get('Close')
            
            current_price = self.buy_price # Fallback
            if open_idx is not None:
                current_price = self.data_np[global_idx, open_idx]
            elif close_idx is not None:
                current_price = self.data_np[global_idx, close_idx]
                
            sell_return = current_price / self.buy_price
            
            # 合併為 70 維
            obs = np.concatenate([feature_values, [sell_return]])
            
        else:
            # Pandas 模式
            features = self.current_episode['features']
            
            if self.current_day >= len(features):
                return np.zeros(len(self.feature_cols) + 1, dtype=np.float32)
            
            # 取得 69 個特徵
            row = features.iloc[self.current_day]
            feature_values = row[self.feature_cols].values.astype(np.float32)
            
            # 計算 SellReturn (公式 20)
            current_open = row.get('Open', row.get('Close', self.buy_price))
            sell_return = current_open / self.buy_price
            
            # 合併為 70 維
            obs = np.concatenate([feature_values, [sell_return]])
        
        # 處理 NaN
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _calculate_reward_paper(self, action: int, current_return: float, features: pd.DataFrame) -> float:
        """
        論文定義的 4 情境獎勵函數
        
        情境 1: 賣出 (action=1) 且報酬 >= 10% → 排名獎勵 (+1 到 +2)
        情境 2: 賣出 (action=1) 且報酬 < 10%  → -1 (錯誤賣低)
        情境 3: 持有 (action=0) 且報酬 < 10%  → +0.5 (正確耐心)
        情境 4: 持有 (action=0) 且報酬 >= 10% → -1 (錯失良機)
        """
        is_profitable = current_return >= self.success_threshold  # 10%
        
        if action == 1:  # 賣出動作
            if is_profitable:
                # 情境 1: 賣出成功 - 給予排名獎勵
                return self._calculate_ranking_reward(current_return, features)
            else:
                # 情境 2: 賣出失敗 (賣太早/賣虧損) - 懲罰
                return -1.0
        else:  # 持有動作 (action == 0)
            if is_profitable:
                # 情境 4: 有錢不賺 - 懲罰 (錯過賣點)
                return -1.0
            else:
                # 情境 3: 正確等待 - 小獎勵
                return 0.5
    
    def _calculate_ranking_reward(self, sell_return: float, features: pd.DataFrame) -> float:
        """
        計算排名獎勵 (公式 21)
        
        在 120 天內所有 >= 10% 的日子中，根據當前賣出點的排名給予獎勵
        賣在最高點得 +2，賣在最低的 10% 點得 +1
        """
        holding_period = features.iloc[:self.max_holding_days] if len(features) > self.max_holding_days else features
        
        # 計算所有可能的賣出報酬率
        all_returns = (holding_period['Close'] - self.buy_price) / self.buy_price
        
        # 只考慮 >= 10% 的日子
        profitable_returns = all_returns[all_returns >= self.success_threshold]
        
        if len(profitable_returns) == 0:
            return 1.0  # 沒有其他 >= 10% 的日子，給予基本獎勵
        
        # 計算排名 (0-1，1 = 最高)
        rank = (profitable_returns <= sell_return).mean()
        
        # 轉換為獎勵 (+1 到 +2)
        # rank=1 (最高) -> reward=2
        # rank=0 (最低) -> reward=1
        reward = rank + 1.0
        
        return float(reward)
    
    def _calculate_reward_paper_shm(self, action: int, current_return: float, episode_info: Dict, current_day: int) -> float:
        """
        計算論文定義的獎勵 (Shared Memory 版本)
        
        Args:
            action: 動作
            current_return: 當前報酬率
            episode_info: episode 資訊 (包含 start_idx, end_idx)
            current_day: 當前天數 (相對 index)
        """
        # 1. 如果是持有 (Action 0)
        if action == 0:
            return -1.0 if current_return > 0.10 else 0.5
        
        # 2. 如果是賣出 (Action 1)
        # 需要計算排名獎勵
        
        # 取得 episode 期間的所有價格
        # 我們不能直接拿整個 DataFrame，因為是在 Shared Memory
        # 但我們可以利用 start_idx 和 end_idx 來切片
        start_idx = episode_info['start_idx']
        episode_len = episode_info['end_idx'] - start_idx
        
        # 我們需要 Close 價格序列
        close_idx = self.feature_idx_map.get('Close')
        if close_idx is None: return 0.0 # Should not happen
        
        # 取得整個 episode 的 Close 價格序列 (這是一個 copy，雖然有點開銷但比 Pandas 快)
        # 注意: 只取到 max_holding_days (如果 episode 更長)
        actual_len = min(episode_len, self.max_holding_days)
        prices = self.data_np[start_idx : start_idx + actual_len, close_idx]
        
        # 計算每天的報酬率
        returns = (prices - self.buy_price) / self.buy_price
        
        # 排序報酬率 (由大到小)
        sorted_returns = np.sort(returns)[::-1]
        
        # 找出當前報酬率的排名 (0-based)
        # 處理浮點數精確度問題，使用 isclose 或差值
        # 這裡簡化: 找出最接近的值的 index
        # diff = np.abs(sorted_returns - current_return)
        # rank = np.argmin(diff) 
        
        # 或者更嚴謹: 有多少個報酬率大於當前報酬率
        rank = np.sum(sorted_returns > current_return)
        
        # 計算排名比例 (前幾%)
        rank_pct = rank / len(sorted_returns)
        
        # 根據排名給予獎勵 (公式 21)
        # 論文: Top 10% -> 2, Top 25% -> 1, Bottom 25% -> -1
        # 但論文實際上是說: R_sell = 2 (if rank in top 1/8), 1 (if rank in top 1/4) ...
        # 這裡沿用 _calculate_ranking_reward 的邏輯 (該方法已實作論文邏輯)
        # 但我們需要將邏輯搬過來或呼叫它 (如果它不依賴 DataFrame)
        
        # 公式 21 實作:
        T = len(sorted_returns)
        idx = rank + 1 # 1-based ranking
        
        if idx <= T/8:
            return 2.0
        elif idx <= T/4:
            return 1.0
        elif idx >= (7*T)/8:
            return -1.0 # 論文可能有錯字，通常賣在最低點應該懲罰
        else:
            return 0.0 # 中間區域


    
    def get_state_dim(self) -> int:
        """取得狀態維度 (70)"""
        return len(self.feature_cols) + 1


class SellEnvSimple(gym.Env):
    """
    簡化版 SellEnv (用於單一股票的訓練)
    
    輸入: 單一股票的日線資料
    每個 episode: 一次完整的買入-持有-賣出過程
    """
    
    def __init__(self, 
                 daily_features: pd.DataFrame,
                 buy_signals: pd.Series,
                 config: dict = None):
        """
        Args:
            daily_features: 每日特徵資料 (正規化後)
            buy_signals: 買入訊號 (True/False Series)
            config: 設定
        """
        super().__init__()
        
        config = config or {}
        self.max_holding_days = config.get('max_holding_days', 120)
        self.success_threshold = config.get('success_threshold', 0.10)
        
        self.features = daily_features
        self.buy_signals = buy_signals
        
        # 找出所有成功的買入點
        self.successful_entries = self._find_successful_entries()
        
        # 特徵欄位
        self.feature_cols = [col for col in daily_features.columns 
                             if '_norm' in col or col in ['Return', 'Index_Return', 'SuperTrend_14', 'SuperTrend_21']]
        
        # 狀態空間 (70 維)
        n_features = len(self.feature_cols) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
        
        # 動作空間
        self.action_space = spaces.Discrete(2)
        
        # 狀態
        self.current_entry_idx = 0
        self.current_day = 0
        self.buy_price = 0.0
        self.buy_idx = 0
        
        logger.info(f"SellEnvSimple 初始化 - {len(self.successful_entries)} 個成功入場點")
    
    def _find_successful_entries(self) -> List[int]:
        """找出所有成功的買入點索引"""
        entries = []
        signal_indices = self.buy_signals[self.buy_signals == True].index.tolist()
        
        for idx in signal_indices:
            if idx not in self.features.index:
                continue
            
            loc = self.features.index.get_loc(idx)
            buy_price = self.features.loc[idx, 'Close'] if 'Close' in self.features.columns else 0
            
            if buy_price <= 0:
                continue
            
            # 檢查後續 120 天內是否達到 10%
            end_loc = min(loc + self.max_holding_days, len(self.features))
            future_prices = self.features.iloc[loc:end_loc]['Close'] if 'Close' in self.features.columns else pd.Series()
            
            if len(future_prices) > 0:
                max_return = (future_prices.max() - buy_price) / buy_price
                if max_return >= self.success_threshold:
                    entries.append(loc)
        
        return entries
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置環境"""
        super().reset(seed=seed)
        
        if len(self.successful_entries) == 0:
            raise ValueError("沒有成功的入場點")
        
        # 隨機選擇入場點
        self.current_entry_idx = np.random.randint(len(self.successful_entries))
        self.buy_idx = self.successful_entries[self.current_entry_idx]
        self.current_day = 0
        self.buy_price = self.features.iloc[self.buy_idx]['Close']
        
        obs = self._get_observation()
        info = {'buy_idx': self.buy_idx, 'buy_price': self.buy_price}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """執行動作"""
        current_idx = self.buy_idx + self.current_day
        
        if current_idx >= len(self.features):
            return np.zeros(len(self.feature_cols) + 1, dtype=np.float32), 0.0, True, False, {}
        
        current_price = self.features.iloc[current_idx]['Close']
        current_return = (current_price - self.buy_price) / self.buy_price
        
        self.current_day += 1
        force_sell = self.current_day >= self.max_holding_days
        terminated = (action == 1) or force_sell
        
        if terminated:
            # 計算相對排名獎勵
            end_idx = min(self.buy_idx + self.max_holding_days, len(self.features))
            all_prices = self.features.iloc[self.buy_idx:end_idx]['Close']
            all_returns = (all_prices - self.buy_price) / self.buy_price
            rank = (all_returns <= current_return).mean()
            reward = 3 * rank - 1
        else:
            reward = 0.0
        
        obs = self._get_observation() if not terminated else np.zeros(len(self.feature_cols) + 1, dtype=np.float32)
        
        info = {
            'current_return': current_return,
            'holding_days': self.current_day,
            'force_sell': force_sell
        }
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """取得觀察"""
        current_idx = self.buy_idx + self.current_day
        
        if current_idx >= len(self.features):
            return np.zeros(len(self.feature_cols) + 1, dtype=np.float32)
        
        row = self.features.iloc[current_idx]
        features = row[self.feature_cols].values.astype(np.float32) if self.feature_cols else np.array([])
        
        current_open = row.get('Open', row.get('Close', self.buy_price))
        sell_return = current_open / self.buy_price if self.buy_price > 0 else 1.0
        
        obs = np.concatenate([features, [sell_return]]) if len(features) > 0 else np.array([sell_return], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    print("SellEnv 模組載入成功")
    print("SellEnv: 70 維狀態 (69 特徵 + SellReturn)")
    print("動作: 0=持有, 1=賣出")
    print("獎勵: 基於相對排名 (-1 到 +2)")
