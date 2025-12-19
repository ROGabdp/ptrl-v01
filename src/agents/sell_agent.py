"""
SellAgent - Sell Knowledge RL Agent

使用 PPO 演算法 (Stable-Baselines3) 實作的賣出決策 Agent

功能特點:
- PPO 演算法 (論文 Table 6 超參數)
- 70 維輸入 (69 特徵 + SellReturn)
- 賣出判斷閾值: |sell_prob - hold_prob| > 0.85
- 定期儲存檢查點
- TensorBoard 即時監控
"""

import os
import json
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from loguru import logger

# 從 buy_agent 匯入共用 Callbacks
from .buy_agent import TensorBoardCallback, TrainingStateCallback


class SellAgent:
    """
    Sell Knowledge RL Agent
    
    在買入後的 120 天內，決定最佳賣出時機
    
    賣出判斷邏輯 (論文):
        1. 計算 hold_prob 和 sell_prob
        2. 如果 |sell_prob - hold_prob| > 0.85 且 sell_prob > hold_prob，則賣出
        3. 否則持有
    
    使用方式:
        agent = SellAgent(config)
        agent.train(env, total_timesteps=1000000)
        action, should_sell = agent.predict_with_threshold(observation)
    """
    
    def __init__(self, config: dict = None):
        """
        初始化 SellAgent
        
        Args:
            config: 設定字典
        """
        config = config or {}
        
        # PPO 超參數 (論文 Table 6)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.n_steps = config.get('n_steps', 2048)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # 神經網路架構 (論文: 70 -> 40 -> 2)
        self.policy_kwargs = config.get('policy_kwargs', {
            'net_arch': dict(pi=[40], vf=[40])
        })
        
        # 賣出判斷閾值
        self.sell_threshold = config.get('sell_threshold', 0.85)
        
        # 路徑設定
        self.checkpoint_dir = config.get('checkpoint_dir', 'models/checkpoints/sell_agent/')
        self.best_model_dir = config.get('best_model_dir', 'models/sell_agent/')
        self.tensorboard_log = config.get('tensorboard_log', 'logs/training/sell_agent/')
        
        # 訓練設定
        self.checkpoint_freq = config.get('checkpoint_freq', 10000)
        self.eval_freq = config.get('eval_freq', 5000)
        self.n_eval_episodes = config.get('n_eval_episodes', 10)
        self.verbose = config.get('verbose', 1)
        
        # 模型
        self.model = None
        
        # 確保目錄存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log, exist_ok=True)
        
        logger.info("SellAgent 初始化完成")
    
    def train(self, env, total_timesteps: int, 
              eval_env = None,
              resume: bool = False) -> None:
        """
        訓練 Agent
        
        Args:
            env: 訓練環境 (SellEnv)
            total_timesteps: 總訓練步數
            eval_env: 評估環境 (可選)
            resume: 是否從最新檢查點恢復訓練
        """
        # 包裝環境
        if not isinstance(env, DummyVecEnv):
            env = DummyVecEnv([lambda: Monitor(env)])
        
        resumed_steps = 0
        
        # 檢查是否恢復訓練
        if resume:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path:
                logger.info(f"從檢查點恢復訓練: {checkpoint_path}")
                # 注意: 不要傳入 tensorboard_log，讓它使用 checkpoint 中保存的設定
                self.model = PPO.load(checkpoint_path, env=env)
                # 設定 tensorboard log 路徑
                self.model.tensorboard_log = self.tensorboard_log
                resumed_steps = self.model.num_timesteps
                logger.info(f"從第 {resumed_steps:,} 步繼續訓練，目標 {total_timesteps:,} 步")
                
                # 計算剩餘需要訓練的步數
                remaining_steps = max(0, total_timesteps - resumed_steps)
                if remaining_steps == 0:
                    logger.info("已達到目標步數，無需繼續訓練")
                    return
            else:
                logger.warning("找不到檢查點，從頭開始訓練")
                resume = False
        
        # 建立新模型
        if not resume or self.model is None:
            self.model = PPO(
                policy='MlpPolicy',
                env=env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                policy_kwargs=self.policy_kwargs,
                tensorboard_log=self.tensorboard_log,
                verbose=self.verbose
            )
        
        # 建立 Callbacks
        callbacks = self._create_callbacks(eval_env)
        
        # 開始訓練
        if resume and resumed_steps > 0:
            # Resume 時：只訓練剩餘的步數，讓總步數達到目標
            remaining = total_timesteps - resumed_steps
            logger.info(f"繼續訓練 SellAgent - 剩餘 {remaining:,} 步 (已完成 {resumed_steps:,}，目標 {total_timesteps:,})")
            train_steps = remaining
        else:
            logger.info(f"開始訓練 SellAgent - 總步數: {total_timesteps:,}")
            train_steps = total_timesteps
        logger.info(f"TensorBoard 指令: tensorboard --logdir={self.tensorboard_log}")
        
        self.model.learn(
            total_timesteps=train_steps,  # 使用剩餘步數而非總步數
            callback=callbacks,
            reset_num_timesteps=False,  # 永遠不重置，保持累積計數
            tb_log_name="PPO"  # 使用固定名稱，避免每次創建新的 run
        )

        
        # 儲存最終模型
        final_path = os.path.join(self.best_model_dir, 'final_model.zip')
        self.model.save(final_path)
        logger.info(f"訓練完成，最終模型已儲存至: {final_path}")

    
    def _create_callbacks(self, eval_env = None) -> CallbackList:
        """建立訓練 Callbacks"""
        callbacks = []
        
        # 1. 檢查點 Callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_freq,
            save_path=self.checkpoint_dir,
            name_prefix='sell_agent',
            save_replay_buffer=False,
            save_vecnormalize=True
        )
        callbacks.append(checkpoint_callback)
        
        # 2. 評估 Callback (僅用於監控，不儲存 best_model)
        # 注意: eval_env 使用測試期資料，因此不應用於模型選擇
        if eval_env is not None:
            if not isinstance(eval_env, VecEnv):
                eval_env = DummyVecEnv([lambda: Monitor(eval_env)])
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=None,  # 不根據 eval 儲存 best model
                log_path=self.tensorboard_log,
                eval_freq=self.eval_freq,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        
        # 3. 訓練狀態 Callback
        state_callback = TrainingStateCallback(
            save_path=self.checkpoint_dir,
            save_freq=self.checkpoint_freq
        )
        callbacks.append(state_callback)
        
        # 4. TensorBoard Callback
        tb_callback = TensorBoardCallback()
        callbacks.append(tb_callback)
        
        return CallbackList(callbacks)
    
    def predict(self, observation: np.ndarray, 
                deterministic: bool = True) -> int:
        """
        預測動作 (純模型輸出)
        
        Args:
            observation: 70 維觀察向量
            deterministic: 是否使用確定性策略
            
        Returns:
            動作 (0: 持有, 1: 賣出)
        """
        if self.model is None:
            raise ValueError("模型尚未訓練或載入")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def predict_proba(self, observation: np.ndarray) -> np.ndarray:
        """
        預測動作機率
        
        Args:
            observation: 70 維觀察向量
            
        Returns:
            [持有機率, 賣出機率]
        """
        if self.model is None:
            raise ValueError("模型尚未訓練或載入")
        
        obs = np.array(observation).reshape(1, -1)
        action_probs = self.model.policy.get_distribution(
            self.model.policy.obs_to_tensor(obs)[0]
        ).distribution.probs.detach().cpu().numpy()[0]
        
        return action_probs
    
    def predict_with_threshold(self, observation: np.ndarray) -> Tuple[int, bool]:
        """
        使用論文閾值判斷是否賣出
        
        論文邏輯:
            如果 |sell_prob - hold_prob| > 0.85 且 sell_prob > hold_prob，則賣出
        
        Args:
            observation: 70 維觀察向量
            
        Returns:
            (action, should_sell): action 是模型原始輸出，should_sell 是閾值判斷結果
        """
        probs = self.predict_proba(observation)
        hold_prob = probs[0]
        sell_prob = probs[1]
        
        # 論文判斷邏輯
        prob_diff = abs(sell_prob - hold_prob)
        should_sell = prob_diff > self.sell_threshold and sell_prob > hold_prob
        
        action = 1 if should_sell else 0
        
        return action, should_sell
    
    def save(self, path: str) -> None:
        """儲存模型"""
        if self.model is None:
            raise ValueError("模型尚未訓練")
        self.model.save(path)
        logger.info(f"模型已儲存至: {path}")
    
    def load(self, path: str) -> None:
        """載入模型"""
        self.model = PPO.load(path)
        logger.info(f"模型已載入: {path}")
    
    def load_best_model(self) -> None:
        """載入最佳模型"""
        best_path = os.path.join(self.best_model_dir, 'best_model.zip')
        if os.path.exists(best_path):
            self.load(best_path)
        else:
            logger.warning(f"最佳模型不存在: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str = None) -> None:
        """載入檢查點"""
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
        else:
            logger.warning(f"檢查點不存在: {checkpoint_path}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """取得最新的檢查點路徑"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                       if f.startswith('sell_agent_') and f.endswith('_steps.zip')]
        
        if not checkpoints:
            return None
        
        # 按步數排序
        checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
        latest = checkpoints[-1]
        
        return os.path.join(self.checkpoint_dir, latest)
    
    def get_training_progress(self) -> Dict:
        """取得訓練進度"""
        state_path = os.path.join(self.checkpoint_dir, 'training_state.json')
        
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                return json.load(f)
        
        return {
            'total_timesteps': 0,
            'best_mean_reward': None,
            'last_checkpoint': None
        }


# =============================================================================
# 使用範例
# =============================================================================

if __name__ == '__main__':
    print("SellAgent 載入成功")
    print("使用方式:")
    print("  agent = SellAgent(config)")
    print("  agent.train(env, total_timesteps=1000000)")
    print("  action, should_sell = agent.predict_with_threshold(obs)")
