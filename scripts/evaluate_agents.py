"""
Agent 獨立評估腳本

用途:
- Buy Agent 準確率評估 (Accuracy, Precision, Recall)
- Sell Agent 準確率評估 (捕獲率, 高點命中率)
- 多 Checkpoint 批次評估
- 產生比較報告

使用方式:
    python scripts/evaluate_agents.py --agent buy
    python scripts/evaluate_agents.py --agent sell
    python scripts/evaluate_agents.py --agent both
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# 確保 src 在路徑中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from src.data.data_loader import DataLoader
from src.data.feature_calculator import FeatureCalculator
from src.data.normalizer import DataNormalizer
from src.rules.stop_loss import DonchianChannel
import yaml


def load_config():
    """載入設定"""
    config_path = Path("config/default_config.yaml")
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_checkpoints(agent_type: str, target_steps: list = None):
    """
    找到指定步數的 checkpoint
    
    Args:
        agent_type: 'buy' or 'sell'
        target_steps: 目標步數列表，如 [5000000, 7500000, 10000000]
    
    Returns:
        dict: {步數: checkpoint 路徑}
    """
    checkpoint_dir = Path(f"models/checkpoints/{agent_type}_agent")
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint 目錄不存在: {checkpoint_dir}")
        return {}
    
    # 預設步數: 5M, 7.5M, 10M, 12.5M, 15M
    if target_steps is None:
        target_steps = [5000000, 7500000, 10000000, 12500000, 15000000]
    
    checkpoints = {}
    all_files = list(checkpoint_dir.glob(f"{agent_type}_agent_*_steps.zip"))
    
    # 解析所有可用的 checkpoint
    available = {}
    for f in all_files:
        try:
            steps = int(f.stem.split('_')[-2])
            available[steps] = f
        except:
            continue
    
    # 為每個目標步數找最接近的 checkpoint
    for target in target_steps:
        closest = min(available.keys(), key=lambda x: abs(x - target), default=None)
        if closest and abs(closest - target) < 500000:  # 容許 50萬步誤差
            checkpoints[target] = available[closest]
            logger.info(f"Target {target/1e6:.1f}M -> Found {closest/1e6:.2f}M: {available[closest].name}")
    
    # 加入 final model
    final_model = Path(f"models/{agent_type}_agent/final_model.zip")
    if final_model.exists():
        checkpoints['final'] = final_model
        logger.info(f"Final model: {final_model}")
    
    return checkpoints


def prepare_buy_test_data(config: dict):
    """
    準備 Buy Agent 測試資料
    
    Returns:
        DataFrame with columns: features..., actual_return, is_successful
    """
    logger.info("準備 Buy Agent 測試資料...")
    
    data_config = config.get('data', {})
    loader = DataLoader(data_config)
    feature_calc = FeatureCalculator(config.get('features', {}))
    normalizer = DataNormalizer()
    donchian = DonchianChannel(period=config.get('features', {}).get('donchian_period', 20))
    
    symbols = loader.load_symbols_list()
    index_data = loader.load_index()
    
    # 測試期間
    test_start = pd.Timestamp('2017-10-16')
    test_end = pd.Timestamp('2023-10-15')
    max_holding_days = config.get('sell_env', {}).get('max_holding_days', 120)
    success_threshold = config.get('buy_env', {}).get('success_threshold', 0.10)
    
    test_signals = []
    feature_cols = normalizer.get_normalized_feature_columns()
    
    for i, symbol in enumerate(symbols):
        df = loader.load_symbol(symbol)
        if df is None:
            continue
        
        try:
            # 計算特徵
            cache_file = Path(f"data/cache/{symbol}_features.pkl")
            if cache_file.exists():
                df_normalized = pd.read_pickle(cache_file)
            else:
                df_features = feature_calc.calculate_all_features(df, index_data)
                df_normalized = normalizer.normalize(df_features)
            
            # 產生買入訊號
            buy_signals = donchian.generate_buy_signals(df_normalized)
            signal_indices = buy_signals[buy_signals == 1].index
            
            for signal_date in signal_indices:
                # 只取測試期間的訊號
                if signal_date < test_start or signal_date > test_end:
                    continue
                
                if signal_date not in df_normalized.index:
                    continue
                
                signal_loc = df_normalized.index.get_loc(signal_date)
                buy_price = df_normalized.iloc[signal_loc]['Close']
                
                # 計算未來報酬
                future_data = df_normalized.iloc[signal_loc+1 : signal_loc+1+max_holding_days]
                if len(future_data) < 5:
                    continue
                
                max_return = (future_data['High'].max() - buy_price) / buy_price
                is_successful = max_return >= success_threshold
                
                # 取得特徵
                available_cols = [c for c in feature_cols if c in df_normalized.columns]
                row = df_normalized.iloc[signal_loc][available_cols].to_dict()
                row['actual_return'] = max_return
                row['is_successful'] = is_successful
                row['symbol'] = symbol
                row['date'] = signal_date
                
                test_signals.append(row)
                
        except Exception as e:
            continue
        
        if (i + 1) % 100 == 0:
            logger.info(f"進度: {i+1}/{len(symbols)}, 已收集 {len(test_signals)} 個訊號")
    
    df = pd.DataFrame(test_signals)
    logger.info(f"Buy Agent 測試資料: {len(df)} 個訊號, 成功率 {df['is_successful'].mean()*100:.1f}%")
    return df


def prepare_sell_test_data(config: dict):
    """
    準備 Sell Agent 測試資料 (只含成功交易的 episodes)
    
    Returns:
        list of episodes, each with: features_df, buy_price, max_return, max_return_day
    """
    logger.info("準備 Sell Agent 測試資料...")
    
    data_config = config.get('data', {})
    loader = DataLoader(data_config)
    feature_calc = FeatureCalculator(config.get('features', {}))
    normalizer = DataNormalizer()
    donchian = DonchianChannel(period=config.get('features', {}).get('donchian_period', 20))
    
    symbols = loader.load_symbols_list()
    index_data = loader.load_index()
    
    # 測試期間
    test_start = pd.Timestamp('2017-10-16')
    test_end = pd.Timestamp('2023-10-15')
    max_holding_days = config.get('sell_env', {}).get('max_holding_days', 120)
    success_threshold = 0.10
    
    episodes = []
    feature_cols = normalizer.get_normalized_feature_columns()
    
    for i, symbol in enumerate(symbols):
        df = loader.load_symbol(symbol)
        if df is None:
            continue
        
        try:
            cache_file = Path(f"data/cache/{symbol}_features.pkl")
            if cache_file.exists():
                df_normalized = pd.read_pickle(cache_file)
            else:
                df_features = feature_calc.calculate_all_features(df, index_data)
                df_normalized = normalizer.normalize(df_features)
            
            buy_signals = donchian.generate_buy_signals(df_normalized)
            signal_indices = buy_signals[buy_signals == 1].index
            
            for signal_date in signal_indices:
                if signal_date < test_start or signal_date > test_end:
                    continue
                
                if signal_date not in df_normalized.index:
                    continue
                
                signal_loc = df_normalized.index.get_loc(signal_date)
                buy_price = df_normalized.iloc[signal_loc]['Close']
                
                future_data = df_normalized.iloc[signal_loc+1 : signal_loc+1+max_holding_days]
                if len(future_data) < 5:
                    continue
                
                # 只取成功交易
                returns = (future_data['Close'] - buy_price) / buy_price
                max_return = returns.max()
                
                if max_return < success_threshold:
                    continue
                
                # 記錄最高報酬日
                max_return_day = returns.idxmax()
                max_return_day_idx = list(future_data.index).index(max_return_day)
                
                # 取得特徵
                available_cols = [c for c in feature_cols if c in df_normalized.columns]
                episode_df = future_data[available_cols + ['Close']].copy()
                
                episodes.append({
                    'symbol': symbol,
                    'buy_date': signal_date,
                    'buy_price': buy_price,
                    'features_df': episode_df,
                    'max_return': max_return,
                    'max_return_day_idx': max_return_day_idx
                })
                
        except Exception as e:
            continue
        
        if (i + 1) % 100 == 0:
            logger.info(f"進度: {i+1}/{len(symbols)}, 已收集 {len(episodes)} 個 episodes")
    
    logger.info(f"Sell Agent 測試資料: {len(episodes)} 個成功交易 episodes")
    return episodes


def evaluate_buy_accuracy(model_path: str, test_data: pd.DataFrame, feature_cols: list):
    """
    評估 Buy Agent - 使用信心度分布和排序精準度
    
    Buy Agent 的設計是輸出「會漲 10% 的機率」，用於 Top-10 排序選股
    因此評估方式是：
    1. 信心度分布統計
    2. 排序精準度：高信心度的訊號是否真的更容易成功
    3. Top-10 模擬成功率
    
    Returns:
        dict with confidence stats and ranking metrics
    """
    model = PPO.load(model_path)
    
    # 預測所有訊號的信心度
    confidences = []
    
    for idx, row in test_data.iterrows():
        features = row[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        
        # 取得信心度 (買入機率)
        obs_tensor = model.policy.obs_to_tensor(features.reshape(1, -1))[0]
        probs = model.policy.get_distribution(obs_tensor).distribution.probs.detach().cpu().numpy()[0]
        confidences.append(float(probs[1]))
    
    test_data = test_data.copy()
    test_data['confidence'] = confidences
    
    # === 1. 信心度分布統計 ===
    conf_mean = np.mean(confidences)
    conf_std = np.std(confidences)
    conf_max = np.max(confidences)
    conf_min = np.min(confidences)
    
    # === 2. 信心度與成功率的相關性 ===
    # 按信心度分組，計算各組成功率
    test_data['conf_decile'] = pd.qcut(test_data['confidence'], 10, labels=False, duplicates='drop')
    
    # 計算各分位數的成功率
    decile_stats = test_data.groupby('conf_decile').agg({
        'is_successful': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    # Top 10% vs Bottom 10% 成功率差異
    top_decile_success = test_data[test_data['conf_decile'] == test_data['conf_decile'].max()]['is_successful'].mean()
    bottom_decile_success = test_data[test_data['conf_decile'] == test_data['conf_decile'].min()]['is_successful'].mean()
    
    # 信心度與成功的相關係數
    correlation = test_data['confidence'].corr(test_data['is_successful'].astype(float))
    
    # === 3. Top-10 模擬 (每日選 Top-10) ===
    # 按日期分組，每天選信心度最高的 Top-10
    test_data['date'] = pd.to_datetime(test_data['date'])
    
    top10_results = []
    for date, group in test_data.groupby('date'):
        if len(group) >= 10:
            top10 = group.nlargest(10, 'confidence')
            top10_success_rate = top10['is_successful'].mean()
            all_success_rate = group['is_successful'].mean()
            top10_results.append({
                'date': date,
                'top10_success': top10_success_rate,
                'all_success': all_success_rate,
                'improvement': top10_success_rate - all_success_rate
            })
    
    if top10_results:
        top10_df = pd.DataFrame(top10_results)
        avg_top10_success = top10_df['top10_success'].mean()
        avg_all_success = top10_df['all_success'].mean()
        avg_improvement = top10_df['improvement'].mean()
    else:
        avg_top10_success = 0
        avg_all_success = test_data['is_successful'].mean()
        avg_improvement = 0
    
    return {
        # 信心度分布
        'conf_mean': conf_mean,
        'conf_std': conf_std,
        'conf_max': conf_max,
        'conf_min': conf_min,
        # 排序精準度
        'top_decile_success': top_decile_success,
        'bottom_decile_success': bottom_decile_success,
        'correlation': correlation,
        # Top-10 模擬
        'top10_success_rate': avg_top10_success,
        'all_success_rate': avg_all_success,
        'top10_improvement': avg_improvement,
        # 基本統計
        'total_signals': len(test_data),
        'actual_success_rate': test_data['is_successful'].mean()
    }


def evaluate_sell_accuracy(model_path: str, episodes: list, feature_cols: list, max_episodes: int = 2000):
    """
    評估 Sell Agent 準確率
    
    Args:
        model_path: 模型路徑
        episodes: 測試 episode 列表
        feature_cols: 特徵欄位
        max_episodes: 最大評估 episode 數量 (用於加速)
    
    Returns:
        dict with avg_return, capture_rate, hit_rate, avg_holding_days
    """
    model = PPO.load(model_path)
    
    # 抽樣以加速評估
    if len(episodes) > max_episodes:
        np.random.seed(42)  # 確保可重複性
        sample_idx = np.random.choice(len(episodes), max_episodes, replace=False)
        episodes = [episodes[i] for i in sample_idx]
        logger.info(f"  抽樣 {max_episodes} 個 episodes 進行評估")
    
    results = []
    
    for ep in episodes:
        features_df = ep['features_df']
        buy_price = ep['buy_price']
        max_return = ep['max_return']
        max_return_day_idx = ep['max_return_day_idx']
        
        # 模擬持有期
        sold = False
        sell_day = None
        sell_return = None
        
        for day_idx, (date, row) in enumerate(features_df.iterrows()):
            try:
                # 計算 SellReturn
                current_price = row['Close']
                if pd.isna(current_price) or current_price <= 0:
                    continue
                    
                sell_return_value = current_price / buy_price
                
                # 建構觀察向量 (69 特徵 + SellReturn)
                available_cols = [c for c in feature_cols if c in row.index]
                features = row[available_cols].values.astype(np.float32)
                
                # 強化 NaN/Inf 處理
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                obs = np.concatenate([features, [sell_return_value]])
                obs = np.clip(obs, -10, 10)  # 限制範圍避免極端值
                
                # 確認沒有 NaN
                if np.isnan(obs).any() or np.isinf(obs).any():
                    continue
                
                # 預測
                action, _ = model.predict(obs, deterministic=True)
                
                # 取得信心度
                obs_tensor = model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
                probs = model.policy.get_distribution(obs_tensor).distribution.probs.detach().cpu().numpy()[0]
                
                hold_prob = float(probs[0])
                sell_prob = float(probs[1])
                
                # 論文判斷邏輯: |sell_prob - hold_prob| > 0.85 且 sell_prob > hold_prob
                prob_diff = abs(sell_prob - hold_prob)
                should_sell = prob_diff > 0.85 and sell_prob > hold_prob
                
                if should_sell:
                    sold = True
                    sell_day = day_idx
                    sell_return = (current_price - buy_price) / buy_price
                    break
                    
            except Exception as e:
                continue
        
        # 如果沒賣，用最後一天
        if not sold:
            sell_day = len(features_df) - 1
            last_price = features_df.iloc[-1]['Close']
            if pd.notna(last_price) and last_price > 0:
                sell_return = (last_price - buy_price) / buy_price
            else:
                sell_return = 0.0
        
        # 計算是否賣在高點 (前 20%)
        hit_top20 = sell_day <= max_return_day_idx + int(len(features_df) * 0.2)
        
        results.append({
            'sell_return': sell_return,
            'max_return': max_return,
            'capture_rate': sell_return / max_return if max_return > 0 else 0,
            'hit_top20': hit_top20,
            'holding_days': sell_day + 1
        })
    
    df = pd.DataFrame(results)
    
    return {
        'avg_return': df['sell_return'].mean(),
        'max_possible_return': df['max_return'].mean(),
        'capture_rate': df['capture_rate'].mean(),
        'hit_rate': df['hit_top20'].mean(),
        'avg_holding_days': df['holding_days'].mean(),
        'total_episodes': len(episodes)
    }


def generate_report(results: dict, output_dir: Path, agent_type: str):
    """產生評估報告"""
    report_lines = []
    report_lines.append(f"# {agent_type.upper()} Agent Checkpoint 評估報告")
    report_lines.append(f"\n評估時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"測試期間: 2017-10-16 ~ 2023-10-15\n")
    
    if agent_type == 'buy':
        report_lines.append("## 信心度分布與排序效果\n")
        report_lines.append("| Checkpoint | Conf Mean | Conf Std | Top10% 成功率 | Bottom10% 成功率 | 相關係數 | Top-10 成功率 | 全體成功率 | 改善 |")
        report_lines.append("|------------|-----------|----------|--------------|-----------------|---------|-------------|-----------|------|")
        
        for step, metrics in results.items():
            step_label = f"{step/1e6:.1f}M" if isinstance(step, (int, float)) else step
            report_lines.append(
                f"| {step_label} | {metrics['conf_mean']:.3f} | {metrics['conf_std']:.3f} | "
                f"{metrics['top_decile_success']*100:.1f}% | {metrics['bottom_decile_success']*100:.1f}% | "
                f"{metrics['correlation']:.3f} | {metrics['top10_success_rate']*100:.1f}% | "
                f"{metrics['all_success_rate']*100:.1f}% | {metrics['top10_improvement']*100:+.1f}% |"
            )
    else:
        report_lines.append("## 準確率評估\n")
        report_lines.append("| Checkpoint | Avg Return | Max Return | Capture Rate | Hit Rate | Holding Days | Episodes |")
        report_lines.append("|------------|-----------|-----------|-------------|---------|------------:|--------:|")
        
        for step, metrics in results.items():
            step_label = f"{step/1e6:.1f}M" if isinstance(step, (int, float)) else step
            report_lines.append(
                f"| {step_label} | {metrics['avg_return']*100:.1f}% | {metrics['max_possible_return']*100:.1f}% | "
                f"{metrics['capture_rate']*100:.1f}% | {metrics['hit_rate']*100:.1f}% | {metrics['avg_holding_days']:.0f} | {metrics['total_episodes']} |"
            )

    
    report_content = "\n".join(report_lines)
    
    # 儲存報告
    report_path = output_dir / f"{agent_type}_agent" / "summary_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 儲存 CSV
    csv_path = output_dir / f"{agent_type}_agent" / "summary.csv"
    df = pd.DataFrame(results).T
    df.index.name = 'checkpoint'
    df.to_csv(csv_path)
    
    logger.info(f"報告已儲存: {report_path}")
    return report_content


def main():
    parser = argparse.ArgumentParser(description='Agent 獨立評估')
    parser.add_argument('--agent', type=str, choices=['buy', 'sell', 'both'], required=True,
                        help='要評估的 Agent')
    parser.add_argument('--steps', type=int, nargs='+', default=None,
                        help='指定 checkpoint 步數 (預設: 5M, 7.5M, 10M, 12.5M, 15M)')
    args = parser.parse_args()
    
    config = load_config()
    normalizer = DataNormalizer()
    feature_cols = normalizer.get_normalized_feature_columns()
    
    # 建立輸出目錄
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"outputs/evaluation/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"評估輸出目錄: {output_dir}")
    
    if args.agent in ['buy', 'both']:
        logger.info("=== Buy Agent 評估 ===")
        
        # 準備測試資料
        test_data = prepare_buy_test_data(config)
        
        # 找 checkpoint
        checkpoints = find_checkpoints('buy', args.steps)
        
        if not checkpoints:
            logger.error("找不到 Buy Agent checkpoints")
        else:
            results = {}
            for step, ckpt_path in checkpoints.items():
                logger.info(f"評估 checkpoint: {ckpt_path.name}")
                metrics = evaluate_buy_accuracy(str(ckpt_path), test_data, feature_cols)
                results[step] = metrics
                logger.info(f"  Conf: {metrics['conf_mean']:.3f}±{metrics['conf_std']:.3f}, Top-10 成功率: {metrics['top10_success_rate']*100:.1f}%, 相關係數: {metrics['correlation']:.3f}")
            
            report = generate_report(results, output_dir, 'buy')
            print("\n" + report)
    
    if args.agent in ['sell', 'both']:
        logger.info("=== Sell Agent 評估 ===")
        
        # 準備測試資料
        episodes = prepare_sell_test_data(config)
        
        # 找 checkpoint
        checkpoints = find_checkpoints('sell', args.steps)
        
        if not checkpoints:
            logger.error("找不到 Sell Agent checkpoints")
        else:
            results = {}
            for step, ckpt_path in checkpoints.items():
                logger.info(f"評估 checkpoint: {ckpt_path.name}")
                metrics = evaluate_sell_accuracy(str(ckpt_path), episodes, feature_cols)
                results[step] = metrics
                logger.info(f"  Avg Return: {metrics['avg_return']*100:.1f}%, Capture Rate: {metrics['capture_rate']*100:.1f}%")
            
            report = generate_report(results, output_dir, 'sell')
            print("\n" + report)
    
    logger.info(f"\n評估完成！結果已儲存至: {output_dir}")


if __name__ == '__main__':
    main()
