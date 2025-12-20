"""驗證資料平衡是否正常運作"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('d:/000-github-repositories/ptrl-v01')))

import pandas as pd
import yaml

from src.data import DataLoader, FeatureCalculator, DataNormalizer
from src.rules import DonchianChannel

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config('d:/000-github-repositories/ptrl-v01/config/default_config.yaml')
loader = DataLoader(config.get('data', {}))
normalizer = DataNormalizer()
donchian = DonchianChannel(period=20)

symbols = loader.load_symbols_list()[:50]  # 只測試前 50 支
index_data = loader.load_index()
success_threshold = 0.10
max_holding_days = 120

all_buy_signals = []

for symbol in symbols:
    df = loader.load_symbol(symbol)
    if df is None: continue
    try:
        cache_file = Path(f"d:/000-github-repositories/ptrl-v01/data/cache/{symbol}_features.pkl")
        df_normalized = pd.read_pickle(cache_file) if cache_file.exists() else None
        if df_normalized is None: continue
        
        buy_signals = donchian.generate_buy_signals(df_normalized)
        signal_indices = buy_signals[buy_signals == 1].index
        
        for signal_date in signal_indices:
            if signal_date not in df_normalized.index: continue
            signal_loc = df_normalized.index.get_loc(signal_date)
            buy_price = df_normalized.iloc[signal_loc]['Close']
            
            future_data = df_normalized.iloc[signal_loc+1 : signal_loc+1+max_holding_days]
            if len(future_data) < 5: continue
            
            max_return = (future_data['High'].max() - buy_price) / buy_price
            label = 2 if max_return >= success_threshold else 1
            
            feature_cols = normalizer.get_normalized_feature_columns()
            available_cols = [c for c in feature_cols if c in df_normalized.columns]
            
            row = df_normalized.iloc[signal_loc][available_cols].copy()
            row['label'] = label
            all_buy_signals.append(row)
    except: continue

buy_df = pd.DataFrame(all_buy_signals)
print(f"總樣本數: {len(buy_df)}")

# 檢查 label 分布
label_counts = buy_df['label'].value_counts()
print(f"Label 分布:\n{label_counts}")
print(f"成功率 (label=2): {label_counts.get(2, 0) / len(buy_df) * 100:.1f}%")

# 模擬 BuyEnv 的 label -> is_successful 轉換
buy_df['is_successful'] = buy_df['label'] == 2
print(f"\nis_successful 分布:\n{buy_df['is_successful'].value_counts()}")
