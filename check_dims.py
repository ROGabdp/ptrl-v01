
import pandas as pd
import sys
from pathlib import Path
import yaml

project_root = Path('d:/000-github-repositories/ptrl-v01')
sys.path.insert(0, str(project_root))

from src.data import DataNormalizer

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Load a random cache file
cache_dir = Path('d:/000-github-repositories/ptrl-v01/data/cache')
cache_files = list(cache_dir.glob('*_features.pkl'))
if not cache_files:
    print("No cache files found!")
    sys.exit(1)

sample_file = cache_files[0]
df = pd.read_pickle(sample_file)

print(f"Loaded {sample_file.name}")
print(f"Total Columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")

normalizer = DataNormalizer()
norm_cols = normalizer.get_normalized_feature_columns()
print(f"Normalizer expects {len(norm_cols)} normalized columns.")

# Check intersection
overlap = [c for c in df.columns if c in norm_cols]
print(f"Overlap with expected normalized columns: {len(overlap)}")
