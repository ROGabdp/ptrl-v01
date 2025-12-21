import os
import shutil
import re
import json

CHECKPOINT_DIR = r"d:\000-github-repositories\ptrl-v01\models\checkpoints\buy_agent"
ARCHIVE_DIR = os.path.join(CHECKPOINT_DIR, "tainted_flatline")
CUTOFF_STEPS = 8660000

print(f"Cleaning up checkpoints > {CUTOFF_STEPS} steps in {CHECKPOINT_DIR}...")

os.makedirs(ARCHIVE_DIR, exist_ok=True)

files = os.listdir(CHECKPOINT_DIR)
moved_count = 0
max_step = 0

for f in files:
    if f.endswith(".zip") and f.startswith("buy_agent_"):
        try:
            match = re.search(r"buy_agent_(\d+)_steps.zip", f)
            if match:
                steps = int(match.group(1))
                if steps > CUTOFF_STEPS:
                    src = os.path.join(CHECKPOINT_DIR, f)
                    dst = os.path.join(ARCHIVE_DIR, f)
                    shutil.move(src, dst)
                    moved_count += 1
                    # print(f"Moved {f} to archive.")
                else:
                    max_step = max(max_step, steps)
        except Exception as e:
            print(f"Skipping {f}: {e}")

print(f"Total moved: {moved_count}")
print(f"New max step: {max_step}")

# Update state json
json_path = os.path.join(CHECKPOINT_DIR, "training_state.json")
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    current_total = data.get('total_timesteps', 0)
    print(f"Current recorded total_timesteps: {current_total}")
    
    if current_total > max_step:
        print(f"Updating training_state.json: {current_total} -> {max_step}")
        data['total_timesteps'] = max_step
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        print("training_state.json appears consistent (<= max_step).")
else:
    print("training_state.json not found.")
