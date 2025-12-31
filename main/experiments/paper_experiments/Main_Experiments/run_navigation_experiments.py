



# one flowchart for complete pipeline , one scene screenshot and trajectory plot








# run_navigation_experiments.py - SIMPLE ONE-BY-ONE RUNNER
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any
import time
import sys


print("[DEBUG] Current script:", Path(__file__).resolve())

# Add Main_Experiments/ dir to path (where policies/ lives)
script_dir = Path(__file__).parent  # Main_Experiments/
sys.path.insert(0, str(script_dir))

print("[DEBUG] Added to path:", script_dir)

# Policy imports (CORRECTED - using main functions from each policy) 
# from experiments.paper_experiments.Main_Experiments.policies.straight_turn_policy import run_straight_turn_policy

from experiments.paper_experiments.Main_Experiments.policies.ours_full.ours_full_policy import run_ours_full
# from experiments.paper_experiments.Main_Experiments.policies.ours_no_ijepa.ours_no_ijepa_policy import run_ours_no_ijepa
# from experiments.paper_experiments.Main_Experiments.policies.ours_no_tinyllama.ours_no_tinyllama_policy import run_ours_no_tinyllama

print("[DEBUG] ✅ ALL policies imported successfully!")


print("[DEBUG] Imported all policy entrypoints")

policies = {
    # "straight_turn": run_straight_turn_policy,
    "ours_full": run_ours_full,
    # "ours_no_ijepa": run_ours_no_ijepa,
    # "ours_no_tinyllama": run_ours_no_tinyllama
}

print(f"[DEBUG] Registered policies: {list(policies.keys())}")

def load_episodes() -> list:
    """Load evaluation_episodes.json from Main_Experiments/"""
    print("[DEBUG] Loading evaluation_episodes.json")
    
    # FULL PATH to where the JSON actually lives
    json_path = Path(__file__).parent / 'evaluation_episodes.json'
    print(f"[DEBUG] Looking for JSON at: {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found at {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    episodes = data['episodes']
    print(f"[DEBUG] Loaded {len(episodes)} episodes")
    return episodes


def compute_spl(trajectory: list, goal_pos: list) -> float:
    """SPL = Success * (straight / path_length)"""
    print(f"[DEBUG] Computing SPL | trajectory_len={len(trajectory)} | goal_pos={goal_pos}")

    if len(trajectory) < 2:
        print("[DEBUG] Trajectory too short for SPL, returning 0.0")
        return 0.0

    straight_dist = np.linalg.norm(np.array(trajectory[0]) - np.array(goal_pos))
    path_length = sum(np.linalg.norm(np.diff(trajectory, axis=0)))

    spl = straight_dist / path_length if path_length > 0 else 0.0
    print(f"[DEBUG] SPL computed | straight={straight_dist:.3f} | path={path_length:.3f} | spl={spl:.3f}")
    return spl

def log_experiment(policy_name: str, episode: dict, metrics: Dict[str, Any]):
    """Append to navigation_experiments.csv"""
    print(f"[DEBUG] Logging results | policy={policy_name} | episode={episode['episode_id']}")

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    df = pd.DataFrame([{
        'experiment_id': episode['episode_id'],
        'condition': policy_name,
        'scene': episode['scene_path'],
        'goal_object': episode['goal']['object_category'],
        'success': metrics.get('success', 0),
        'spl': metrics.get('spl', 0.0),
        'path_length': metrics.get('path_length', 0.0),
        'collisions': metrics.get('collisions', 0),
        'time_taken': metrics.get('time_taken', 0.0),
        'final_distance': metrics.get('final_distance', 999),
        'failure_mode': metrics.get('failure_mode', 'none'),
        'notes': ''
    }])

    csv_path = log_dir / 'navigation_experiments.csv'
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print("[DEBUG] Appended row to existing CSV")
    else:
        df.to_csv(csv_path, index=False)
        print("[DEBUG] Created new CSV with header")

    print(f"✅ Logged {policy_name}: success={metrics.get('success', 0)}, SPL={metrics.get('spl', 0.0):.3f}")

def save_failure(policy_name: str, episode: dict, trajectory: list, metrics: Dict[str, Any]):
    """Save failure trajectory to logs/failures/"""
    if metrics.get('success', 0) == 1:
        print("[DEBUG] Episode succeeded, not saving failure")
        return

    print(f"[DEBUG] Saving failure data | policy={policy_name} | episode={episode['episode_id']}")

    failures_dir = Path('logs/failures')
    failures_dir.mkdir(exist_ok=True)

    filename = f"{episode['episode_id']}_{policy_name}_failure.npy"
    np.save(failures_dir / filename, {
        'trajectory': np.array(trajectory),
        'final_distance': metrics['final_distance'],
        'failure_mode': metrics['failure_mode'],
        'scene': episode['scene_path']
    })

    print(f"💾 Saved failure file: {filename}")

def main():
    print("[DEBUG] Starting SINGLE episode run")

    episodes = load_episodes()
    single_episode = episodes[5]
    print(f"🎯 SINGLE EPISODE: {single_episode['episode_id']}")

    print(f"🏃 RUNNING ALL POLICIES: {list(policies.keys())}")

    for policy_name, policy_func in policies.items():
        print("\n" + "=" * 60)
        print(f"🚀 RUNNING POLICY: {policy_name}")
        print("=" * 60)

        start_time = time.time()

        try:
            metrics = policy_func(single_episode)
            metrics['time_taken'] = time.time() - start_time

            log_experiment(policy_name, single_episode, metrics)

            if 'trajectory' in metrics:
                save_failure(policy_name, single_episode, metrics['trajectory'], metrics)

            print(
                f"✅ {policy_name} DONE | "
                f"success={metrics['success']} | "
                f"SPL={metrics['spl']:.3f}"
            )

        except Exception as e:
            print(f"❌ {policy_name} FAILED | error={e}")


if __name__ == "__main__":
    print("[DEBUG] __main__ entry point")
    main()
