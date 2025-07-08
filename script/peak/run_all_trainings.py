"""
run_all_trainings.py

Run training scripts sequentially with total time tracking, safety delays, and beep alerts.
"""

import subprocess
import time
import sys
import os

# List your training scripts here
scripts = [
    "train_rppo_peak.py",
    "train_ppo_peak.py",
    "train_ddqn_peak.py"
]

def run_script(script_name):
    print(f"\n[🚀] Starting: {script_name}\n{'='*60}\n")
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"\n[✔] Completed: {script_name}\n{'='*60}\n")
    except subprocess.CalledProcessError as e:
        print(f"[❌] {script_name} failed with return code {e.returncode}. Exiting.")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n[❌] Interrupted during {script_name}. Exiting safely.\n")
        sys.exit(1)

def beep():
    try:
        # Windows beep
        import winsound
        winsound.Beep(1000, 500)  # 1000 Hz for 500 ms
    except ImportError:
        # Fallback beep
        print('\a', end='', flush=True)

if __name__ == "__main__":
    start_time = time.time()
    try:
        for script in scripts:
            run_script(script)
            time.sleep(5)  # Safety delay to prevent SUMO port conflicts
    finally:
        elapsed = time.time() - start_time
        print(f"\n🎉 All training jobs completed. Total time: {elapsed/3600:.2f} hours.\n")
        beep()
        beep()
