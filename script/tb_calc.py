import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dirs = {
    "RPPO": r"C:/Users/gr0664rx/RLProjects/GongguanRoundabout/tb_roundabout_highdemand/rppo",
    "PPO": r"C:/Users/gr0664rx/RLProjects/GongguanRoundabout/tb_roundabout_highdemand/ppo",
    "DDQN": r"C:/Users/gr0664rx/RLProjects/GongguanRoundabout/tb_roundabout_highdemand/ddqn",
    "Fixed_Cycle": r"C:/Users/gr0664rx/RLProjects/GongguanRoundabout/tb_roundabout_highdemand/fixed_cycle",
}

tags = [
    'custom/avg_queue_len',
    'custom/avg_speed',
    'custom/avg_step_reward',
    'custom/avg_travel_time',
    'custom/avg_phase_duration'
]

print("\n===== Last 5 Records Mean ± Std Summary =====\n")

for label, log_dir in log_dirs.items():
    print(f"[{label}]")
    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()

    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            values = [e.value for e in events]
            if len(values) >= 5:
                last5 = values[-5:]
                mean_val = pd.Series(last5).mean()
                std_val = pd.Series(last5).std()
                print(f"{tag.split('/')[-1]:<20}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"{tag.split('/')[-1]:<20}: [WARN] Less than 5 records available.")
        else:
            print(f"{tag.split('/')[-1]:<20}: [WARN] Tag not found.")
    print()

print("[✔] Calculation completed.")
