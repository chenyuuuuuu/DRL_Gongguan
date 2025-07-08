import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from functools import reduce

log_dirs = {
    "RPPO": r"C:/Users/gr0664rx/RLProjects/GongguanRoundabout/tb_roundabout_baseline/rppo",
    "PPO": r"C:/Users/gr0664rx/RLProjects/GongguanRoundabout/tb_roundabout_baseline/ppo"
}
tags = ['custom/avg_step_reward', 'train/explained_variance']
exported_data = {}

for label, log_dir in log_dirs.items():
    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    dataframes = []
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            df = pd.DataFrame({
                'step': [e.step for e in events],
                tag: [e.value for e in events]
            })
            dataframes.append(df)
        else:
            print(f"[WARN] Tag {tag} not found in {log_dir}")
    if dataframes:
        df_merged = reduce(lambda left, right: pd.merge(left, right, on='step', how='outer'), dataframes)
        df_merged = df_merged.sort_values('step').reset_index(drop=True)
        df_merged.to_csv(f"{label}_export.csv", index=False)
        exported_data[label] = df_merged
    else:
        print(f"[WARN] No valid tags found in {log_dir}")

def ema_smoothing(y, alpha=0.6):
    return pd.Series(y).ewm(alpha=alpha).mean().to_numpy()

# Plot avg_step_reward
plt.figure(figsize=(3.7, 2.8))  # suitable size
for label, df in exported_data.items():
    if 'custom/avg_step_reward' in df.columns:
        plot_df = df[['step', 'custom/avg_step_reward']].dropna()
        x = plot_df['step'].to_numpy()
        y_raw = plot_df['custom/avg_step_reward'].to_numpy()
        y_smooth = ema_smoothing(y_raw, alpha=0.6)
        y_upper = np.maximum(y_smooth, y_raw)
        y_lower = np.minimum(y_smooth, y_raw)
        plt.plot(x, y_smooth, label=label, linewidth=1)
        plt.fill_between(x, y_lower, y_upper, alpha=0.15)
plt.xlabel('Timesteps', fontsize=12)
plt.ylabel('Avg. Step Reward', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100000))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
plt.tight_layout()
plt.savefig("avg_step_reward_curve.pdf")
plt.close()

# Plot explained_variance
plt.figure(figsize=(3.7, 2.8))  # suitable size
for label, df in exported_data.items():
    if 'train/explained_variance' in df.columns:
        plot_df = df[['step', 'train/explained_variance']].dropna()
        x = plot_df['step'].to_numpy()
        y_raw = plot_df['train/explained_variance'].to_numpy()
        y_smooth = ema_smoothing(y_raw, alpha=0.6)
        y_upper = np.maximum(y_smooth, y_raw)
        y_lower = np.minimum(y_smooth, y_raw)
        plt.plot(x, y_smooth, label=label, linewidth=1)
        plt.fill_between(x, y_lower, y_upper, alpha=0.15)
plt.xlabel('Timesteps', fontsize=12)
plt.ylabel('Explained Variance', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100000))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
plt.tight_layout()
plt.savefig("explained_variance_curve.pdf")
plt.close()

print("[✔] Plots regenerated with x/y axis labels at 12pt for balanced Overleaf readability.")
