"""
Exploration and visualization of Wind Farm pretraining datasets.

Generates a multi-page PDF with:
    1. Layout overview: turbine positions + profiles polar plot
    2. Episode timeseries: ws, wd, yaw, power for a few episodes
    3. Distribution plots: histograms of power, ws, wd, yaw across all data
    4. Per-turbine statistics: mean power, power std, correlation matrix
    5. Episode metadata overview: scatter of (ws, wd, ti) conditions

Usage:
    python explore_dataset.py <path_to_h5_file>
    python explore_dataset.py ./pretrain_data/layout_test_layout.h5
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pathlib import Path


# Style
plt.rcParams.update({
    "figure.facecolor": "#f8f8f8",
    "axes.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2",
          "#4f46e5", "#be123c", "#15803d", "#7e22ce"]


def load_all_data(filepath: str) -> dict:
    """Load entire dataset into memory for plotting."""
    data = {"episodes": []}
    
    with h5py.File(filepath, "r") as f:
        data["layout_name"] = str(f.attrs["layout_name"])
        data["n_turbines"] = int(f.attrs["n_turbines"])
        data["rotor_diameter"] = float(f.attrs["rotor_diameter"])
        data["turbine_type"] = str(f.attrs["turbine_type"])
        data["positions"] = f["positions/xy"][:].astype(np.float32)
        
        if "profiles" in f:
            data["receptivity"] = f["profiles/receptivity"][:].astype(np.float32)
            data["influence"] = f["profiles/influence"][:].astype(np.float32)
        
        for ep_key in sorted(f["episodes"].keys()):
            ep = f[f"episodes/{ep_key}"]
            ep_data = {
                "key": ep_key,
                "n_steps": int(ep.attrs["n_steps"]),
                "mean_ws": float(ep.attrs["mean_ws"]),
                "mean_wd": float(ep.attrs["mean_wd"]),
                "mean_ti": float(ep.attrs["mean_ti"]),
                "ws": ep["ws"][:],
                "wd": ep["wd"][:],
                "yaw": ep["yaw"][:],
                "power": ep["power"][:],
                "actions": ep["actions"][:],
            }
            data["episodes"].append(ep_data)
    
    return data


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def plot_layout_overview(data: dict, fig):
    """Page 1: Turbine positions and profile polar plots."""
    n_turb = data["n_turbines"]
    pos = data["positions"]
    D = data["rotor_diameter"]
    has_profiles = "receptivity" in data

    if has_profiles:
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1, 1])
    else:
        gs = GridSpec(1, 1, figure=fig)

    # --- Turbine positions ---
    ax = fig.add_subplot(gs[0])
    ax.set_aspect("equal")
    
    # Normalize by rotor diameter for display
    x_norm = pos[:, 0] / D
    y_norm = pos[:, 1] / D
    
    ax.scatter(x_norm, y_norm, s=120, c=COLORS[:n_turb], edgecolors="black", linewidths=0.8, zorder=5)
    for i in range(n_turb):
        ax.annotate(f"T{i}", (x_norm[i], y_norm[i]), fontsize=8,
                    ha="center", va="bottom", xytext=(0, 8), textcoords="offset points",
                    fontweight="bold")
    
    # Add wind rose arrow
    ax.annotate("", xy=(x_norm.min() - 1.5, y_norm.mean()),
                xytext=(x_norm.min() - 3.5, y_norm.mean()),
                arrowprops=dict(arrowstyle="->", color="#666", lw=2))
    ax.text(x_norm.min() - 2.5, y_norm.mean() + 0.5, "Wind", ha="center", fontsize=8, color="#666")
    
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_title(f"Layout: {data['layout_name']}\n{n_turb} × {data['turbine_type']}, D={D:.1f}m")
    
    # --- Profiles ---
    if has_profiles:
        n_dirs = data["receptivity"].shape[1]
        angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
        
        # Receptivity
        ax_r = fig.add_subplot(gs[1], projection="polar")
        for i in range(n_turb):
            vals = data["receptivity"][i]
            vals_plot = np.append(vals, vals[0])  # close the loop
            angles_plot = np.append(angles, angles[0])
            ax_r.plot(angles_plot, vals_plot, color=COLORS[i % len(COLORS)], alpha=0.8, label=f"T{i}")
        ax_r.set_title("Receptivity\nProfiles", pad=15)
        ax_r.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.0))
        
        # Influence
        ax_i = fig.add_subplot(gs[2], projection="polar")
        for i in range(n_turb):
            vals = data["influence"][i]
            vals_plot = np.append(vals, vals[0])
            angles_plot = np.append(angles, angles[0])
            ax_i.plot(angles_plot, vals_plot, color=COLORS[i % len(COLORS)], alpha=0.8, label=f"T{i}")
        ax_i.set_title("Influence\nProfiles", pad=15)


def plot_episode_timeseries(data: dict, fig, ep_indices=None):
    """Page 2: Timeseries of ws, wd, yaw, power for selected episodes."""
    episodes = data["episodes"]
    n_turb = data["n_turbines"]
    
    if ep_indices is None:
        # Show up to 3 episodes
        ep_indices = list(range(min(3, len(episodes))))
    
    n_eps = len(ep_indices)
    features = [("ws", "Wind Speed [m/s]"), ("wd", "Wind Dir [°]"),
                ("yaw", "Yaw [°]"), ("power", "Power [kW]")]
    
    gs = GridSpec(len(features), n_eps, figure=fig, hspace=0.4, wspace=0.3)
    
    for col, ei in enumerate(ep_indices):
        ep = episodes[ei]
        n_steps = ep["n_steps"]
        t = np.arange(n_steps)
        
        for row, (feat, ylabel) in enumerate(features):
            ax = fig.add_subplot(gs[row, col])
            vals = ep[feat]  # (n_steps, n_turb)
            
            for ti in range(n_turb):
                ax.plot(t, vals[:, ti], color=COLORS[ti % len(COLORS)], 
                        alpha=0.8, linewidth=1.0, label=f"T{ti}" if col == 0 and row == 0 else None)
            
            if row == 0:
                ax.set_title(f"Ep {ei}: ws={ep['mean_ws']:.1f}, "
                            f"wd={ep['mean_wd']:.0f}°, ti={ep['mean_ti']:.3f}",
                            fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == len(features) - 1:
                ax.set_xlabel("Step")
            else:
                ax.set_xticklabels([])
    
    # Legend
    if n_turb <= 10:
        handles = [plt.Line2D([0], [0], color=COLORS[i % len(COLORS)], label=f"T{i}")
                   for i in range(n_turb)]
        fig.legend(handles=handles, loc="lower center", ncol=min(n_turb, 8), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))


def plot_distributions(data: dict, fig):
    """Page 3: Histograms of all features across all episodes."""
    # Concatenate all episode data
    all_ws = np.concatenate([ep["ws"].flatten() for ep in data["episodes"]])
    all_wd = np.concatenate([ep["wd"].flatten() for ep in data["episodes"]])
    all_yaw = np.concatenate([ep["yaw"].flatten() for ep in data["episodes"]])
    all_power = np.concatenate([ep["power"].flatten() for ep in data["episodes"]])
    all_actions = np.concatenate([ep["actions"].flatten() for ep in data["episodes"]])
    
    features = [
        ("Wind Speed [m/s]", all_ws, "#2563eb"),
        ("Wind Direction [°]", all_wd, "#dc2626"),
        ("Yaw Angle [°]", all_yaw, "#16a34a"),
        ("Power [kW]", all_power, "#9333ea"),
        ("Actions (yaw setpoint)", all_actions, "#ea580c"),
    ]
    
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    for idx, (label, vals, color) in enumerate(features):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"μ={vals.mean():.2f}, σ={vals.std():.2f}", fontsize=9)
    
    # Summary stats in last cell
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    n_eps = len(data["episodes"])
    total_steps = sum(ep["n_steps"] for ep in data["episodes"])
    summary = (
        f"Dataset Summary\n"
        f"{'─' * 25}\n"
        f"Layout: {data['layout_name']}\n"
        f"Turbines: {data['n_turbines']}\n"
        f"Episodes: {n_eps}\n"
        f"Total steps: {total_steps:,}\n"
        f"Steps/episode: {total_steps/n_eps:.0f}\n"
        f"{'─' * 25}\n"
        f"Power range: [{all_power.min():.0f}, {all_power.max():.0f}] kW\n"
        f"WS range: [{all_ws.min():.1f}, {all_ws.max():.1f}] m/s\n"
        f"Yaw range: [{all_yaw.min():.1f}, {all_yaw.max():.1f}]°"
    )
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#ccc"))


def plot_per_turbine_stats(data: dict, fig):
    """Page 4: Per-turbine power statistics and cross-turbine correlations."""
    n_turb = data["n_turbines"]
    
    # Collect per-turbine power across all episodes
    all_power = np.concatenate([ep["power"] for ep in data["episodes"]], axis=0)  # (total_steps, n_turb)
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    
    # Mean power per turbine
    ax = fig.add_subplot(gs[0, 0])
    means = all_power.mean(axis=0)
    stds = all_power.std(axis=0)
    x = np.arange(n_turb)
    ax.bar(x, means, yerr=stds, color=[COLORS[i % len(COLORS)] for i in range(n_turb)],
           edgecolor="white", linewidth=0.8, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n_turb)])
    ax.set_ylabel("Power [kW]")
    ax.set_title("Mean Power per Turbine (±1σ)")
    
    # Power distribution per turbine (violin/box)
    ax = fig.add_subplot(gs[0, 1])
    bp = ax.boxplot([all_power[:, i] for i in range(n_turb)],
                     labels=[f"T{i}" for i in range(n_turb)],
                     patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)
    ax.set_ylabel("Power [kW]")
    ax.set_title("Power Distribution per Turbine")
    
    # Cross-turbine power correlation
    ax = fig.add_subplot(gs[1, 0])
    if n_turb > 1:
        corr = np.corrcoef(all_power.T)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(n_turb))
        ax.set_yticks(range(n_turb))
        ax.set_xticklabels([f"T{i}" for i in range(n_turb)])
        ax.set_yticklabels([f"T{i}" for i in range(n_turb)])
        for i in range(n_turb):
            for j in range(n_turb):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr[i, j]) > 0.5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Power Correlation Matrix")
    
    # Power vs wind speed scatter (per turbine)
    ax = fig.add_subplot(gs[1, 1])
    all_ws = np.concatenate([ep["ws"] for ep in data["episodes"]], axis=0)
    for i in range(n_turb):
        ax.scatter(all_ws[:, i], all_power[:, i], s=2, alpha=0.3,
                   color=COLORS[i % len(COLORS)], label=f"T{i}")
    ax.set_xlabel("Wind Speed [m/s]")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Power vs Wind Speed")
    if n_turb <= 8:
        ax.legend(fontsize=7, markerscale=4)


def plot_episode_conditions(data: dict, fig):
    """Page 5: Episode-level metadata scatter and action statistics."""
    episodes = data["episodes"]
    
    ws_vals = [ep["mean_ws"] for ep in episodes]
    wd_vals = [ep["mean_wd"] for ep in episodes]
    ti_vals = [ep["mean_ti"] for ep in episodes]
    step_counts = [ep["n_steps"] for ep in episodes]
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    
    # WD vs WS scatter
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(ws_vals, wd_vals, c=ti_vals, cmap="viridis", s=40,
                    edgecolors="white", linewidths=0.5)
    fig.colorbar(sc, ax=ax, label="TI", shrink=0.8)
    ax.set_xlabel("Mean Wind Speed [m/s]")
    ax.set_ylabel("Mean Wind Direction [°]")
    ax.set_title("Episode Conditions")
    
    # WD histogram
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(wd_vals, bins=36, range=(0, 360), color="#2563eb", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Mean Wind Direction [°]")
    ax.set_ylabel("# Episodes")
    ax.set_title("Wind Direction Coverage")
    
    # Steps per episode
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(step_counts, bins=20, color="#16a34a", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Steps per Episode")
    ax.set_ylabel("# Episodes")
    ax.set_title(f"Episode Lengths (mean={np.mean(step_counts):.0f})")
    
    # Action distribution over time (one episode)
    ax = fig.add_subplot(gs[1, 1])
    ep = episodes[0]
    n_turb = data["n_turbines"]
    for i in range(n_turb):
        ax.plot(ep["actions"][:, i], color=COLORS[i % len(COLORS)], alpha=0.8, label=f"T{i}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Action (yaw setpoint)")
    ax.set_title(f"Actions — Episode 0")
    if n_turb <= 8:
        ax.legend(fontsize=7)


# =============================================================================
# MAIN
# =============================================================================

def explore_dataset(filepath: str, output_path: str = None):
    """Generate a multi-page PDF exploring the dataset."""
    filepath = Path(filepath)
    if output_path is None:
        output_path = filepath.with_suffix(".pdf")
    
    print(f"Loading {filepath}...")
    data = load_all_data(str(filepath))
    
    print(f"Generating plots for '{data['layout_name']}' "
          f"({data['n_turbines']} turbines, {len(data['episodes'])} episodes)...")
    
    with PdfPages(str(output_path)) as pdf:
        # Page 1: Layout overview
        fig = plt.figure(figsize=(14, 5))
        fig.suptitle("Layout Overview", fontsize=14, fontweight="bold", y=1.02)
        plot_layout_overview(data, fig)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        # Page 2: Timeseries
        n_show = min(3, len(data["episodes"]))
        fig = plt.figure(figsize=(5 * n_show, 10))
        fig.suptitle("Episode Timeseries", fontsize=14, fontweight="bold", y=1.01)
        plot_episode_timeseries(data, fig)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        # Page 3: Distributions
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle("Feature Distributions (all episodes)", fontsize=14, fontweight="bold", y=1.01)
        plot_distributions(data, fig)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        # Page 4: Per-turbine stats
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle("Per-Turbine Statistics", fontsize=14, fontweight="bold", y=1.01)
        plot_per_turbine_stats(data, fig)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        # Page 5: Episode conditions
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle("Episode Conditions & Actions", fontsize=14, fontweight="bold", y=1.01)
        plot_episode_conditions(data, fig)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    
    print(f"Saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_dataset.py <path_to_h5_file> [output.pdf]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    explore_dataset(filepath, output)