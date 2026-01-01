"""Analyze and visualize aggregated experiment results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Experiment groupings
EXPERIMENT_GROUPS = {
    "Functional DNA models": ["v13", "v41", "v28", "v17"],
    "Functional DNA models (random weights)": ["v14", "v26"],
    "Whole-genome DNA models": ["v33", "v34", "v37"],
    "Text models": ["v30", "v35", "v36"],
}

# Markers for each experiment version
MARKERS = {
    "v13": "s", "v14": "^", "v17": "p", "v26": "v", "v28": "X",
    "v30": "*", "v31": "D", "v33": "o", "v34": "h", "v35": "P",
    "v36": "<", "v37": ">", "v41": "8",
}

# Colors for each experiment
EXPERIMENT_COLORS = {
    "v13": "#1f77b4",
    "v14": "#ff7f0e",
    "v17": "#2ca02c",
    "v26": "#17becf",
    "v28": "#9467bd",
    "v30": "#e377c2",
    "v31": "#d62728",
    "v33": "#bcbd22",
    "v34": "#8c564b",
    "v35": "#7f7f7f",
    "v36": "#e377c2",
    "v37": "#17becf",
    "v41": "#ff7f0e",
}


def get_version_to_group() -> dict[str, str]:
    """Map experiment version to its group name."""
    return {v: group for group, versions in EXPERIMENT_GROUPS.items() for v in versions}


def plot_eigenspectra_on_axis(ax, df, versions, x_min, x_max, y_min, y_max):
    """Plot eigenspectra for given versions on an axis with shared scales."""
    for version in versions:
        exp_df = df[df["version"] == version].sort_values("rank")
        if len(exp_df) == 0:
            print(f"WARNING: No data found for version {version}")
            continue
        label = f"{version}: {exp_df['title'].iloc[0]}"
        ax.plot(
            exp_df["log10_rank_pct"],
            exp_df["log10_eigenvalue_normalized"],
            marker=MARKERS[version],
            color=EXPERIMENT_COLORS[version],
            label=label,
            markersize=3,
            linewidth=1,
            alpha=0.6,
        )
    
    # Set x-axis ticks as percentages
    pct_ticks = [0.1, 0.25, 1, 5, 10, 20, 50, 100]
    ax.set_xticks([np.log10(p) for p in pct_ticks])
    ax.set_xticklabels([f"{p}%" for p in pct_ticks])
    
    # Set fixed axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)


def save_outputs(fig, df, output_base):
    """Save figure and CSV outputs."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_base}.pdf", dpi=192, bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=192, bbox_inches="tight")
    plt.close(fig)
    
    # Export CSV
    export_df = df[["version", "title", "group", "rank", "rank_normalized", "rank_pct", "log10_rank_pct", "eigenvalue", "eigenvalue_normalized", "log10_eigenvalue", "log10_eigenvalue_normalized", "n_samples"]].copy()
    export_df.to_csv(f"{output_base}.csv", index=False)
    
    print(f"Exported: {output_base}.pdf, {output_base}.png, {output_base}.csv")


def cmd_visualize_multimodal_eigenspectra(args: argparse.Namespace) -> None:
    """Visualize eigenspectra across modalities and models."""
    base_dir = Path(__file__).parent.parent
    spectral_path = base_dir / "results" / "datasets" / "spectral_curves.parquet"
    if not spectral_path.exists():
        raise FileNotFoundError(f"Missing {spectral_path}")

    df = pd.read_parquet(spectral_path)

    # Filter to target experiments
    all_versions = [v for versions in EXPERIMENT_GROUPS.values() for v in versions]
    df = df[df["version"].isin(all_versions)]
    missing = set(all_versions) - set(df["version"].unique())
    if missing:
        raise ValueError(f"Missing experiments in data: {missing}")

    # For each experiment, keep only max n_samples
    df = df.loc[df.groupby("version")["n_samples"].idxmax().values]
    df = df.merge(
        df.groupby("version")["n_samples"].max().reset_index(),
        on=["version", "n_samples"],
    )
    # Re-load to get all rows for max n_samples
    df = pd.read_parquet(spectral_path)
    df = df[df["version"].isin(all_versions)]
    max_samples = df.groupby("version")["n_samples"].max()
    df = df[df.apply(lambda r: r["n_samples"] == max_samples[r["version"]], axis=1)]

    # Add group and normalized rank/eigenvalue (divide by max to get values in (0, 1])
    version_to_group = get_version_to_group()
    df["group"] = df["version"].map(version_to_group)
    df["rank_normalized"] = df.groupby("version")["rank"].transform(lambda x: x / x.max())
    # Normalize by max eigenvalue cf. https://openreview.net/pdf?id=vXxardq6db (A.4)
    df["eigenvalue_normalized"] = df.groupby("version")["eigenvalue"].transform(lambda x: x / x.max())
    df["rank_pct"] = df["rank_normalized"] * 100
    df["log10_eigenvalue"] = np.log10(df["eigenvalue"])
    df["log10_eigenvalue_normalized"] = np.log10(df["eigenvalue_normalized"])
    df["log10_rank_pct"] = np.log10(df["rank_pct"])

    # Calculate global axis limits with margins
    x_min, x_max = df["log10_rank_pct"].min(), df["log10_rank_pct"].max()
    y_min, y_max = df["log10_eigenvalue_normalized"].min(), df["log10_eigenvalue_normalized"].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    # Create figure with 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    
    group_names = list(EXPERIMENT_GROUPS.keys())
    
    for idx, group_name in enumerate(group_names):
        ax = axes[idx]
        group_versions = EXPERIMENT_GROUPS[group_name]
        
        # Plot eigenspectra using helper function
        plot_eigenspectra_on_axis(ax, df, group_versions, x_min, x_max, y_min, y_max)
        
        # Set labels and title
        # X-axis labels only on bottom row
        if idx >= 2:
            ax.set_xlabel("Model Width %")
        
        # Y-axis labels only on left column
        if idx % 2 == 0:
            ax.set_ylabel("log₁₀(Normalized Eigenvalue)")
        
        ax.set_title(group_name)
        
        # Legend in the plot area
        ax.legend(fontsize=7, loc="lower left")
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    # Export
    output_dir = base_dir / "results" / "figures"
    output_base = output_dir / "multimodal_eigenspectra"
    save_outputs(fig, df, output_base)


def cmd_visualize_performance_overlay(args: argparse.Namespace) -> None:
    """Visualize eigenspectra with F1 performance overlay for Functional DNA models."""
    base_dir = Path(__file__).parent.parent
    spectral_path = base_dir / "results" / "datasets" / "spectral_curves.parquet"
    performance_path = base_dir / "results" / "datasets" / "performance_curves.parquet"
    
    if not spectral_path.exists():
        raise FileNotFoundError(f"Missing {spectral_path}")
    if not performance_path.exists():
        raise FileNotFoundError(f"Missing {performance_path}")

    # Load data
    spectral_df = pd.read_parquet(spectral_path)
    perf_df = pd.read_parquet(performance_path)
    
    # Filter to Functional DNA models only
    target_group = "Functional DNA models"
    target_versions = EXPERIMENT_GROUPS[target_group]
    
    spectral_df = spectral_df[spectral_df["version"].isin(target_versions)]
    perf_df = perf_df[perf_df["version"].isin(target_versions)]
    
    # Filter to species target and max n_samples for spectral data
    perf_df = perf_df[perf_df["target"] == "species"]
    max_samples = spectral_df.groupby("version")["n_samples"].max()
    spectral_df = spectral_df[spectral_df.apply(lambda r: r["n_samples"] == max_samples[r["version"]], axis=1)]
    
    # Add normalized columns to spectral data
    version_to_group = get_version_to_group()
    spectral_df["group"] = spectral_df["version"].map(version_to_group)
    spectral_df["rank_normalized"] = spectral_df.groupby("version")["rank"].transform(lambda x: x / x.max())
    spectral_df["eigenvalue_normalized"] = spectral_df.groupby("version")["eigenvalue"].transform(lambda x: x / x.max())
    spectral_df["rank_pct"] = spectral_df["rank_normalized"] * 100
    spectral_df["log10_eigenvalue"] = np.log10(spectral_df["eigenvalue"])
    spectral_df["log10_eigenvalue_normalized"] = np.log10(spectral_df["eigenvalue_normalized"])
    spectral_df["log10_rank_pct"] = np.log10(spectral_df["rank_pct"])
    
    # Calculate global axis limits with margins (same as multimodal plot)
    x_min, x_max = spectral_df["log10_rank_pct"].min(), spectral_df["log10_rank_pct"].max()
    y_min, y_max = spectral_df["log10_eigenvalue_normalized"].min(), spectral_df["log10_eigenvalue_normalized"].max()
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    # Create figure with one subplot per experiment
    n_exps = len(target_versions)
    fig, axes = plt.subplots(1, n_exps, figsize=(4 * n_exps, 4))
    if n_exps == 1:
        axes = [axes]
    
    # Collect handles and labels for combined legend
    all_handles = []
    all_labels = []
    
    for idx, version in enumerate(target_versions):
        ax1 = axes[idx]
        
        # Plot eigenspectra on primary axis
        plot_eigenspectra_on_axis(ax1, spectral_df, [version], x_min, x_max, y_min, y_max)
        
        # Create second y-axis for F1 scores
        ax2 = ax1.twinx()
        
        # Get performance data for this version
        perf_version = perf_df[perf_df["version"] == version].sort_values("n_components")
        
        if len(perf_version) == 0:
            print(f"WARNING: No performance data for version {version}")
        else:
            # Match n_components to rank_pct from spectral data
            perf_version = perf_version.merge(
                spectral_df[spectral_df["version"] == version][["rank", "log10_rank_pct"]],
                left_on="n_components",
                right_on="rank",
                how="inner"
            )
            
            ax2.plot(
                perf_version["log10_rank_pct"],
                perf_version["f1_macro"],
                marker="o",
                color="black",
                label="F1 Species",
                markersize=4,
                linewidth=1.5,
                alpha=0.8,
            )
            # Only show F1 y-axis label on last plot
            if idx == n_exps - 1:
                ax2.set_ylabel("F1 Species Score", color="black")
            ax2.tick_params(axis="y", labelcolor="black")
        
        # Set labels and title
        ax1.set_xlabel("Model Width %")
        # Only show eigenvalue y-axis label on first plot
        if idx == 0:
            ax1.set_ylabel("log₁₀(Normalized Eigenvalue)")
        
        exp_title = spectral_df[spectral_df["version"] == version]["title"].iloc[0] if len(spectral_df[spectral_df["version"] == version]) > 0 else version
        ax1.set_title(f"{version}: {exp_title}")
        
        # Collect legend entries from first plot only
        if idx == 0:
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            all_handles.extend(h1 + h2)
            all_labels.extend(l1 + l2)
    
    # Add single legend at bottom
    fig.legend(all_handles, all_labels, loc="lower center", ncol=len(all_labels), bbox_to_anchor=(0.5, -0.05), fontsize=8)
    
    plt.tight_layout()
    
    # Export
    output_dir = base_dir / "results" / "figures"
    output_base = output_dir / "performance_overlay"
    save_outputs(fig, spectral_df, output_base)


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run all visualization commands."""
    print("Running all visualizations...")
    print("\n" + "="*60)
    print("1. Visualizing multimodal eigenspectra...")
    print("="*60)
    cmd_visualize_multimodal_eigenspectra(args)
    
    print("\n" + "="*60)
    print("2. Visualizing performance overlay...")
    print("="*60)
    cmd_visualize_performance_overlay(args)
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "visualize_multimodal_eigenspectra",
        help="Visualize eigenspectra across modalities",
    )
    
    subparsers.add_parser(
        "visualize_performance_overlay",
        help="Visualize eigenspectra with F1 performance overlay for Functional DNA models",
    )
    
    subparsers.add_parser(
        "run_all",
        help="Run all visualization commands",
    )

    args = parser.parse_args()

    if args.command == "visualize_multimodal_eigenspectra":
        cmd_visualize_multimodal_eigenspectra(args)
    elif args.command == "visualize_performance_overlay":
        cmd_visualize_performance_overlay(args)
    elif args.command == "run_all":
        cmd_run_all(args)


if __name__ == "__main__":
    main()

