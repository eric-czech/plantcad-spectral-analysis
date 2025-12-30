"""Analyze and visualize aggregated experiment results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Experiment groupings
EXPERIMENT_GROUPS = {
    "Text pretrained": ["v29", "v30"],
    "DNA random init": ["v12", "v14", "v26"],
    "DNA pretrained": ["v13", "v17", "v18", "v22", "v28"],
}

# Markers for each experiment version
MARKERS = {
    "v12": "o", "v13": "s", "v14": "^",
    "v17": "p", "v18": "h",
    "v22": "D", "v26": "v", "v28": "X",
    "v29": "P", "v30": "*",
}

# Colors for each experiment
EXPERIMENT_COLORS = {
    "v12": "#bcbd22",
    "v13": "#1f77b4",
    "v14": "#ff7f0e",
    "v17": "#2ca02c",
    "v18": "#d62728",
    "v22": "#7f7f7f",
    "v26": "#17becf",
    "v28": "#9467bd",
    "v29": "#8c564b",
    "v30": "#e377c2",
}


def get_version_to_group() -> dict[str, str]:
    """Map experiment version to its group name."""
    return {v: group for group, versions in EXPERIMENT_GROUPS.items() for v in versions}


def cmd_visualize_multimodal_eigenspectra(args: argparse.Namespace) -> None:
    """Visualize eigenspectra across modalities and models."""
    base_dir = Path(__file__).parent.parent
    spectral_path = base_dir / "results" / "spectral_curves.parquet"
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
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    group_names = list(EXPERIMENT_GROUPS.keys())
    
    for idx, group_name in enumerate(group_names):
        ax = axes[idx]
        group_versions = EXPERIMENT_GROUPS[group_name]
        
        # Plot each experiment in this group
        for version in group_versions:
            exp_df = df[df["version"] == version].sort_values("rank")
            label = f"{version}: {exp_df['title'].iloc[0]}"
            ax.plot(
                exp_df["log10_rank_pct"],
                exp_df["log10_eigenvalue_normalized"],
                marker=MARKERS[version],
                color=EXPERIMENT_COLORS[version],
                label=label,
                markersize=3,
                linewidth=1,
                alpha=0.8,
            )
        
        # Set labels and title
        ax.set_xlabel("Model Width %")
        if idx == 0:
            ax.set_ylabel("log₁₀(Normalized Eigenvalue)")
        else:
            ax.set_yticklabels([])
        ax.set_title(group_name)
        
        # Set x-axis ticks as percentages
        pct_ticks = [0.1, 0.25, 1, 5, 10, 20, 50, 100]
        ax.set_xticks([np.log10(p) for p in pct_ticks])
        ax.set_xticklabels([f"{p}%" for p in pct_ticks])
        
        # Set fixed axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Legend in the plot area
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)

    # Export
    output_dir = base_dir / "results" / "figures"
    output_dir.mkdir(exist_ok=True)
    output_base = output_dir / "multimodal_eigenspectra"
    fig.savefig(f"{output_base}.pdf", dpi=192, bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=192, bbox_inches="tight")
    plt.close(fig)

    # Export CSV
    export_df = df[["version", "title", "group", "rank", "rank_normalized", "rank_pct", "log10_rank_pct", "eigenvalue", "eigenvalue_normalized", "log10_eigenvalue", "log10_eigenvalue_normalized", "n_samples"]].copy()
    export_df.to_csv(f"{output_base}.csv", index=False)

    print(f"Exported: {output_base}.pdf, {output_base}.png, {output_base}.csv")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "visualize_multimodal_eigenspectra",
        help="Visualize eigenspectra across modalities",
    )

    args = parser.parse_args()

    if args.command == "visualize_multimodal_eigenspectra":
        cmd_visualize_multimodal_eigenspectra(args)


if __name__ == "__main__":
    main()

