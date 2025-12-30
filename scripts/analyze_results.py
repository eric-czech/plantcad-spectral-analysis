"""Analyze and visualize aggregated experiment results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Experiment groupings
EXPERIMENT_GROUPS = {
    "PlantCAD2-S + Genic DNA": ["v13", "v14"],
    "Marin + Genic DNA": ["v17", "v18"],
    "Marin + Promoter DNA": ["v28"],
    "Qwen + Text": ["v29", "v30"],
}

# Markers for each experiment version
MARKERS = {
    "v13": "s", "v14": "^",
    "v17": "p", "v18": "h",
    "v28": "X",
    "v29": "P", "v30": "*",
}

# Colors for each group
GROUP_COLORS = {
    "PlantCAD2-S + Genic DNA": "#1f77b4",
    "Marin + Genic DNA": "#ff7f0e",
    "Marin + Promoter DNA": "#2ca02c",
    "Qwen + Text": "#d62728",
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

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each experiment
    for version in all_versions:
        exp_df = df[df["version"] == version].sort_values("rank")
        group = version_to_group[version]
        label = f"{version}: {exp_df['title'].iloc[0]}"
        ax.plot(
            exp_df["log10_rank_pct"],
            exp_df["log10_eigenvalue_normalized"],
            marker=MARKERS[version],
            color=GROUP_COLORS[group],
            label=label,
            markersize=3,
            linewidth=1,
            alpha=0.8,
        )

    ax.set_xlabel("Model Width %")
    ax.set_ylabel("log₁₀(Normalized Eigenvalue)")
    ax.set_title("Multimodal Eigenspectra Comparison")
    
    # Set x-axis ticks as percentages
    pct_ticks = [0.1, 0.5, 1, 5, 10, 25, 50, 100]
    ax.set_xticks([np.log10(p) for p in pct_ticks])
    ax.set_xticklabels([f"{p}%" for p in pct_ticks])
    
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

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

