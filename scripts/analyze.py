"""Analyze and visualize aggregated experiment results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_GROUPS = {
    "Functional DNA models": ["v13", "v41", "v28", "v17"],
    "Functional DNA models (random weights)": ["v14", "v26"],
    "Whole-genome DNA models": ["v33", "v34", "v37"],
    "Text models": ["v30", "v35", "v36"],
}

MARKERS = {
    "v13": "s", "v14": "^", "v17": "p", "v26": "v", "v28": "X",
    "v30": "*", "v31": "D", "v33": "o", "v34": "h", "v35": "P",
    "v36": "<", "v37": ">", "v41": "8",
}

EXPERIMENT_COLORS = {
    "v13": "#1f77b4", "v14": "#ff7f0e", "v17": "#2ca02c", "v26": "#17becf",
    "v28": "#9467bd", "v30": "#e377c2", "v31": "#d62728", "v33": "#bcbd22",
    "v34": "#8c564b", "v35": "#7f7f7f", "v36": "#e377c2", "v37": "#17becf",
    "v41": "#ff7f0e",
}

TASK_COLORS = {
    "species": "#1f77b4", "membership": "#ff7f0e", "gc_content": "#2ca02c",
    "repeat_fraction": "#d62728", "kmer_entropy_3": "#9467bd",
}

TASK_MARKERS = {
    "species": "o", "membership": "s", "gc_content": "^",
    "repeat_fraction": "D", "kmer_entropy_3": "P",
}

TASK_LABELS = {
    "species": "Species", "membership": "Membership", "gc_content": "GC Content",
    "repeat_fraction": "Repeat Fraction", "kmer_entropy_3": "K-mer (k=3)",
}

METRIC_LABELS = {
    "accuracy": "Accuracy", "f1_macro": "F1 (Macro)",
    "roc_auc_macro": "ROC AUC (Macro)", "auprc_macro": "AUPRC (Macro)",
}


# =============================================================================
# Data Loading Helpers
# =============================================================================

def get_base_dir() -> Path:
    return Path(__file__).parent.parent


def load_spectral_data(versions: list[str] | None = None) -> pd.DataFrame:
    """Load and prepare spectral data with normalized columns."""
    path = get_base_dir() / "results" / "datasets" / "spectral_curves.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    
    df = pd.read_parquet(path)
    if versions:
        df = df[df["version"].isin(versions)]
    
    # Keep only max n_samples per version
    max_samples = df.groupby("version")["n_samples"].max()
    df = df[df.apply(lambda r: r["n_samples"] == max_samples[r["version"]], axis=1)]
    
    # Add normalized columns
    version_to_group = {v: g for g, vs in EXPERIMENT_GROUPS.items() for v in vs}
    df["group"] = df["version"].map(version_to_group)
    df["rank_normalized"] = df.groupby("version")["rank"].transform(lambda x: x / x.max())
    df["eigenvalue_normalized"] = df.groupby("version")["eigenvalue"].transform(lambda x: x / x.max())
    df["rank_pct"] = df["rank_normalized"] * 100
    df["log10_eigenvalue"] = np.log10(df["eigenvalue"])
    df["log10_eigenvalue_normalized"] = np.log10(df["eigenvalue_normalized"])
    df["log10_rank_pct"] = np.log10(df["rank_pct"])
    
    return df


def load_performance_data(versions: list[str] | None = None, targets: list[str] | None = None) -> pd.DataFrame:
    """Load performance data with optional filtering."""
    path = get_base_dir() / "results" / "datasets" / "performance_curves.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    
    df = pd.read_parquet(path)
    if versions:
        df = df[df["version"].isin(versions)]
    if targets:
        df = df[df["target"].isin(targets)]
    return df


# =============================================================================
# Plotting Helpers
# =============================================================================

def setup_pc_axis(ax, n_components_list: list[int], show_xlabel: bool = True, fontsize: int = 9):
    """Configure x-axis with sequential positions for n_components (handles 0 correctly).
    
    Uses evenly-spaced positions (0, 1, 2, ...) with actual n_components as labels.
    This is the same approach used in decompose.py to handle n_components=0 on pseudo-log scale.
    """
    ax.set_xticks(range(len(n_components_list)))
    ax.set_xticklabels([str(n) for n in n_components_list], fontsize=fontsize, rotation=45, ha="right")
    ax.set_xlim(-0.5, len(n_components_list) - 0.5)
    if show_xlabel:
        ax.set_xlabel("Number of PCs", fontsize=9)


def get_pc_positions(n_components_values: pd.Series, n_components_list: list[int]) -> list[int]:
    """Map n_components values to sequential x positions."""
    n_comp_to_pos = {n: i for i, n in enumerate(n_components_list)}
    return [n_comp_to_pos[n] for n in n_components_values]


def save_figure(fig, output_base: Path, df: pd.DataFrame | None = None, csv_columns: list[str] | None = None):
    """Save figure as PDF/PNG and optionally export CSV."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_base}.pdf", dpi=192, bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=192, bbox_inches="tight")
    plt.close(fig)
    
    if df is not None and csv_columns:
        df[csv_columns].to_csv(f"{output_base}.csv", index=False)
    
    print(f"Exported: {output_base}.pdf, {output_base}.png" + (f", {output_base}.csv" if df is not None else ""))


def plot_eigenspectra_on_axis(ax, df, versions, x_min, x_max, y_min, y_max):
    """Plot eigenspectra for given versions on an axis with shared scales."""
    for version in versions:
        exp_df = df[df["version"] == version].sort_values("rank")
        if len(exp_df) == 0:
            print(f"WARNING: No data found for version {version}")
            continue
        ax.plot(
            exp_df["log10_rank_pct"], exp_df["log10_eigenvalue_normalized"],
            marker=MARKERS[version], color=EXPERIMENT_COLORS[version],
            label=f"{version}: {exp_df['title'].iloc[0]}",
            markersize=3, linewidth=1, alpha=0.6,
        )
    
    # Set x-axis ticks as percentages (log scale)
    pct_ticks = [0.1, 0.25, 1, 5, 10, 20, 50, 100]
    ax.set_xticks([np.log10(p) for p in pct_ticks])
    ax.set_xticklabels([f"{p}%" for p in pct_ticks])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)


def plot_performance_curve(ax, perf_df, version: str, target: str, metric: str, 
                           n_components_list: list[int], show_label: bool = True):
    """Plot a single performance curve on the axis."""
    subset = perf_df[(perf_df["version"] == version) & (perf_df["target"] == target)].copy()
    if len(subset) == 0:
        return
    
    subset = subset.sort_values("n_components")
    x_positions = get_pc_positions(subset["n_components"], n_components_list)
    
    ax.plot(
        x_positions, subset[metric],
        marker=TASK_MARKERS.get(target, "o"),
        color=TASK_COLORS.get(target, "#333333"),
        label=TASK_LABELS.get(target, target) if show_label else None,
        markersize=4, linewidth=1.2, alpha=0.8,
    )


# =============================================================================
# Visualization Commands
# =============================================================================

def cmd_visualize_multimodal_eigenspectra(args: argparse.Namespace) -> None:
    """Visualize eigenspectra across modalities and models."""
    all_versions = [v for vs in EXPERIMENT_GROUPS.values() for v in vs]
    df = load_spectral_data(all_versions)
    
    missing = set(all_versions) - set(df["version"].unique())
    if missing:
        raise ValueError(f"Missing experiments in data: {missing}")
    
    # Calculate global axis limits with margins
    x_min, x_max = df["log10_rank_pct"].min(), df["log10_rank_pct"].max()
    y_min, y_max = df["log10_eigenvalue_normalized"].min(), df["log10_eigenvalue_normalized"].max()
    x_margin, y_margin = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    
    for idx, (group_name, group_versions) in enumerate(EXPERIMENT_GROUPS.items()):
        ax = axes[idx]
        plot_eigenspectra_on_axis(ax, df, group_versions, x_min, x_max, y_min, y_max)
        if idx >= 2:
            ax.set_xlabel("Model Width %")
        if idx % 2 == 0:
            ax.set_ylabel("log₁₀(Normalized Eigenvalue)")
        ax.set_title(group_name)
        ax.legend(fontsize=7, loc="lower left")
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    
    csv_cols = ["version", "title", "group", "rank", "rank_normalized", "rank_pct", 
                "log10_rank_pct", "eigenvalue", "eigenvalue_normalized", 
                "log10_eigenvalue", "log10_eigenvalue_normalized", "n_samples"]
    save_figure(fig, get_base_dir() / "results" / "figures" / "multimodal_eigenspectra", df, csv_cols)


def cmd_visualize_performance_overlay(args: argparse.Namespace) -> None:
    """Visualize eigenspectra with AUPRC performance overlay for Functional DNA models."""
    target_versions = EXPERIMENT_GROUPS["Functional DNA models"]
    target_tasks = ["species", "membership", "gc_content", "repeat_fraction"]
    spectral_df = load_spectral_data(target_versions)
    perf_df = load_performance_data(target_versions, targets=target_tasks)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    all_handles, all_labels = [], []
    
    for idx, version in enumerate(target_versions):
        ax = axes[idx]
        row, col = idx // 2, idx % 2
        is_left = col == 0
        is_right = col == 1
        is_bottom = row == 1
        
        # Get performance data for this version to determine x-axis values
        perf_v = perf_df[perf_df["version"] == version].sort_values("n_components")
        
        # Use per-experiment n_components list (only where performance data exists)
        n_components_list = sorted(perf_v["n_components"].unique())
        
        # Get spectral data for this version
        spec_v = spectral_df[spectral_df["version"] == version].sort_values("rank")
        n_comp_to_pos = {n: i for i, n in enumerate(n_components_list)}
        
        # Plot AUPRC performance for each task on primary y-axis
        for task in target_tasks:
            task_perf = perf_v[perf_v["target"] == task].sort_values("n_components")
            if len(task_perf) > 0:
                x_positions = get_pc_positions(task_perf["n_components"], n_components_list)
                show_label = idx == 0  # Only add legend labels for first subplot
                ax.plot(
                    x_positions, task_perf["auprc_macro"],
                    marker=TASK_MARKERS.get(task, "o"),
                    color=TASK_COLORS.get(task, "#333333"),
                    label=TASK_LABELS.get(task, task) if show_label else None,
                    markersize=3, linewidth=2, alpha=0.9, zorder=2,
                )
        
        ax.set_ylim(0, 1.05)
        
        # Y-axis 1 (AUPRC) labels only on left column
        if is_left:
            ax.set_ylabel("AUPRC Score")
        else:
            ax.tick_params(axis="y", labelleft=False)
        
        # Create second y-axis for eigenspectrum
        ax2 = ax.twinx()
        
        # Plot eigenspectrum in background (black, with dots and lines)
        spec_positions = []
        spec_eigenvalues = []
        for _, row_data in spec_v.iterrows():
            if row_data["rank"] in n_comp_to_pos:
                spec_positions.append(n_comp_to_pos[row_data["rank"]])
                spec_eigenvalues.append(row_data["log10_eigenvalue_normalized"])
        
        if spec_positions:
            ax2.plot(
                spec_positions, spec_eigenvalues,
                marker="o", color="black", label="Eigenspectrum",
                markersize=3, linewidth=1, alpha=0.5, zorder=1,
            )
        
        ax2.set_ylim(spectral_df["log10_eigenvalue_normalized"].min() * 1.05, 0.05)
        
        # Y-axis 2 (eigenvalue) labels only on right column
        if is_right:
            ax2.set_ylabel("log₁₀(Normalized Eigenvalue)")
        else:
            ax2.tick_params(axis="y", labelleft=False, labelright=False)
        
        # Configure axis with per-experiment n_components
        setup_pc_axis(ax, n_components_list, show_xlabel=is_bottom)
        ax.grid(True, alpha=0.3, zorder=0)
        
        exp_title = spec_v["title"].iloc[0] if len(spec_v) > 0 else version
        ax.set_title(f"{version}: {exp_title}", fontsize=9)
        
        if idx == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            all_handles.extend(h1 + h2)
            all_labels.extend(l1 + l2)
    
    fig.legend(all_handles, all_labels, loc="lower center", ncol=len(all_labels), 
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, wspace=0.1)
    
    csv_cols = ["version", "title", "group", "rank", "rank_normalized", "rank_pct",
                "log10_rank_pct", "eigenvalue", "eigenvalue_normalized",
                "log10_eigenvalue", "log10_eigenvalue_normalized", "n_samples"]
    save_figure(fig, get_base_dir() / "results" / "figures" / "performance_overlay", spectral_df, csv_cols)


def cmd_visualize_performance_metrics(args: argparse.Namespace) -> None:
    """Visualize performance metrics faceted by metric (rows) and experiment (cols)."""
    target_versions = EXPERIMENT_GROUPS["Functional DNA models"]
    target_tasks = ["species", "membership", "gc_content", "repeat_fraction", "kmer_entropy_3"]
    metrics = ["accuracy", "f1_macro", "roc_auc_macro", "auprc_macro"]
    
    perf_df = load_performance_data(target_versions, target_tasks)
    n_components_list = sorted(perf_df["n_components"].unique())
    
    fig, axes = plt.subplots(len(metrics), len(target_versions), 
                             figsize=(3.5 * len(target_versions), 2.5 * len(metrics)))
    
    for row_idx, metric in enumerate(metrics):
        for col_idx, version in enumerate(target_versions):
            ax = axes[row_idx, col_idx]
            
            for task in target_tasks:
                plot_performance_curve(ax, perf_df, version, task, metric, n_components_list,
                                       show_label=(row_idx == 0 and col_idx == 0))
            
            setup_pc_axis(ax, n_components_list, show_xlabel=(row_idx == len(metrics) - 1))
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            
            if col_idx == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            if row_idx == 0:
                exp_title = perf_df[perf_df["version"] == version]["title"].iloc[0] if len(perf_df[perf_df["version"] == version]) > 0 else version
                ax.set_title(f"{version}: {exp_title}", fontsize=9)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc="lower right")
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    csv_cols = ["version", "title", "target", "n_components"] + metrics
    save_figure(fig, get_base_dir() / "results" / "figures" / "performance_metrics", perf_df, csv_cols)


# =============================================================================
# Main Entry Point
# =============================================================================

COMMANDS = {
    "visualize_multimodal_eigenspectra": (cmd_visualize_multimodal_eigenspectra, "Visualize eigenspectra across modalities"),
    "visualize_performance_overlay": (cmd_visualize_performance_overlay, "Visualize eigenspectra with F1 performance overlay"),
    "visualize_performance_metrics": (cmd_visualize_performance_metrics, "Visualize performance metrics faceted by metric and experiment"),
}


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run all visualization commands."""
    for i, (name, (cmd_func, _)) in enumerate(COMMANDS.items(), 1):
        print(f"\n{'='*60}\n{i}. {name}...\n{'='*60}")
        cmd_func(args)
    print(f"\n{'='*60}\nAll visualizations complete!\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    for name, (_, help_text) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    subparsers.add_parser("run_all", help="Run all visualization commands")
    
    args = parser.parse_args()
    
    if args.command == "run_all":
        cmd_run_all(args)
    else:
        COMMANDS[args.command][0](args)


if __name__ == "__main__":
    main()
