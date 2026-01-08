"""Analyze and visualize aggregated experiment results."""

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_GROUPS = {
    "Functional DNA models": ["v13", "v17", "v41", "v28"],
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

# Portland palette colors for reference:
# 0.0: #0C3383 (dark blue), 0.25: #0A88BA (teal), 0.5: #F2D338 (yellow), 
# 0.75: #F28F38 (orange), 1.0: #D91E1E (red)
_PORTLAND_COLORS = ["#0C3383", "#0A88BA", "#F2D338", "#F28F38", "#D91E1E"]
_PORTLAND_CMAP = mcolors.LinearSegmentedColormap.from_list("portland", _PORTLAND_COLORS)

def _portland_color(pos: float) -> str:
    """Get hex color from Portland palette at position 0-1."""
    rgba = _PORTLAND_CMAP(pos)
    return mcolors.rgb2hex(rgba[:3])

TASK_COLORS = {
    # Species: darker blue from Portland (pos ~0.08)
    "species": _portland_color(0.08),
    # Sequence composition tasks: yellow-green range from Portland (lighter to darker as k increases)
    "gc_content": _portland_color(0.56),
    "kmer_entropy_1": _portland_color(0.60),  # yellow/orange
    "kmer_entropy_3": _portland_color(0.525), # yellow
    "kmer_entropy_9": _portland_color(0.425), # yellow/green
    # Auxiliary tasks from Portland palette
    "membership": _portland_color(1.0),       # red
    "repeat_fraction": _portland_color(0.4),  # green
}

TASK_MARKERS = {
    "species": "o", "membership": "s", "gc_content": "^",
    "repeat_fraction": "D", "kmer_entropy_1": "v", 
    "kmer_entropy_3": "P", "kmer_entropy_9": "<",
}

TASK_LABELS = {
    "species": "Species", "membership": "Train Membership (balanced)", "gc_content": "GC Content",
    "repeat_fraction": "Repeat Fraction", "kmer_entropy_1": "K-mer (k=1)",
    "kmer_entropy_3": "K-mer (k=3)", "kmer_entropy_9": "K-mer (k=9)",
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


def get_experiment_title(version: str) -> str:
    """Load experiment title from experiment.json."""
    exp_path = get_base_dir() / "results" / "sep" / version / "experiment.json"
    with open(exp_path) as f:
        return json.load(f)["title"]


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
    
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
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
            ax.set_ylabel("AUPRC")
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
               bbox_to_anchor=(0.5, -0.035), fontsize=8)
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
    
    # Get n_components list per version (x-axis varies per experiment/column)
    n_components_per_version = {
        v: sorted(perf_df[perf_df["version"] == v]["n_components"].unique())
        for v in target_versions
    }
    
    fig, axes = plt.subplots(len(metrics), len(target_versions), 
                             figsize=(3.2 * len(target_versions), 2.0 * len(metrics)))
    
    for row_idx, metric in enumerate(metrics):
        for col_idx, version in enumerate(target_versions):
            ax = axes[row_idx, col_idx]
            n_components_list = n_components_per_version[version]
            is_bottom_row = row_idx == len(metrics) - 1
            
            for task in target_tasks:
                plot_performance_curve(ax, perf_df, version, task, metric, n_components_list,
                                       show_label=(row_idx == 0 and col_idx == 0))
            
            setup_pc_axis(ax, n_components_list, show_xlabel=is_bottom_row)
            # ROC AUC ranges from 0.5 (random) to 1, others from 0 to 1
            if metric == "roc_auc_macro":
                ax.set_ylim(0.45, 1.02)
            else:
                ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            
            # Only show x-axis tick labels on bottom row
            if not is_bottom_row:
                ax.tick_params(axis="x", labelbottom=False)
            
            # Only show y-axis tick labels on left-most column
            if col_idx == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
            else:
                ax.tick_params(axis="y", labelleft=False)
            
            if row_idx == 0:
                exp_title = perf_df[perf_df["version"] == version]["title"].iloc[0] if len(perf_df[perf_df["version"] == version]) > 0 else version
                # Break title after "*bp" for readability
                title_text = f"{version}: {exp_title}"
                title_text = title_text.replace("bp ", "bp\n", 1)
                ax.set_title(title_text, fontsize=7)
    
    # Legend outside plot area, top right (aligned with top of first row)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="upper left", bbox_to_anchor=(0.925, 0.95))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08, wspace=0.05, right=0.92)
    
    csv_cols = ["version", "title", "target", "n_components"] + metrics
    save_figure(fig, get_base_dir() / "results" / "figures" / "performance_metrics", perf_df, csv_cols)


def load_spectral_data_all_samples(versions: list[str] | None = None) -> pd.DataFrame:
    """Load spectral data with ALL sample sizes (not just max)."""
    path = get_base_dir() / "results" / "datasets" / "spectral_curves.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    
    df = pd.read_parquet(path)
    if versions:
        df = df[df["version"].isin(versions)]
    
    # Assert uniqueness by (version, n_samples, rank)
    key_cols = ["version", "n_samples", "rank"]
    assert df.duplicated(subset=key_cols).sum() == 0, f"Duplicate records found by {key_cols}"
    assert df["rank"].min() >= 1, f"Rank must be >= 1, got min={df['rank'].min()}"
    
    # Add normalized columns (normalize per version AND per n_samples)
    version_to_group = {v: g for g, vs in EXPERIMENT_GROUPS.items() for v in vs}
    df["group"] = df["version"].map(version_to_group)
    df["eigenvalue_normalized"] = df.groupby(["version", "n_samples"])["eigenvalue"].transform(lambda x: x / x.max())
    df["log10_eigenvalue"] = np.log10(df["eigenvalue"])
    df["log10_eigenvalue_normalized"] = np.log10(df["eigenvalue_normalized"])
    
    return df


def cmd_visualize_spectral_convergence(args: argparse.Namespace) -> None:
    """Visualize spectral convergence metrics across sample sizes for all experiments."""
    all_versions = [v for vs in EXPERIMENT_GROUPS.values() for v in vs]
    primary_versions = EXPERIMENT_GROUPS["Functional DNA models"]
    df = load_spectral_data_all_samples(all_versions)
    
    # Compute metrics per (version, n_samples)
    def compute_metrics(g):
        eigs = g["eigenvalue"].values
        nonzero = eigs[eigs > 0]
        return pd.Series({
            "max_eigenvalue": eigs.max(),
            "condition_number": nonzero.max() / nonzero.min() if len(nonzero) > 0 else np.inf,
            "effective_sample_size": (eigs.sum() ** 2) / (eigs ** 2).sum() if (eigs ** 2).sum() > 0 else 0,
            "title": g["title"].iloc[0],
            "group": g["group"].iloc[0],
        })
    
    metrics_df = df.groupby(["version", "n_samples"]).apply(compute_metrics, include_groups=False).reset_index()
    
    # Get max n_samples per version for row 2
    max_samples_df = metrics_df.loc[metrics_df.groupby("version")["n_samples"].idxmax()]
    
    # Group colors for row 2
    group_colors = {
        "Functional DNA models": "#1f77b4",
        "Functional DNA models (random weights)": "#ff7f0e", 
        "Whole-genome DNA models": "#2ca02c",
        "Text models": "#d62728",
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metric_configs = [
        ("max_eigenvalue", "Largest Eigenvalue", True),
        ("condition_number", "Condition Number", True),
        ("effective_sample_size", "Effective Sample Size", False),
    ]
    
    # Row 1: Convergence curves for primary functional DNA models (n_samples >= 1024)
    row1_handles, row1_labels = [], []
    for col, (metric, ylabel, use_log_y) in enumerate(metric_configs):
        ax = axes[0, col]
        for version in primary_versions:
            v_df = metrics_df[(metrics_df["version"] == version) & (metrics_df["n_samples"] >= 1024)].sort_values("n_samples")
            if len(v_df) == 0:
                continue
            line, = ax.plot(
                v_df["n_samples"], v_df[metric],
                marker=MARKERS[version], color=EXPERIMENT_COLORS[version],
                markersize=5, linewidth=1.5, alpha=0.8,
            )
            if col == 0:  # Collect handles from first column only
                row1_handles.append(line)
                row1_labels.append(f"{version}: {v_df['title'].iloc[0]}")
        
        ax.set_xscale("log", base=2)
        if use_log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Sample Size")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
    
    # Row 2: Bar charts at max n_samples for all experiments
    for col, (metric, ylabel, use_log_y) in enumerate(metric_configs):
        ax = axes[1, col]
        
        # Order by group then version
        ordered_versions = [v for vs in EXPERIMENT_GROUPS.values() for v in vs]
        bar_data = max_samples_df.set_index("version").loc[ordered_versions].reset_index()
        
        x = np.arange(len(bar_data))
        colors = [group_colors[bar_data.iloc[i]["group"]] for i in range(len(bar_data))]
        
        ax.bar(x, bar_data[metric], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        
        if use_log_y:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} (at Max Samples)")
        ax.set_xticks(x)
        ax.set_xticklabels(bar_data["version"], rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    
    # Legends: Row 1 between rows, Row 2 at bottom
    from matplotlib.patches import Patch
    group_handles = [Patch(facecolor=c, edgecolor="black", label=g) for g, c in group_colors.items()]
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.75)
    
    # Row 1 legend centered between rows (2x2 layout)
    leg1 = fig.legend(row1_handles, row1_labels, loc="center", ncol=2, fontsize=7,
                      bbox_to_anchor=(0.5, 0.54), title="Experiment")
    # Row 2 legend centered at bottom (2x2 layout)
    fig.legend(group_handles, group_colors.keys(), loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), title="Experiment Group")
    fig.add_artist(leg1)
    
    csv_cols = ["version", "title", "group", "n_samples", "max_eigenvalue", "condition_number", "effective_sample_size"]
    save_figure(fig, get_base_dir() / "results" / "figures" / "spectral_convergence", metrics_df, csv_cols)


def cmd_visualize_plantcad_decomposition(args: argparse.Namespace) -> None:
    """Visualize PlantCAD decomposition: eigenspectra by sample size + performance overlay."""
    version = "v13"
    target_tasks = ["membership", "kmer_entropy_1", "kmer_entropy_3", "kmer_entropy_9", "species"]
    
    # Load data - spectral with all sample sizes
    spectral_df = load_spectral_data_all_samples([version])
    perf_df = load_performance_data([version], targets=target_tasks)
    
    # Get sample sizes and create grey-to-black color palette
    sample_sizes = sorted(spectral_df["n_samples"].unique())
    # Grey scale: light grey (visible on white) → black
    grey_colors = ["#BBBBBB", "#000000"]  # light grey to black
    grey_cmap = mcolors.LinearSegmentedColormap.from_list("greys", grey_colors)
    sample_colors = {s: grey_cmap(i / (len(sample_sizes) - 1)) for i, s in enumerate(sample_sizes)}
    
    # Get the common x-axis values (PC counts from performance data n_components)
    n_components_list = sorted(perf_df["n_components"].unique())
    max_sample_size = max(sample_sizes)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # =========================================================================
    # Facet 1: Eigenvalues by PC count (all ranks), colored by sample size
    # =========================================================================
    ax1 = axes[0]
    
    for n_samples in sample_sizes:
        sample_df = spectral_df[spectral_df["n_samples"] == n_samples].sort_values("rank")
        
        # Plot all eigenvalues - use rank directly on x-axis
        ax1.plot(
            sample_df["rank"], sample_df["log10_eigenvalue_normalized"],
            marker="o", color=sample_colors[n_samples],
            label=f"n={n_samples:,}",
            markersize=2, linewidth=1, alpha=0.8,
        )
    
    # Fit log-linear line to largest sample size for ranks 10-100
    spec_max = spectral_df[spectral_df["n_samples"] == max_sample_size].sort_values("rank")
    fit_data = spec_max[(spec_max["rank"] >= 10) & (spec_max["rank"] <= 100)]
    log10_ranks = np.log10(fit_data["rank"])
    log10_eigenvalues = fit_data["log10_eigenvalue_normalized"]
    coeffs = np.polyfit(log10_ranks, log10_eigenvalues, 1)  # slope, intercept
    
    # Plot best-fit line across whole x-axis range
    all_ranks = spectral_df["rank"].unique()
    x_fit = np.array([all_ranks.min(), all_ranks.max()])
    y_fit = coeffs[0] * np.log10(x_fit) + coeffs[1]
    ax1.plot(x_fit, y_fit, linestyle="--", color="grey", linewidth=1.5, alpha=0.8,
             label=f"Fit (ranks 10–100): slope={coeffs[0]:.2f}")
    
    ax1.set_xscale("log")
    ax1.set_xlabel("Eigenvalue Rank")
    ax1.set_ylabel("log₁₀(Normalized Eigenvalue)")
    ax1.set_title("Eigenspectrum by Sample Size")
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="Sample Size", fontsize=7, loc="lower left")
    
    # =========================================================================
    # Facet 2: AUPRC performance with eigenspectrum overlay (largest sample size)
    # =========================================================================
    ax2 = axes[1]
    
    # Add subtle grey shaded region for eigenvalue ranks 128-768 (83% of representation space)
    pos_128 = n_components_list.index(128)
    pos_768 = n_components_list.index(768)
    ax2.axvspan(pos_128, pos_768 + 0.5, color="#e0e0e0", alpha=0.5, zorder=0)
    
    # Plot AUPRC for each task
    perf_v = perf_df.sort_values("n_components")
    for task in target_tasks:
        task_perf = perf_v[perf_v["target"] == task].sort_values("n_components")
        if len(task_perf) > 0:
            x_positions = get_pc_positions(task_perf["n_components"], n_components_list)
            ax2.plot(
                x_positions, task_perf["auprc_macro"],
                marker=TASK_MARKERS.get(task, "o"),
                color=TASK_COLORS.get(task, "#333333"),
                label=TASK_LABELS.get(task, task),
                markersize=5, linewidth=2, alpha=0.9, zorder=2,
            )
    
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("AUPRC Score")
    
    # Secondary y-axis for eigenspectrum at largest sample size
    ax2_twin = ax2.twinx()
    
    spec_max = spectral_df[spectral_df["n_samples"] == max_sample_size].sort_values("rank")
    n_comp_to_pos = {n: i for i, n in enumerate(n_components_list)}
    
    spec_positions = []
    spec_eigenvalues = []
    for _, row_data in spec_max.iterrows():
        if row_data["rank"] in n_comp_to_pos:
            spec_positions.append(n_comp_to_pos[row_data["rank"]])
            spec_eigenvalues.append(row_data["log10_eigenvalue_normalized"])
    
    if spec_positions:
        ax2_twin.plot(
            spec_positions, spec_eigenvalues,
            marker="o", color="black", label=f"Eigenspectrum (n={max_sample_size:,})",
            markersize=2.5, linewidth=1, alpha=0.5, zorder=1,
        )
    
    ax2_twin.set_ylim(spectral_df["log10_eigenvalue_normalized"].min() * 1.05, 0.05)
    ax2_twin.set_ylabel("log₁₀(Normalized Eigenvalue)")
    
    setup_pc_axis(ax2, n_components_list, show_xlabel=False)
    ax2.set_xlabel("Eigenvalue Rank", fontsize=9)
    ax2.grid(True, alpha=0.3, zorder=0)
    ax2.set_title("AUPRC Performance with Eigenspectrum Overlay")
    
    # Combined legend - centered at rank 128 position
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, title="Classification Task", fontsize=7, loc="lower center", bbox_to_anchor=(0.73, 0.0))
    
    # Annotation for shaded region (outside plot, bottom right)
    ax2.annotate("Shaded: ranks 128–768 (83% of representation)", 
                 xy=(1.02, -0.14), xycoords="axes fraction",
                 fontsize=7, color="#666666", ha="right")
    
    # Suptitle
    exp_title = spectral_df["title"].iloc[0] if len(spectral_df) > 0 else version
    fig.suptitle(f"{version}: {exp_title}", fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    csv_cols = ["version", "title", "n_samples", "rank", "eigenvalue", 
                "eigenvalue_normalized", "log10_eigenvalue", "log10_eigenvalue_normalized"]
    save_figure(fig, get_base_dir() / "results" / "figures" / "plantcad_decomposition", spectral_df, csv_cols)


def cmd_visualize_select_eigenspectra(args: argparse.Namespace) -> None:
    """Visualize eigenspectra for select experiments (v13 PlantCAD plants, v40 GLM animal promoter, v46 Marin metagenomes)."""
    versions = ["v13", "v40", "v46"]
    df = load_spectral_data(versions)
    
    colors = {"v13": "#1f77b4", "v40": "#2ca02c", "v46": "#d62728"}
    vlines = {"v13": [128, 650], "v40": [580, 790, 950], "v46": [410, 450, 580]}
    
    def add_vlines(ax, ranks, color="gray", stagger_y=False):
        for i, rank in enumerate(ranks):
            ax.axvline(rank, color=color, linestyle="--", linewidth=1, alpha=0.7)
            y_offset = -2 - (i * 12 if stagger_y else 0)
            ax.annotate(f"~{rank}", xy=(rank, ax.get_ylim()[1]), xytext=(2, y_offset),
                        textcoords="offset points", fontsize=9, va="top", ha="left", color=color)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 6))
    
    for col, version in enumerate(versions):
        subset = df[df["version"] == version].sort_values("rank")
        n_samples = subset["n_samples"].iloc[0]
        title = get_experiment_title(version)
        
        # Top row: linear x-axis
        ax_top = axes[0, col]
        ax_top.plot(subset["rank"], subset["eigenvalue"], color=colors[version], linewidth=1.5)
        ax_top.set_xlabel("Eigenvalue Rank", fontsize=11)
        ax_top.set_ylabel("Eigenvalue", fontsize=11)
        ax_top.set_yscale("log")
        ax_top.set_title(f"{version}: {title}\n(n={n_samples:,})", fontsize=11)
        ax_top.grid(True, alpha=0.3)
        add_vlines(ax_top, vlines[version])
        
        # Bottom row: log x-axis
        ax_bot = axes[1, col]
        ax_bot.plot(subset["rank"], subset["eigenvalue"], color=colors[version], linewidth=1.5)
        ax_bot.set_xlabel("Eigenvalue Rank (log₁₀)", fontsize=11)
        ax_bot.set_ylabel("Eigenvalue", fontsize=11)
        ax_bot.set_xscale("log")
        ax_bot.set_yscale("log")
        ax_bot.grid(True, alpha=0.3)
        # Stagger annotations for v40 and v46 to avoid overlap
        add_vlines(ax_bot, vlines[version], stagger_y=(version in ["v40", "v46"]))
    
    plt.tight_layout()
    csv_cols = ["version", "rank", "eigenvalue", "n_samples"]
    save_figure(fig, get_base_dir() / "results" / "figures" / "select_eigenspectra", df, csv_cols)


# =============================================================================
# Main Entry Point
# =============================================================================

COMMANDS = {
    "visualize_multimodal_eigenspectra": (cmd_visualize_multimodal_eigenspectra, "Visualize eigenspectra across modalities"),
    "visualize_performance_overlay": (cmd_visualize_performance_overlay, "Visualize eigenspectra with F1 performance overlay"),
    "visualize_performance_metrics": (cmd_visualize_performance_metrics, "Visualize performance metrics faceted by metric and experiment"),
    "visualize_plantcad_decomposition": (cmd_visualize_plantcad_decomposition, "Visualize PlantCAD decomposition with eigenspectra by sample size"),
    "visualize_spectral_convergence": (cmd_visualize_spectral_convergence, "Visualize spectral convergence metrics by sample size"),
    "visualize_select_eigenspectra": (cmd_visualize_select_eigenspectra, "Visualize eigenspectra for select experiments (v13, v40, v46)"),
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
