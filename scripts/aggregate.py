"""Aggregate experiment results from results/sep into consolidated parquet files."""

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd


def load_experiment_metadata(exp_dir: Path) -> dict:
    """Load experiment.json from a result directory."""
    with open(exp_dir / "experiment.json") as f:
        return json.load(f)


def add_metadata_columns(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Add metadata columns to DataFrame, converting lists/dicts to JSON strings."""
    for key, value in metadata.items():
        # Convert lists/dicts to JSON strings for storage
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        df[key] = value
    return df


def find_performance_csv(exp_dir: Path) -> Path | None:
    """Find pc_scaling_curves.csv in an experiment directory."""
    csv_path = exp_dir / "pc_scaling_curves.csv"
    return csv_path if csv_path.exists() else None


def find_spectral_csv(exp_dir: Path) -> Path | None:
    """Find eigenspectrum CSV (excluding confusion_matrix) in an experiment directory."""
    candidates = [
        p for p in exp_dir.glob("eigenspectrum_*.csv")
        if "confusion_matrix" not in p.name
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Multiple eigenspectrum CSVs in {exp_dir}: {candidates}")
    return candidates[0]


def aggregate_curves(
    results_dir: Path,
    csv_path_finder: Callable[[Path], Path | None]
) -> pd.DataFrame:
    """Aggregate CSV files from experiments using a provided path finder function."""
    dfs = []
    for exp_dir in sorted(results_dir.glob("v*")):
        csv_path = csv_path_finder(exp_dir)
        if csv_path is None:
            continue
        
        metadata = load_experiment_metadata(exp_dir)
        df = pd.read_csv(csv_path)
        df = add_metadata_columns(df, metadata)
        dfs.append(df)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated", category=FutureWarning)
        return pd.concat(dfs, ignore_index=True)


def aggregate_performance_curves(results_dir: Path) -> pd.DataFrame:
    """Aggregate pc_scaling_curves.csv from all experiments."""
    return aggregate_curves(results_dir, find_performance_csv)


def aggregate_spectral_curves(results_dir: Path) -> pd.DataFrame:
    """Aggregate eigenspectrum_*.csv (excluding confusion_matrix) from all experiments."""
    return aggregate_curves(results_dir, find_spectral_csv)


def cmd_aggregate_sep_results(args: argparse.Namespace) -> None:
    """Aggregate results from results/sep into parquet files."""
    base_dir = Path(__file__).parent.parent
    sep_dir = base_dir / "results" / "sep"
    output_dir = base_dir / "results" / "datasets"

    performance_df = aggregate_performance_curves(sep_dir)
    performance_path = output_dir / "performance_curves.parquet"
    performance_df.to_parquet(performance_path, index=False)
    print(f"Wrote {len(performance_df)} rows to {performance_path}")

    spectral_df = aggregate_spectral_curves(sep_dir)
    spectral_path = output_dir / "spectral_curves.parquet"
    spectral_df.to_parquet(spectral_path, index=False)
    print(f"Wrote {len(spectral_df)} rows to {spectral_path}")


def main():
    parser = argparse.ArgumentParser(description="Process experiment results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("aggregate_sep_results", help="Aggregate results from results/sep")

    args = parser.parse_args()

    if args.command == "aggregate_sep_results":
        cmd_aggregate_sep_results(args)


if __name__ == "__main__":
    main()

