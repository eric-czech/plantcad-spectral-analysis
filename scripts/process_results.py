"""Aggregate experiment results from results/sep into consolidated parquet files."""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_experiment_metadata(exp_dir: Path) -> dict:
    """Load experiment.json from a result directory."""
    with open(exp_dir / "experiment.json") as f:
        return json.load(f)


def aggregate_performance_curves(results_dir: Path) -> pd.DataFrame:
    """Aggregate pc_scaling_curves.csv from all experiments."""
    dfs = []
    for exp_dir in sorted(results_dir.glob("v*")):
        csv_path = exp_dir / "pc_scaling_curves.csv"
        if not csv_path.exists():
            continue
        metadata = load_experiment_metadata(exp_dir)
        df = pd.read_csv(csv_path)
        for key, value in metadata.items():
            df[key] = value
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def aggregate_spectral_curves(results_dir: Path) -> pd.DataFrame:
    """Aggregate eigenspectrum_*.csv (excluding confusion_matrix) from all experiments."""
    dfs = []
    for exp_dir in sorted(results_dir.glob("v*")):
        # Find eigenspectrum CSV, excluding confusion_matrix files
        candidates = [
            p for p in exp_dir.glob("eigenspectrum_*.csv")
            if "confusion_matrix" not in p.name
        ]
        if not candidates:
            continue
        if len(candidates) > 1:
            raise ValueError(f"Multiple eigenspectrum CSVs in {exp_dir}: {candidates}")
        metadata = load_experiment_metadata(exp_dir)
        df = pd.read_csv(candidates[0])
        for key, value in metadata.items():
            df[key] = value
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def cmd_aggregate_sep_results(args: argparse.Namespace) -> None:
    """Aggregate results from results/sep into parquet files."""
    base_dir = Path(__file__).parent.parent
    sep_dir = base_dir / "results" / "sep"
    output_dir = base_dir / "results"

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

