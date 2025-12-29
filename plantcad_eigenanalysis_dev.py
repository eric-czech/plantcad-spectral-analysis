"""
Development script for loading and plotting eigenspectrum analysis results.

Usage:
    python plantcad_eigenanalysis_dev.py --checkpoint path/to/checkpoint.pkl --output path/to/output.png
"""

import argparse
import os

from plantcad_eigenanalysis import (
    AnalysisResults,
    PCAResult,
    load_checkpoint,
    plot_eigenspectrum,
)

# These imports are needed for pickle to deserialize the checkpoint
_ = AnalysisResults, PCAResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load eigenspectrum checkpoint and plot results"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint pkl file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output plot (e.g., output.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load checkpoint
    results = load_checkpoint(args.checkpoint)
    
    # Create parent directories for output path
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot eigenspectrum and save to output path
    plot_eigenspectrum(results.pca_results, output_path=args.output, min_n_samples=1024)
    
    print("Done!")


if __name__ == "__main__":
    main()

