"""
Script to create a subset of the OpenGenome2 dataset for PlantCAD2 spectral analysis.

This script:
1. Streams records from train, validation and test splits of arcinstitute/opengenome2
2. Shuffles with a buffer size of 10000
3. Filters sequences to at least CONTEXT_LENGTH bp
4. Truncates sequences to exactly CONTEXT_LENGTH bp
5. Gathers samples to match plantcad/Angiosperm_65_genomes_8192bp split sizes
6. Uploads to plantcad/opengenome2-plantcad2-c{CONTEXT_LENGTH}
"""

import argparse
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_url


# Target sample sizes matching plantcad/Angiosperm_65_genomes_8192bp;
# counts taken from https://github.com/plantcad/plantcad-dev/issues/39
TARGET_COUNTS = {
    "train": 2_638_656,
    "validation": 329_832,
    "test": 329_832,
}

# Default parameters
CONTEXT_LENGTH = 4096
SHUFFLE_BUFFER_SIZE = 10000

# Source and destination dataset paths
SOURCE_DATASET = "arcinstitute/opengenome2"
SOURCE_SUBFOLDER = "json/pretraining_or_both_phases/metagenomes"
DEST_DATASET = "plantcad/opengenome2-metagenomes-plantcad2-c4096"

# Mapping from our split names to the file naming convention in opengenome2
SPLIT_TO_FILE_PREFIX = {
    "train": "data_metagenomics_train_",
    "validation": "data_metagenomics_valid_",
    "test": "data_metagenomics_test_",
}

# README content
README_CONTENT = f"""---
license: apache-2.0
tags:
- biology
- DNA
- genomics
- genetics
- metagenomics
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_examples: 2638656
  - name: validation
    num_examples: 329832
  - name: test
    num_examples: 329832
---

# OpenGenome2 Metagenomes PlantCAD2 Subset ({CONTEXT_LENGTH}bp)

This dataset is a curated subset of [arcinstitute/opengenome2](https://huggingface.co/datasets/arcinstitute/opengenome2) 
designed for comparative spectral analysis with plant genomic data.

## Dataset Description

Sequences were randomly sampled from OpenGenome2, filtered and truncated to match the sample sizes 
per split of the [plantcad/Angiosperm_65_genomes_8192bp](https://huggingface.co/datasets/plantcad/Angiosperm_65_genomes_8192bp) dataset.

### Processing Steps

1. **Streaming**: Records were streamed from the metagenomes subfolder (`json/pretraining_or_both_phases/metagenomes`) of OpenGenome2
2. **Shuffling**: Applied shuffle with buffer size of 10,000 for random sampling
3. **Filtering**: Sequences shorter than {CONTEXT_LENGTH}bp were excluded
4. **Truncation**: Sequences ≥{CONTEXT_LENGTH}bp were truncated to exactly {CONTEXT_LENGTH}bp
5. **Sampling**: Collected samples to match PlantCAD split sizes

### Split Sizes

| Split | Number of Examples |
|-------|-------------------|
| train | 2,638,656 |
| validation | 329,832 |
| test | 329,832 |

### Sequence Length

All sequences are exactly **{CONTEXT_LENGTH} base pairs**.

## Source Dataset

OpenGenome2 is a database of nearly 9 trillion base pairs of curated DNA from across all domains of life,
used to train Evo 2 models. Please refer to the [Evo 2 preprint](https://www.biorxiv.org/content/early/2025/02/21/2025.02.18.638918) 
for further details.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("plantcad/opengenome2-metagenomes-plantcad2-c{CONTEXT_LENGTH}")
```

## Citation

If you use this dataset, please cite the original OpenGenome2:

```bibtex
@article{{Brixi2025.02.18.638918,
    author = {{Brixi, Garyk and Durrant, Matthew G and Ku, Jerome and others}},
    title = {{Genome modeling and design across all domains of life with Evo 2}},
    year = {{2025}},
    doi = {{10.1101/2025.02.18.638918}},
    journal = {{bioRxiv}}
}}
```

## License

Apache 2.0 (inherited from OpenGenome2)
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create OpenGenome2 subset for PlantCAD2 spectral analysis"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process data but don't upload to Hugging Face",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Local directory to save the dataset",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default=DEST_DATASET,
        help=f"Hugging Face dataset repo to upload to (default: {DEST_DATASET})",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
        help="Splits to process",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=SHUFFLE_BUFFER_SIZE,
        help="Buffer size for shuffling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    return parser.parse_args()


def get_split_files(split_name: str) -> list[str]:
    """
    Get the list of JSONL.gz files for a given split from the OpenGenome2 repository.
    
    Args:
        split_name: Name of the split (train, validation, test)
        
    Returns:
        List of file paths within the repository
    """
    api = HfApi()
    
    # List all files in the metagenomes subfolder
    files = api.list_repo_files(
        repo_id=SOURCE_DATASET,
        repo_type="dataset",
    )
    
    # Filter to the subfolder and matching split prefix
    prefix = SPLIT_TO_FILE_PREFIX[split_name]
    matching_files = [
        f for f in files
        if f.startswith(SOURCE_SUBFOLDER) and prefix in f and f.endswith(".jsonl.gz")
    ]
    
    return sorted(matching_files)


def process_split(split_name: str, target_count: int, shuffle_buffer_size: int, seed: int) -> Dataset:
    """
    Process a single split from OpenGenome2.
    
    Args:
        split_name: Name of the split (train, validation, test)
        target_count: Number of samples to collect
        shuffle_buffer_size: Buffer size for shuffling
        seed: Random seed for reproducibility
        
    Returns:
        Dataset with processed sequences
    """
    print(f"\nProcessing {split_name} split (target: {target_count:,} samples)...")
    
    # Get the list of files for this split
    split_files = get_split_files(split_name)
    print(f"  Found {len(split_files)} files for {split_name} split")
    
    if not split_files:
        raise ValueError(f"No files found for split '{split_name}' in {SOURCE_SUBFOLDER}")
    
    # Build URLs for the files
    data_files = [
        hf_hub_url(repo_id=SOURCE_DATASET, filename=f, repo_type="dataset")
        for f in split_files
    ]
    
    # Load dataset in streaming mode from specific files
    ds = load_dataset(
        "json",
        data_files=data_files,
        split="train",  # load_dataset always uses "train" for data_files
        streaming=True,
    )
    
    # Shuffle the stream
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    
    # Collect sequences
    sequences = []
    processed = 0
    skipped_short = 0
    
    pbar = tqdm(total=target_count, desc=f"Collecting {split_name}")
    
    for example in ds:
        processed += 1
        text = example.get("text", "")
        
        # Skip sequences shorter than minimum length
        if len(text) < CONTEXT_LENGTH:
            skipped_short += 1
            continue
        
        # Truncate to target length
        truncated_seq = text[:CONTEXT_LENGTH]
        sequences.append(truncated_seq)
        pbar.update(1)
        
        # Check if we've collected enough
        if len(sequences) >= target_count:
            break
    
    pbar.close()
    
    print(f"  Processed: {processed:,} records")
    print(f"  Skipped (too short): {skipped_short:,} records")
    print(f"  Collected: {len(sequences):,} sequences")
    
    if len(sequences) < target_count:
        print(f"  WARNING: Only collected {len(sequences):,} of {target_count:,} target sequences!")
    
    # Create dataset
    return Dataset.from_dict({"text": sequences})


def main():
    args = parse_args()
    
    print("=" * 60)
    print("OpenGenome2 PlantCAD2 Subset Creator")
    print("=" * 60)
    print(f"Source: {SOURCE_DATASET}")
    print(f"Destination: {args.output_dataset}")
    print(f"Shuffle buffer size: {args.shuffle_buffer_size:,}")
    print(f"Context length: {CONTEXT_LENGTH:,}bp")
    print(f"Seed: {args.seed}")
    print(f"Splits to process: {args.splits}")
    if args.dry_run:
        print("DRY RUN - will not upload to Hugging Face")
    print("=" * 60)
    
    # Process each split
    datasets = {}
    for split_name in args.splits:
        target_count = TARGET_COUNTS[split_name]
        datasets[split_name] = process_split(
            split_name=split_name,
            target_count=target_count,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=args.seed,
        )
    
    # Create DatasetDict
    dataset_dict = DatasetDict(datasets)
    
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    for split_name, ds in dataset_dict.items():
        print(f"  {split_name}: {len(ds):,} sequences")
    
    # Save locally
    print(f"\nSaving dataset to {args.output_dir}...")
    dataset_dict.save_to_disk(args.output_dir, max_shard_size="500MB")
    print("Local save complete.")
    
    # Upload to Hugging Face
    if not args.dry_run:
        print(f"\nUploading dataset to {args.output_dataset}...")
        dataset_dict.push_to_hub(
            args.output_dataset,
            private=False,
        )
        print("Upload complete.")
        
        # Upload README
        print("Uploading README.md...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=README_CONTENT.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.output_dataset,
            repo_type="dataset",
        )
        print("README upload complete.")
    else:
        print("\nDry run complete - skipping upload.")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

