"""
Script to create a subset of the Vertebrate Genomes dataset for PlantCAD2 spectral analysis.

This script:
1. Streams records from train, validation and test splits of emarro/vertebrate_genomes
2. Shuffles with a buffer size of 10000
3. Truncates sequences to exactly CONTEXT_LENGTH bp (all source sequences are 12kbp)
4. Gathers samples to match plantcad/Angiosperm_65_genomes_8192bp split sizes
5. Uploads to plantcad/vertebrate-genomes-plantcad2-c{CONTEXT_LENGTH}
"""

import argparse
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from huggingface_hub import HfApi


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
SOURCE_DATASET = "emarro/vertebrate_genomes"
SOURCE_REVISION = "9703952e2c90c822ea8a96c9638b584ccaf36d4e"
DEST_DATASET = "plantcad/vertebrate-genomes-plantcad2-c4096"

# README content
README_CONTENT = f"""---
license: apache-2.0
tags:
- biology
- DNA
- genomics
- genetics
- vertebrates
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

# Vertebrate Genomes PlantCAD2 Subset ({CONTEXT_LENGTH}bp)

This dataset is a curated subset of [emarro/vertebrate_genomes](https://huggingface.co/datasets/emarro/vertebrate_genomes) 
designed for comparative spectral analysis with plant genomic data.

## Dataset Description

Sequences were randomly sampled from Vertebrate Genomes (revision {SOURCE_REVISION}), 
truncated to match the sample sizes per split of the 
[plantcad/Angiosperm_65_genomes_8192bp](https://huggingface.co/datasets/plantcad/Angiosperm_65_genomes_8192bp) dataset.

### Processing Steps

1. **Streaming**: Records were streamed from the standard train/validation/test splits
2. **Shuffling**: Applied shuffle with buffer size of 10,000 for random sampling
3. **Truncation**: All sequences (originally 12kbp) were truncated to exactly {CONTEXT_LENGTH}bp
4. **Sampling**: Collected samples to match PlantCAD split sizes

### Split Sizes

| Split | Number of Examples |
|-------|-------------------|
| train | 2,638,656 |
| validation | 329,832 |
| test | 329,832 |

### Sequence Length

All sequences are exactly **{CONTEXT_LENGTH} base pairs**.

## Source Dataset

Vertebrate Genomes contains DNA sequences from vertebrate species, with all sequences being 12,000 base pairs in length.
This subset uses revision `{SOURCE_REVISION}`.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("plantcad/vertebrate-genomes-plantcad2-c{CONTEXT_LENGTH}")
```

## License

Apache 2.0
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Vertebrate Genomes subset for PlantCAD2 spectral analysis"
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


def process_split(split_name: str, target_count: int, shuffle_buffer_size: int, seed: int) -> Dataset:
    """
    Process a single split from Vertebrate Genomes.
    
    Args:
        split_name: Name of the split (train, validation, test)
        target_count: Number of samples to collect
        shuffle_buffer_size: Buffer size for shuffling
        seed: Random seed for reproducibility
        
    Returns:
        Dataset with processed sequences
    """
    print(f"\nProcessing {split_name} split (target: {target_count:,} samples)...")
    
    # Load dataset in streaming mode with specific revision
    ds = load_dataset(
        SOURCE_DATASET,
        split=split_name,
        revision=SOURCE_REVISION,
        streaming=True,
    )
    
    # Shuffle the stream
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    
    # Collect sequences
    sequences = []
    processed = 0
    
    pbar = tqdm(total=target_count, desc=f"Collecting {split_name}")
    
    for example in ds:
        processed += 1
        text = example["sequence"]
        
        # Truncate to target length (all source sequences are 12kbp)
        truncated_seq = text[:CONTEXT_LENGTH]
        if len(truncated_seq) != CONTEXT_LENGTH:
            raise ValueError(f"Truncated sequence length mismatch: {len(truncated_seq)} != {CONTEXT_LENGTH}")
        sequences.append(truncated_seq)
        pbar.update(1)
        
        # Check if we've collected enough
        if len(sequences) >= target_count:
            break
    
    pbar.close()
    
    print(f"  Processed: {processed:,} records")
    print(f"  Collected: {len(sequences):,} sequences")
    
    if len(sequences) < target_count:
        print(f"  WARNING: Only collected {len(sequences):,} of {target_count:,} target sequences!")
    
    # Create dataset
    return Dataset.from_dict({"text": sequences})


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Vertebrate Genomes PlantCAD2 Subset Creator")
    print("=" * 60)
    print(f"Source: {SOURCE_DATASET}")
    print(f"Revision: {SOURCE_REVISION}")
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

