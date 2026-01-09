"""Synthetic genome sequences using randomly initialized Qwen2 model."""

import argparse
import logging
import os
import random
import shutil
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COUNTS = {"train": 2_638_656, "validation": 329_832}
TEST_COUNTS = {"train": 25000, "validation": 50}

CONTEXT_LENGTH = 4096
BATCH_SIZE = 32
FLUSH_EVERY = 128  # Save every N batches
BATCHES_SUBDIR = "batches"

BASE_MODEL = "Qwen/Qwen2-1.5B"
TOKENIZER_MODEL = "kuleshov-group/PlantCAD2-Small-l24-d0768"
DEST_DATASET = "plantcad/synthetic-genomes-plantcad2-c4096"
DNA_TOKENS = ["a", "c", "g", "t"]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_slurm_rank_info():
    rank = int(os.environ.get("PMIX_RANK", 0))
    world_size = int(os.environ.get("SLURM_NNODES", 1))
    return rank, world_size


def get_dna_token_ids(tokenizer):
    dna_token_ids = []
    for token in DNA_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            raise ValueError(f"Token '{token}' not found")
        dna_token_ids.append(token_id)
    logger.info(f"DNA tokens: {dict(zip(DNA_TOKENS, dna_token_ids))}")
    return dna_token_ids


def create_random_model(tokenizer, device, base_model: str):
    logger.info(f"Loading config from {base_model}")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.vocab_size = len(tokenizer)
    logger.info(f"Creating random model (vocab_size={config.vocab_size})")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_params / 1e9:.2f}B")
    return model


def save_batch(sequences: list, output_dir: Path, rank: int, batch_idx: int):
    rank_dir = output_dir / f"rank_{rank:04d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    path = rank_dir / f"batch_{batch_idx:06d}.parquet"
    df = pd.DataFrame({"text": sequences})
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(sequences)} sequences to {path}")


def generate_for_rank(model, tokenizer, dna_token_ids, split_counts: dict, batch_size: int, device, output_dir: Path, rank: int, world_size: int, seed: int):
    dna_token_tensor = torch.tensor(dna_token_ids, device=device)
    vocab_size = len(tokenizer)
    dna_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    dna_mask[dna_token_tensor] = True

    for split_name, total_count in split_counts.items():
        # Divide work across ranks
        per_rank = total_count // world_size
        extra = total_count % world_size
        my_count = per_rank + (1 if rank < extra else 0)
        my_start = rank * per_rank + min(rank, extra)

        logger.info(f"[{split_name}] rank {rank}/{world_size}: generating {my_count} sequences (global offset {my_start})")

        # Seed per split+rank for reproducibility
        seed_everything(seed + hash(split_name) % 10000 + rank * 1000)

        split_dir = output_dir / BATCHES_SUBDIR / split_name
        rank_dir = split_dir / f"rank_{rank:04d}"
        existing = sorted(rank_dir.glob("batch_*.parquet")) if rank_dir.exists() else []
        
        # Resume from existing batches
        batch_idx = len(existing)
        generated = batch_idx * batch_size * FLUSH_EVERY
        if generated >= my_count:
            logger.info(f"[{split_name}] rank {rank}: already complete ({len(existing)} batches), skipping")
            continue
        if existing:
            logger.info(f"[{split_name}] rank {rank}: resuming from batch {batch_idx} ({generated} sequences done)")
        
        buffer = []

        while generated < my_count:
            current_batch_size = min(batch_size, my_count - generated)
            random_indices = torch.randint(0, len(dna_token_ids), (current_batch_size, CONTEXT_LENGTH), device=device)
            input_ids = dna_token_tensor[random_indices]

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
                masked_logits = logits.clone()
                masked_logits[:, :, ~dna_mask] = float("-inf")
                flat_logits = masked_logits.view(-1, vocab_size)
                probs = torch.softmax(flat_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                output_ids = sampled_tokens.view(current_batch_size, CONTEXT_LENGTH)

            for i in range(current_batch_size):
                seq_ids = output_ids[i].tolist()
                chars = tokenizer.convert_ids_to_tokens(seq_ids)
                seq = "".join(chars).upper()
                assert len(seq) == CONTEXT_LENGTH, f"Expected {CONTEXT_LENGTH}, got {len(seq)}"
                buffer.append(seq)
                generated += 1

            if len(buffer) >= batch_size * FLUSH_EVERY:
                save_batch(buffer, split_dir, rank, batch_idx)
                batch_idx += 1
                buffer = []

            if generated % (batch_size * 10) == 0:
                logger.info(f"[{split_name}] rank {rank}: {generated}/{my_count} ({100*generated/my_count:.1f}%)")

        if buffer:
            save_batch(buffer, split_dir, rank, batch_idx)

        logger.info(f"[{split_name}] rank {rank}: done, generated {generated} sequences")


def aggregate_and_upload(output_dir: Path, output_dataset: str, dry_run: bool):
    datasets = {}
    for split_name in ["train", "validation"]:
        split_dir = output_dir / BATCHES_SUBDIR / split_name
        if not split_dir.exists():
            logger.warning(f"Split dir {split_dir} not found, skipping")
            continue

        pattern = str(split_dir / "rank_*" / "batch_*.parquet")
        files = sorted(glob(pattern))
        logger.info(f"[{split_name}] Found {len(files)} batch files")

        sequences = []
        for i, f in enumerate(files):
            df = pd.read_parquet(f)
            sequences.extend(df["text"].tolist())
            if (i + 1) % 100 == 0:
                logger.info(f"[{split_name}] Loaded {i+1}/{len(files)} files, {len(sequences)} sequences")

        logger.info(f"[{split_name}] Total: {len(sequences)} sequences")
        datasets[split_name] = Dataset.from_dict({"text": sequences})

    dataset_dict = DatasetDict(datasets)
    for split_name, ds in dataset_dict.items():
        logger.info(f"{split_name}: {len(ds):,} sequences")
        for i, sample in enumerate(ds.select(range(min(3, len(ds))))):
            seq = sample["text"]
            logger.info(f"  [{i}] {seq[:40]}...{seq[-40:]}")

    if not dry_run:
        parquet_dir = output_dir / "parquet"
        if parquet_dir.exists():
            logger.info(f"Clearing {parquet_dir}")
            shutil.rmtree(parquet_dir)
        parquet_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {parquet_dir}")
        dataset_dict.save_to_disk(str(parquet_dir), max_shard_size="500MB")
        logger.info(f"Uploading to {output_dataset}")
        api = HfApi()
        # Clear existing files in the HF dataset
        existing_files = api.list_repo_files(repo_id=output_dataset, repo_type="dataset")
        if existing_files:
            logger.info(f"Clearing {len(existing_files)} existing files from {output_dataset}")
            for f in existing_files:
                api.delete_file(path_in_repo=f, repo_id=output_dataset, repo_type="dataset")
        dataset_dict.push_to_hub(output_dataset, private=False)
        logger.info("Uploading README.md")
        api.upload_file(
            path_or_fileobj=make_readme(args.model).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=output_dataset,
            repo_type="dataset",
        )
        logger.info("Upload complete")
    else:
        logger.info("Dry run - skipping save and upload")


def make_readme(base_model: str):
    return f"""---
license: apache-2.0
tags:
- biology
- DNA
- genomics
- synthetic
---
# Synthetic Genomes PlantCAD2 ({CONTEXT_LENGTH}bp)

Synthetic DNA sequences from randomly initialized Qwen2 using PlantCAD2 tokenizer.

## Details
- Model: {base_model} (random weights)
- Tokenizer: {TOKENIZER_MODEL}
- Method: Teacher-forced parallel generation (single forward pass)
- Length: {CONTEXT_LENGTH}bp

## Splits
| Split | Count |
|-------|-------|
| train | {TARGET_COUNTS['train']:,} |
| validation | {TARGET_COUNTS['validation']:,} |
"""


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate")
    gen.add_argument("--output_dir", type=str, required=True)
    gen.add_argument("--model", type=str, default=BASE_MODEL, help="Base model for random initialization")
    gen.add_argument("--test", action="store_true")
    gen.add_argument("--splits", type=str, nargs="+", default=["train", "validation"])
    gen.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    agg = subparsers.add_parser("aggregate")
    agg.add_argument("--output_dir", type=str, required=True)
    agg.add_argument("--model", type=str, default=BASE_MODEL, help="Base model (for README)")
    agg.add_argument("--output_dataset", type=str, default=DEST_DATASET)
    agg.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "generate":
        rank, world_size = get_slurm_rank_info()
        logger.info(f"SLURM rank {rank}/{world_size}")

        target_counts = TEST_COUNTS if args.test else TARGET_COUNTS
        split_counts = {s: target_counts[s] for s in args.splits}

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
        dna_token_ids = get_dna_token_ids(tokenizer)
        device = torch.device(args.device)
        model = create_random_model(tokenizer, device, args.model)

        generate_for_rank(
            model=model,
            tokenizer=tokenizer,
            dna_token_ids=dna_token_ids,
            split_counts=split_counts,
            batch_size=args.batch_size,
            device=device,
            output_dir=Path(args.output_dir),
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )

    elif args.command == "aggregate":
        aggregate_and_upload(Path(args.output_dir), args.output_dataset, args.dry_run)


if __name__ == "__main__":
    main()
