"""
Eigenspectrum analysis of DNA sequences from PlantCAD dataset.

This script:
1. Samples records from the train split of Angiosperm_65_genomes_8192bp
2. Crops sequences to the specified length
3. Converts sequences to features via:
   - "sequence": one-hot encoding (5 channels: A, C, G, T, N)
   - "plantcad": hidden states from PlantCAD2 model (forward direction, pooled via single_token/mean/max)
   - "marin": hidden states from Marin model (pooled via single_token/mean/max)
4. Centers the data and runs PCA for each sample size
5. Saves aggregated results to disk (checkpointing to avoid recomputation)
6. Plots the eigenspectrum colored by sample size
7. Trains a LightGBM classifier to predict species from top PCs
"""

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from numba import njit, prange
from sklearn.decomposition import PCA
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


# Disable tokenizers parallelism to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TeeLogger:
    """Tee stdout/stderr to a file while preserving console output."""
    
    def __init__(self, log_path: str):
        self.log_file = open(log_path, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self.log_file.close()
        return False
    
    def write(self, message):
        self._original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self._original_stdout.flush()
        self.log_file.flush()


N_BASES = 5  # A, C, G, T, N/other

# Dinucleotides known to vary significantly across plant genomes
DINUCLEOTIDES = ['CG', 'GC', 'AT', 'TA', 'AA', 'TT']

# Default model paths
PLANTCAD_MODEL_PATH = "kuleshov-group/PlantCAD2-Small-l24-d0768"
MARIN_MODEL_PATH = "plantcad/marin_exp1729__pcv1_600m_c512__checkpoints"
DEFAULT_DTYPE = torch.bfloat16

# Default dataset
DEFAULT_DATASET_PATH = "plantcad/Angiosperm_65_genomes_8192bp"
DEFAULT_DATASET_REVISION = "4a444fff5520b992aa978d92a5af509a81977098"

# Sentinel value for unknown species
UNKNOWN_SPECIES = "UNKNOWN"

# Type aliases
PoolingMethod = Literal["single_token", "mean", "max"]
TokenizationMode = Literal["strict", "lenient"]
FeatureSource = Literal["sequence", "plantcad", "plantcad_rand", "marin", "marin_rand"]


# =============================================================================
# Sequence Feature Extraction
# =============================================================================

def compute_gc_content(seq: str) -> float:
    """Compute GC content (fraction of G+C bases)."""
    seq_upper = seq.upper()
    gc = seq_upper.count('G') + seq_upper.count('C')
    valid = sum(1 for c in seq_upper if c in 'ACGT')
    return gc / valid if valid > 0 else 0.0


def compute_repeat_fraction(seq: str) -> float:
    """Compute fraction of lowercase (soft-masked repeat) bases."""
    lowercase = sum(1 for c in seq if c.islower())
    return lowercase / len(seq) if len(seq) > 0 else 0.0


def compute_dinucleotide_freq(seq: str, dinuc: str) -> float:
    """Compute frequency of a specific dinucleotide."""
    seq_upper = seq.upper()
    count = sum(1 for i in range(len(seq_upper) - 1) if seq_upper[i:i+2] == dinuc)
    return count / (len(seq_upper) - 1) if len(seq_upper) > 1 else 0.0


def compute_kmer_entropy(seq: str, k: int = 4) -> float:
    """Compute Shannon entropy of k-mer frequency distribution."""
    seq_upper = seq.upper()
    # Only count valid k-mers (no N)
    kmers = [seq_upper[i:i+k] for i in range(len(seq_upper) - k + 1)
             if 'N' not in seq_upper[i:i+k]]
    if not kmers:
        return 0.0
    counts = Counter(kmers)
    freqs = np.array(list(counts.values()), dtype=np.float64)
    freqs /= freqs.sum()
    return entropy(freqs, base=2)


def _compute_single_sequence_features(seq: str) -> dict[str, float]:
    """Compute all features for a single sequence (for multiprocessing)."""
    result = {
        'gc_content': compute_gc_content(seq),
        'repeat_fraction': compute_repeat_fraction(seq),
        'kmer_entropy_1': compute_kmer_entropy(seq, k=1),
        'kmer_entropy_3': compute_kmer_entropy(seq, k=3),
        'kmer_entropy_9': compute_kmer_entropy(seq, k=9),
    }
    for dinuc in DINUCLEOTIDES:
        result[f'dinuc_{dinuc}'] = compute_dinucleotide_freq(seq, dinuc)
    return result


def extract_sequence_features(sequences: list[str], n_workers: int = 32) -> dict[str, np.ndarray]:
    """Extract all sequence-level features for analysis using multiprocessing.
    
    Raises:
        ValueError: If sequences is empty
    """
    if len(sequences) == 0:
        raise ValueError("sequences cannot be empty")
    
    n = len(sequences)
    
    # Parallel extraction with progress bar
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_compute_single_sequence_features, sequences),
            total=n,
            desc="Extracting sequence features",
        ))
    
    # Aggregate results into arrays
    feature_names = list(results[0].keys())
    features = {name: np.array([r[name] for r in results]) for name in feature_names}
    
    return features


def discretize_feature(values: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, list[str]]:
    """Discretize continuous values into quantile-based bins."""
    # Use quantile-based binning for balanced classes
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(values, percentiles)
    # Handle duplicate edges by adding small epsilon
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.array([values.min(), values.max()])
    
    labels = np.digitize(values, bin_edges[1:-1])  # bins 0 to n_bins-1
    
    # Create readable bin labels
    bin_names = []
    for i in range(len(bin_edges) - 1):
        bin_names.append(f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}")
    
    return labels, bin_names


@dataclass
class PCAResult:
    """Results from a single PCA run."""
    n_samples: int
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    components: np.ndarray  # PCA components


@dataclass
class AnalysisResults:
    """Aggregated results from all PCA runs."""
    source: FeatureSource
    seq_len: int
    seed: int
    pca_results: list[PCAResult]
    # Data for largest sample size (for classification)
    max_data_matrix: np.ndarray
    max_sequences: list[str]
    max_species: list[str]
    max_pca: PCA


@njit(parallel=True)
def encode_sequences_to_onehot(
    seq_bytes: np.ndarray, n_samples: int, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled function to encode sequences to dense one-hot matrix.
    
    Encoding:
        - Uppercase A (65), C (67), G (71), T (84): channels 0-3
        - Everything else: channel 4 (hard masked as N)
    
    Args:
        seq_bytes: 2D array of shape (n_samples, seq_len) with ASCII byte values
        n_samples: Number of samples
        seq_len: Sequence length
        
    Returns:
        Tuple of (data_matrix, softmasked_counts)
    """
    n_features = seq_len * N_BASES
    data_matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    softmasked_counts = np.zeros(n_samples, dtype=np.int64)
    
    for i in prange(n_samples):
        for j in range(seq_len):
            byte = seq_bytes[i, j]
            
            # Map byte to channel
            if byte == 65:  # 'A'
                channel = 0
            elif byte == 67:  # 'C'
                channel = 1
            elif byte == 71:  # 'G'
                channel = 2
            elif byte == 84:  # 'T'
                channel = 3
            else:
                channel = 4
                # Count lowercase (softmasked): a-z = 97-122
                if 97 <= byte <= 122:
                    softmasked_counts[i] += 1
            
            data_matrix[i, j * N_BASES + channel] = 1.0
    
    return data_matrix, softmasked_counts


def _get_hidden_size(config) -> int:
    """Get hidden size from model config (handles d_model or hidden_size)."""
    if hasattr(config, 'd_model'):
        return config.d_model
    elif hasattr(config, 'hidden_size'):
        return config.hidden_size
    else:
        raise AttributeError(f"Config has neither 'd_model' nor 'hidden_size': {config}")


def _load_model_and_tokenizer(
    model_path: str,
    model_class: type,
    device: str,
    dtype: torch.dtype,
    random_init: bool,
    subfolder: str,
) -> tuple:
    """Shared helper to load a model and tokenizer.
    
    Args:
        model_path: HuggingFace model path
        model_class: AutoModelForMaskedLM or AutoModelForCausalLM
        device: Device for inference
        dtype: Data type for model weights
        random_init: If True, use random weights (architecture from config only)
        subfolder: Optional subfolder within model_path
    """
    class_name = model_class.__name__
    init_str = " (random init)" if random_init else ""
    sub_str = f" subfolder={subfolder}" if subfolder else ""
    print(f"Loading {class_name} from {model_path}{sub_str}{init_str} with {dtype=}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, subfolder=subfolder, trust_remote_code=True
    )
    if random_init:
        config = AutoConfig.from_pretrained(
            model_path, subfolder=subfolder, trust_remote_code=True
        )
        model = model_class.from_config(config, torch_dtype=dtype, trust_remote_code=True)
        model = model.to(device, dtype=dtype)
    else:
        model = model_class.from_pretrained(
            model_path,
            subfolder=subfolder,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device, dtype=dtype)
    model.eval()
    
    # Print model details
    n_params = sum(p.numel() for p in model.parameters())
    hidden_size = _get_hidden_size(model.config)
    print(f"  Hidden size: {hidden_size}")
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"  Device: {device}, Dtype: {model.dtype}")
    print(f"  Model:\n{model}")
    
    return model, tokenizer


def load_plantcad_model(
    model_path: str | None,
    subfolder: str,
    device: str,
    dtype: torch.dtype,
    random_init: bool,
) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """Load PlantCAD model and tokenizer."""
    if model_path is None:
        model_path = PLANTCAD_MODEL_PATH
    return _load_model_and_tokenizer(
        model_path, AutoModelForMaskedLM, device, dtype, random_init, subfolder
    )


def load_marin_model(
    model_path: str | None,
    subfolder: str,
    device: str,
    dtype: torch.dtype,
    random_init: bool,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Marin model and tokenizer."""
    if model_path is None:
        model_path = MARIN_MODEL_PATH
    return _load_model_and_tokenizer(
        model_path, AutoModelForCausalLM, device, dtype, random_init, subfolder
    )


def tokenize_sequences(
    sequences: list[str],
    tokenizer: AutoTokenizer,
    sequence_length: int,
    mode: TokenizationMode,
) -> torch.Tensor:
    """Tokenize DNA sequences with validation.
    
    Args:
        sequences: List of DNA sequences to tokenize
        tokenizer: HuggingFace tokenizer
        sequence_length: Expected token length for all sequences
        mode: Tokenization mode
            - "strict": No padding/truncation, validate sequences are exact length
            - "lenient": Pad/truncate to sequence_length
            
    Returns:
        torch.Tensor of input_ids with shape (len(sequences), sequence_length)
        
    Raises:
        ValueError: If mode is invalid or validation fails
    """
    # Tokenize based on mode
    if mode == "strict":
        # Tokenize w/o padding or truncation;
        # implicitly requires that all sequences are already of the target length
        inputs = tokenizer(sequences, truncation=False, padding=False, return_length=True)
    elif mode == "lenient":
        # Tokenize with padding and truncation to target length
        inputs = tokenizer(
            sequences,
            max_length=sequence_length,
            truncation=True,
            padding="max_length",
            return_length=True
        )
    else:
        raise ValueError(f"Invalid tokenization mode: {mode}. Must be 'strict' or 'lenient'")
    
    # Validate that all sequences have expected length
    for idx, length in enumerate(inputs["length"]):
        if length != sequence_length:
            example_seq = sequences[idx]
            status = "too long" if length > sequence_length else "too short"
            raise ValueError(
                f"Tokenization validation failed in {mode} mode: "
                f"Sequence at batch position {idx} is {status} "
                f"(expected {sequence_length} tokens, got {length} tokens). "
                f"Example sequence (first 100 chars): {example_seq[:100]}"
            )
    
    return inputs["input_ids"]


def extract_model_embeddings(
    sequences: list[str],
    model,
    tokenizer: AutoTokenizer,
    source: FeatureSource,
    sequence_length: int,
    batch_size: int,
    device: str,
    pooling_method: PoolingMethod,
    tokenization_mode: TokenizationMode,
) -> np.ndarray:
    """
    Extract hidden states from a DNA language model.
    
    For PlantCAD (Caduceus, bidirectional): hidden_dim = 2 * d_model, we extract
    only the forward half. For other models (e.g., marin): use full hidden_dim.
    
    Args:
        sequences: List of DNA sequences
        model: Model (AutoModelForMaskedLM or AutoModelForCausalLM)
        tokenizer: Tokenizer
        source: Model source ("plantcad", "plantcad_rand", "marin", "marin_rand")
        sequence_length: Expected sequence length for validation
        batch_size: Batch size for inference
        device: Device for inference
        pooling_method: How to pool sequence positions:
            - "single_token": Center token for MaskedLM, last token for CausalLM
            - "mean": Average over all positions
            - "max": Max-pool over all positions
        tokenization_mode: "strict" or "lenient"
            - "strict": No padding/truncation, validate all sequences are exact length
            - "lenient": Pad/truncate to sequence_length
        
    Returns:
        Array of shape (n_samples, output_dim) with embeddings
    """
    is_causal = isinstance(model, AutoModelForCausalLM)
    slice_hidden = source.startswith("plantcad")  # Bidirectional model: slice forward half
    
    # Get expected hidden size from config
    if slice_hidden:
        d_model = model.config.d_model
        expected_hidden_dim = 2 * d_model  # Bidirectional: forward + backward
        output_dim = d_model
    else:
        output_dim = model.config.hidden_size
        expected_hidden_dim = output_dim
    
    embeddings = []
    n_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Extracting embeddings"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(sequences))
        batch_seqs = sequences[batch_start:batch_end]
        batch_size_actual = len(batch_seqs)
        
        # Tokenize batch with validation
        input_ids_list = tokenize_sequences(batch_seqs, tokenizer, sequence_length, tokenization_mode)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        
        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            seq_len = hidden_states.shape[1]
            hidden_dim = hidden_states.shape[2]
            
            assert hidden_dim == expected_hidden_dim, (
                f"Expected hidden_dim={expected_hidden_dim}, got {hidden_dim}"
            )
            
            # For plantcad (bidirectional): extract forward representation only
            if slice_hidden:
                states = hidden_states[:, :, :d_model]
            else:
                states = hidden_states
            
            if pooling_method == "single_token":
                if is_causal:
                    pooled = states[:, -1, :]  # Last token for causal
                else:
                    center_idx = (seq_len - 1) // 2
                    pooled = states[:, center_idx, :]  # Center token for masked
            elif pooling_method == "mean":
                pooled = states.mean(dim=1)
            elif pooling_method == "max":
                pooled = states.max(dim=1).values
            else:
                raise ValueError(f"Unknown pooling_method: {pooling_method}")
            
            assert pooled.shape == (batch_size_actual, output_dim), (
                f"Expected pooled shape ({batch_size_actual}, {output_dim}), got {pooled.shape}"
            )
            
            embeddings.append(pooled.to(torch.float32).cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)


def extract_onehot_features(sequences: list[str], seq_len: int) -> np.ndarray:
    """
    Extract one-hot encoded features from DNA sequences.
    
    Args:
        sequences: List of DNA sequences (lowercase = softmasked, will be hard masked as N)
        seq_len: Expected sequence length
        
    Returns:
        Array of shape (n_samples, seq_len * N_BASES) with one-hot encoding
        
    Raises:
        ValueError: If any sequence length doesn't match seq_len
    """
    n_samples = len(sequences)
    if n_samples == 0:
        raise ValueError("sequences cannot be empty")
    
    # Validate sequence lengths before processing
    for i, seq in enumerate(sequences):
        if len(seq) != seq_len:
            raise ValueError(
                f"Sequence {i} has length {len(seq)}, expected {seq_len}. "
                f"Ensure all sequences are at least seq_len bp before cropping."
            )
    
    seq_bytes = np.empty((n_samples, seq_len), dtype=np.uint8)
    for i, seq in enumerate(sequences):
        seq_bytes[i] = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    
    print("Converting to one-hot encoding (numba JIT)...")
    data_matrix, softmasked_counts = encode_sequences_to_onehot(seq_bytes, n_samples, seq_len)
    
    total_softmasked = softmasked_counts.sum()
    total_bases = n_samples * seq_len
    softmasked_pct = 100 * total_softmasked / total_bases
    print(f"  Softmasked (hard masked as N): {total_softmasked:,} / {total_bases:,} ({softmasked_pct:.1f}%)")
    print(f"  Matrix shape: {data_matrix.shape}")
    
    # Compute histogram of encoded bases
    base_names = ['A', 'C', 'G', 'T', 'N']
    base_counts = np.zeros(N_BASES, dtype=np.int64)
    for channel in range(N_BASES):
        base_counts[channel] = data_matrix[:, channel::N_BASES].sum()
    
    print("  Base counts:")
    for name, count in zip(base_names, base_counts, strict=True):
        pct = 100 * count / base_counts.sum()
        print(f"    {name}: {int(count):,} ({pct:.2f}%)")
    
    return data_matrix


def filter_species_data(dataset, species_filter: set[str], species_column: str):
    """
    Filter dataset to a set of species.
    
    Args:
        dataset: HuggingFace dataset
        species_filter: Set of species names (assemblies) to filter to
        species_column: Name of the column containing species/assembly identifiers
        
    Returns:
        Filtered dataset
    """
    species_list = sorted(species_filter)
    if len(species_list) == 1:
        print(f"\nFiltering data to species: {species_list[0]}")
    else:
        print(f"\nFiltering data to {len(species_list)} species:")
        if len(species_list) <= 10:
            print(f"  {', '.join(species_list)}")
        else:
            print(f"  {', '.join(species_list[:10])} ... (+{len(species_list)-10} more)")
    print(f"  Records before filtering: {len(dataset):,}")
    
    # Use HuggingFace's filter method (much faster than manual iteration)
    filtered_dataset = dataset.filter(
        lambda x: x.get(species_column, UNKNOWN_SPECIES) in species_filter,
        desc=f"Filtering to {len(species_filter)} species",
        num_proc=mp.cpu_count(),
    )
    
    print(f"  Records after filtering: {len(filtered_dataset):,}")
    
    if len(filtered_dataset) == 0:
        # Get all available species for error message (sample first 10k for speed)
        sample_size = min(10000, len(dataset))
        sample_indices = list(range(sample_size))
        available_species = sorted(set(dataset[i].get(species_column, UNKNOWN_SPECIES) for i in sample_indices))
        raise ValueError(
            f"No records found for species: {species_list}. "
            f"Available species (sample): {available_species[:20]}"
        )
    
    return filtered_dataset


def parse_species_filter_args(species_filter: str | None, species_filter_file: str | None) -> set[str] | None:
    """
    Parse species filter arguments and return a deduplicated, sorted set of species.
    
    Args:
        species_filter: Single species name from CLI
        species_filter_file: Path to file containing species names (one per line)
        
    Returns:
        Set of species names to filter to, or None if no filtering requested
        
    Raises:
        ValueError: If both arguments are provided
    """
    if species_filter is not None and species_filter_file is not None:
        raise ValueError(
            "Cannot specify both --species_filter and --species_filter_file. "
            "Please use only one species filtering method."
        )
    
    if species_filter is not None:
        return {species_filter}
    
    if species_filter_file is not None:
        print(f"Reading species list from: {species_filter_file}")
        with open(species_filter_file, 'r') as f:
            # Read lines, strip whitespace, filter out empty lines and comments
            species_list = [
                line.strip() 
                for line in f 
                if line.strip() and not line.strip().startswith('#')
            ]
        
        if not species_list:
            raise ValueError(f"No species found in file: {species_filter_file}")
        
        # Deduplicate and sort
        species_set = set(species_list)
        print(f"  Loaded {len(species_list)} species names ({len(species_set)} unique)")
        return species_set
    
    return None


def prepare_sequences(dataset, seq_len: int, text_column: str):
    """
    Prepare sequences by cropping and filtering out empty ones.
    
    Args:
        dataset: HuggingFace dataset
        seq_len: Length to crop sequences to
        text_column: Name of the column containing text/sequences
        
    Returns:
        Filtered dataset with cropped, non-empty sequences
        
    Raises:
        ValueError: If all sequences are empty
    """
    print(f"\nPreparing sequences (cropping to {seq_len} bp and filtering empty)...")
    print(f"  Records before filtering: {len(dataset):,}")
    
    # Crop and filter out empty sequences
    def crop_and_update(record):
        record[text_column] = record.get(text_column, '')[:seq_len]
        return record
    
    dataset = dataset.map(crop_and_update, desc="Cropping sequences", num_proc=mp.cpu_count())
    
    filtered_dataset = dataset.filter(
        lambda x: len(x.get(text_column, '')) > 0,
        desc="Filtering empty sequences",
        num_proc=mp.cpu_count(),
    )
    
    n_empty = len(dataset) - len(filtered_dataset)
    print(f"  Records after filtering: {len(filtered_dataset):,}")
    print(f"  Empty sequences filtered: {n_empty:,}")
    
    if len(filtered_dataset) == 0:
        raise ValueError(
            f"All {len(dataset):,} sequences are empty after cropping to {seq_len} bp"
        )
    
    return filtered_dataset


def load_raw_data(
    n_samples: int,
    seq_len: int,
    seed: int,
    species_filter: set[str] | None,
    split: str,
    dataset_path: str,
    dataset_config: str,
    dataset_revision: str,
    text_column: str,
    dataset_sample_size: int,
    species_column: str,
) -> tuple[list[str], list[str]]:
    """
    Load and sample raw sequences from the dataset.
    
    Args:
        n_samples: Number of samples to draw (should be max of all sample sizes)
        seq_len: Length to crop sequences to
        seed: Random seed for reproducibility
        species_filter: Optional set of species names (assemblies) to filter to before sampling
        split: Dataset split to use ("train", "validation", or "test")
        dataset_path: HuggingFace dataset path
        dataset_config: Optional dataset configuration name
        dataset_revision: Git revision/commit hash for the dataset
        text_column: Name of the column containing text/sequences
        dataset_sample_size: If provided, use streaming and take only this many records
        species_column: Name of the column containing species/assembly identifiers
        
    Returns:
        Tuple of (sequences, species_labels) where species_labels are from the
        species_column field (e.g., genome assembly names used as species identifiers)
        
    Raises:
        ValueError: If not enough non-empty sequences are available
    """
    dataset_config_msg = f" (config: {dataset_config})" if dataset_config else ""
    
    if dataset_sample_size is not None:
        print(f"Loading dataset from HuggingFace with streaming (path={dataset_path}{dataset_config_msg}, revision={dataset_revision}, split={split})...")
        print(f"  Taking first {dataset_sample_size:,} records from stream...")
        dataset_stream = load_dataset(dataset_path, dataset_config, split=split, revision=dataset_revision, streaming=True)
        # Materialize the first dataset_sample_size records into a normal dataset
        dataset = Dataset.from_generator(lambda: iter(dataset_stream.take(dataset_sample_size)))
        print(f"  Materialized {len(dataset):,} records from stream")
    else:
        print(f"Loading dataset from HuggingFace (path={dataset_path}{dataset_config_msg}, revision={dataset_revision}, split={split})...")
        dataset = load_dataset(dataset_path, dataset_config, split=split, revision=dataset_revision)
    
    print(f"Dataset size: {len(dataset):,} records")
    
    # Filter by species if requested
    if species_filter is not None:
        dataset = filter_species_data(dataset, species_filter, species_column)
    
    # Crop sequences and filter out empty ones
    dataset = prepare_sequences(dataset, seq_len, text_column)
    
    # Check if we have enough samples
    if n_samples > len(dataset):
        species_msg = ""
        if species_filter:
            species_list = sorted(species_filter)
            if len(species_list) == 1:
                species_msg = f" to species {species_list[0]}"
            else:
                species_msg = f" to {len(species_list)} species"
        raise ValueError(
            f"Requested n_samples={n_samples:,} but only {len(dataset):,} non-empty sequences available "
            f"after filtering{species_msg}"
        )
    
    print(f"Sampling {n_samples} records...")
    
    # Shuffle and select samples
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    
    # Extract records (keeping sequence and species together)
    print("Extracting records...")
    records = [dataset[int(idx)] for idx in indices]
    sequences = [r[text_column] for r in records]
    species = [r.get(species_column, UNKNOWN_SPECIES) for r in records]
    
    print(f"  Unique species: {len(set(species))}")
    
    return sequences, species


def convert_to_features(
    sequences: list[str],
    source: FeatureSource,
    seq_len: int,
    batch_size: int,
    device: str,
    pooling_method: PoolingMethod,
    model_path: str | None,
    model_subfolder: str,
    tokenization_mode: TokenizationMode,
) -> np.ndarray:
    """
    Convert raw sequences to feature matrix.
    
    Args:
        sequences: List of DNA sequences
        source: "sequence" for one-hot encoding, "plantcad"/"plantcad_rand" for PlantCAD,
                "marin"/"marin_rand" for Marin model embeddings
        seq_len: Sequence length
        batch_size: Batch size for model inference
        device: Device for model inference
        pooling_method: Pooling method for model embeddings
        model_path: Custom model path (overrides default for plantcad/marin sources)
        model_subfolder: Custom model subfolder (overrides default for plantcad/marin sources)
        tokenization_mode: "strict" or "lenient" for tokenization behavior
        
    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    if source == "sequence":
        return extract_onehot_features(sequences, seq_len)
    elif source in ("plantcad", "plantcad_rand"):
        random_init = source == "plantcad_rand"
        model, tokenizer = load_plantcad_model(
            model_path=model_path,
            subfolder=model_subfolder,
            device=device,
            dtype=DEFAULT_DTYPE,
            random_init=random_init,
        )
    elif source in ("marin", "marin_rand"):
        random_init = source == "marin_rand"
        model, tokenizer = load_marin_model(
            model_path=model_path,
            subfolder=model_subfolder,
            device=device,
            dtype=DEFAULT_DTYPE,
            random_init=random_init,
        )
    else:
        raise ValueError(f"Unknown source: {source}")
    
    data_matrix = extract_model_embeddings(
        sequences=sequences,
        model=model,
        tokenizer=tokenizer,
        source=source,
        batch_size=batch_size,
        device=device,
        pooling_method=pooling_method,
        tokenization_mode=tokenization_mode,
        sequence_length=seq_len,
    )
    print(f"  Embedding matrix shape: {data_matrix.shape}")
    del model
    torch.cuda.empty_cache()
    return data_matrix


def compute_pca(data_matrix: np.ndarray, n_samples: int) -> PCAResult:
    """
    Compute PCA on a subset of the data.
    
    Args:
        data_matrix: Full data matrix (will use first n_samples rows)
        n_samples: Number of samples to use
        
    Returns:
        PCAResult with eigenvalues and components
    """
    print(f"\nComputing PCA for n_samples={n_samples}...")
    
    subset = data_matrix[:n_samples]
    n_features = subset.shape[1]
    
    n_components = min(n_samples, n_features)
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(subset)
    
    print(f"  Computed {len(pca.explained_variance_)} eigenvalues")
    print(f"  Top 10 eigenvalues: {pca.explained_variance_[:10]}")
    print(f"  Variance explained by top 10: {pca.explained_variance_ratio_[:10].sum():.2%}")
    
    return PCAResult(
        n_samples=n_samples,
        eigenvalues=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        components=pca.components_,
    )


def run_analysis(
    n_samples_list: list[int],
    args: argparse.Namespace,
) -> AnalysisResults:
    """
    Run PCA analysis for all sample sizes.
    
    Fetches data for max sample size, then runs PCA on increasing subsets.
    
    Args:
        n_samples_list: List of sample sizes (will be sorted ascending)
        args: Command-line arguments containing all configuration parameters
        
    Returns:
        AnalysisResults with all PCA results and data for largest sample size
    """
    n_samples_list = sorted(n_samples_list)
    max_samples = max(n_samples_list)
    
    print("=" * 60)
    print("Eigenspectrum Analysis of DNA Sequences")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  - Source: {args.source}")
    print(f"  - Sample sizes: {n_samples_list}")
    print(f"  - Sequence length: {args.seq_len} bp")
    if args.source == "sequence":
        print(f"  - Feature dimension: {args.seq_len * N_BASES} (one-hot with {N_BASES} channels)")
    else:
        print(f"  - Feature dimension: model hidden size")
        print(f"  - Pooling method: {args.pooling_method}")
        if args.source.endswith("_rand"):
            print(f"  - Random initialization: True")
    print()
    
    # Parse species filter arguments
    species_filter_set = parse_species_filter_args(args.species_filter, args.species_filter_file)
    
    # Load raw data for max sample size (with optional species filtering)
    sequences, species = load_raw_data(
        n_samples=max_samples,
        seq_len=args.seq_len,
        seed=args.seed,
        species_filter=species_filter_set,
        split=args.split,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        text_column=args.text_column,
        dataset_sample_size=args.dataset_sample_size,
        species_column=args.species_column,
    )
    
    # Convert to features
    data_matrix = convert_to_features(
        sequences=sequences,
        source=args.source,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        pooling_method=args.pooling_method,
        model_path=args.model_path,
        model_subfolder=args.model_subfolder,
        tokenization_mode=args.tokenization_mode,
    )
    
    # Run PCA for each sample size (ascending order)
    pca_results = []
    for n in n_samples_list:
        result = compute_pca(data_matrix=data_matrix, n_samples=n)
        pca_results.append(result)
    
    # Get full PCA for max sample size (for classification)
    print(f"\nFitting full PCA for classification (n={max_samples})...")
    max_pca = PCA(n_components=min(max_samples, data_matrix.shape[1]), svd_solver='full')
    max_pca.fit(data_matrix)
    
    return AnalysisResults(
        source=args.source,
        seq_len=args.seq_len,
        seed=args.seed,
        pca_results=pca_results,
        max_data_matrix=data_matrix,
        max_sequences=sequences,
        max_species=species,
        max_pca=max_pca,
    )


SOURCE_ABBREVS = {
    "sequence": "seq",
    "plantcad": "pcad",
    "plantcad_rand": "pcadrand",
    "marin": "marin",
    "marin_rand": "marinrand",
}
MODEL_SOURCES = [s for s in SOURCE_ABBREVS if s != "sequence"]


def get_checkpoint_path(output_dir: str, source: FeatureSource, seq_len: int, n_samples_list: list[int], pooling_method: PoolingMethod, split: str) -> str:
    """Generate checkpoint filename."""
    src_abbrev = SOURCE_ABBREVS[source]
    samples_str = "_".join(str(n) for n in sorted(n_samples_list))
    pool_suffix = f"_p{pooling_method}" if source in MODEL_SOURCES else ""
    split_suffix = "" if split == "train" else f"_{split}"
    return os.path.join(output_dir, f"eigenspectrum_s{src_abbrev}_l{seq_len}_n{samples_str}{pool_suffix}{split_suffix}.pkl")


def save_checkpoint(results: AnalysisResults, path: str):
    """Save analysis results to disk."""
    print(f"\nSaving checkpoint to: {path}")
    with open(path, 'wb') as f:
        pickle.dump(results, f)


def load_checkpoint(path: str) -> AnalysisResults:
    """Load analysis results from disk."""
    print(f"\nLoading checkpoint from: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_effective_sample_size(eigenvalues: np.ndarray) -> float:
    """
    Compute effective sample size from eigenvalues.
    
    Defined as: (sum of eigenvalues)^2 / sum of squared eigenvalues
    This is related to the "participation ratio" and measures how many
    eigenvalues effectively contribute to the total variance.
    
    Args:
        eigenvalues: Array of eigenvalues
        
    Returns:
        Effective sample size (float)
    """
    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()
    if sum_eig_sq == 0:
        return 0.0
    return (sum_eig ** 2) / sum_eig_sq


def compute_condition_number(eigenvalues: np.ndarray) -> float:
    """
    Compute condition number from eigenvalues.
    
    Defined as: max(eigenvalue) / min(eigenvalue)
    
    Args:
        eigenvalues: Array of eigenvalues
        
    Returns:
        Condition number (float)
    """
    # Filter out zero eigenvalues for condition number
    nonzero_eigs = eigenvalues[eigenvalues > 0]
    n_zero = len(eigenvalues) - len(nonzero_eigs)
    if n_zero > 0:
        warnings.warn(f"Found {n_zero} zero-valued eigenvalues out of {len(eigenvalues)} total")
    if len(nonzero_eigs) == 0:
        return float('inf')
    return nonzero_eigs.max() / nonzero_eigs.min()


def plot_eigenspectrum(
    pca_results: list[PCAResult],
    output_path: str = None,
    min_n_samples: int = None,
):
    """
    Plot the eigenspectrum with lines colored by sample size.
    
    Creates a 6-panel figure (2 rows x 3 cols) showing:
    Row 1:
        1. Eigenvalue vs rank (log-log scale, all)
        2. Eigenvalue vs rank (log-log scale, top 100)
        3. Eigenvalue vs rank (log-log scale, top 10)
    Row 2:
        4. Cumulative explained variance
        5. Effective sample size vs actual sample size
        6. Condition number vs actual sample size
    
    Args:
        pca_results: List of PCAResult objects (one per sample size)
        output_path: Path to save the figure (if None, not saved)
        min_n_samples: If provided, filter results to only include sample sizes >= this value
        
    Returns:
        matplotlib Figure object
    """
    # Filter results by min_n_samples if specified
    if min_n_samples is not None:
        pca_results = [r for r in pca_results if r.n_samples >= min_n_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Get colormap
    n_results = len(pca_results)
    cmap = plt.get_cmap('viridis', n_results)
    
    # Collect summary stats for each sample size
    sample_sizes = []
    effective_sizes = []
    condition_numbers = []
    
    for idx, result in enumerate(pca_results):
        eigenvalues = result.eigenvalues
        explained_variance_ratio = result.explained_variance_ratio
        ranks = np.arange(1, len(eigenvalues) + 1)
        color = cmap(idx)
        label = f"n={result.n_samples}"
        
        # Compute summary statistics
        eff_size = compute_effective_sample_size(eigenvalues)
        cond_num = compute_condition_number(eigenvalues)
        sample_sizes.append(result.n_samples)
        effective_sizes.append(eff_size)
        condition_numbers.append(cond_num)
        
        # Row 1, Plot 1: Eigenvalue vs Rank (log-log scale, all)
        ax = axes[0, 0]
        ax.loglog(ranks, eigenvalues, color=color, linewidth=0.8, label=label)
        
        # Row 1, Plot 2: Eigenvalue vs Rank (log-log scale, top 100)
        ax = axes[0, 1]
        n_top = min(100, len(eigenvalues))
        ax.loglog(ranks[:n_top], eigenvalues[:n_top], color=color, linewidth=0.8, label=label)
        
        # Row 1, Plot 3: Eigenvalue vs Rank (log-log scale, top 10)
        ax = axes[0, 2]
        n_top = min(10, len(eigenvalues))
        ax.loglog(ranks[:n_top], eigenvalues[:n_top], color=color, linewidth=0.8, label=label)
        
        # Row 2, Plot 1: Cumulative explained variance
        ax = axes[1, 0]
        cumulative_var = np.cumsum(explained_variance_ratio)
        ax.plot(ranks, cumulative_var, color=color, linewidth=1.0, label=label)
    
    # Configure Row 1 axes (eigenspectrum plots)
    axes[0, 0].set_xlabel('Eigenvalue Rank')
    axes[0, 0].set_ylabel('Eigenvalue')
    axes[0, 0].set_title('Eigenspectrum (Log-Log, All)')
    axes[0, 0].grid(True, alpha=0.3, which='both', axis='x')
    axes[0, 0].legend(loc='upper right', fontsize=7)
    
    axes[0, 1].set_xlabel('Eigenvalue Rank')
    axes[0, 1].set_ylabel('Eigenvalue')
    axes[0, 1].set_title('Eigenspectrum (Log-Log, Top 100)')
    axes[0, 1].grid(True, alpha=0.3, which='both', axis='x')
    axes[0, 1].legend(loc='upper right', fontsize=7)
    
    axes[0, 2].set_xlabel('Eigenvalue Rank')
    axes[0, 2].set_ylabel('Eigenvalue')
    axes[0, 2].set_title('Eigenspectrum (Log-Log, Top 10)')
    axes[0, 2].grid(True, alpha=0.3, which='both', axis='x')
    axes[0, 2].legend(loc='upper right', fontsize=7)
    
    # Configure Row 2 axes
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Cumulative Explained Variance')
    axes[1, 0].set_title('Cumulative Variance')
    axes[1, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[1, 0].axhline(y=0.95, color='g', linestyle='--', alpha=0.5, linewidth=0.8)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(loc='lower right', fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Helper to format sample size labels
    def format_size(n):
        if n >= 1_000_000:
            return f'{n//1_000_000}M'
        elif n >= 1_000:
            return f'{n//1_000}K'
        return str(int(n))
    
    log2_samples = np.log2(sample_sizes)
    
    # Row 2, Plot 2: Effective sample size vs actual sample size (log10 scale)
    ax = axes[1, 1]
    ax.plot(log2_samples, effective_sizes, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Effective Sample Size')
    ax.set_title(r'Effective Sample Size [$(\Sigma\lambda)^2 / \Sigma\lambda^2$]')
    ax.set_xticks(log2_samples)
    ax.set_xticklabels([format_size(n) for n in sample_sizes], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, which='both')
    
    # Row 2, Plot 3: Condition number vs actual sample size (log10 scale)
    ax = axes[1, 2]
    ax.plot(log2_samples, condition_numbers, 'o-', color='coral', linewidth=2, markersize=8)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Condition Number')
    ax.set_title(r'Condition Number [$\lambda_{max} / \lambda_{min}$]')
    ax.set_xticks(log2_samples)
    ax.set_xticklabels([format_size(n) for n in sample_sizes], rotation=45, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # Print variance thresholds for largest sample size
    largest = pca_results[-1]
    cumulative_var = np.cumsum(largest.explained_variance_ratio)
    n_90 = np.searchsorted(cumulative_var, 0.9) + 1
    n_95 = np.searchsorted(cumulative_var, 0.95) + 1
    print(f"\nFor n={largest.n_samples}:")
    print(f"  Components needed for 90% variance: {n_90}")
    print(f"  Components needed for 95% variance: {n_95}")
    print(f"  Effective sample size: {effective_sizes[-1]:.8f}")
    print(f"  Condition number: {condition_numbers[-1]:.8e}")
    
    plt.tight_layout()
    
    if output_path:
        # Save PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        
        # Save PDF at 300 dpi
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"PDF saved to: {pdf_path}")
        
        # Export eigenvalue data needed to recreate plots (one row per eigenvalue)
        csv_path = output_path.replace('.png', '.csv')
        rows = []
        for result in pca_results:
            cumulative_var = np.cumsum(result.explained_variance_ratio)
            for rank, (eig, cum_var) in enumerate(
                zip(result.eigenvalues, cumulative_var, strict=True),
                start=1
            ):
                rows.append({
                    'n_samples': result.n_samples,
                    'rank': rank,
                    'eigenvalue': eig,
                    'cumulative_variance': cum_var,
                })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Eigenvalue data exported to: {csv_path}")
    
    plt.show()
    
    return fig


@dataclass
class ClassificationResult:
    """Results from a classification task."""
    target_name: str
    n_components: int
    n_classes: int
    accuracy: float
    f1_macro: float
    class_names: list[str]
    feature_importances: np.ndarray  # Per-PC importance


def filter_rare_classes(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    min_samples_per_class: int = 2,
    target_name: str = None,
    log_info: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Filter out classes with insufficient samples.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Encoded labels (integers)
        class_names: List of class names
        min_samples_per_class: Minimum samples required per class
        target_name: Name for logging (optional)
        log_info: Whether to print filtering summary
        
    Returns:
        Tuple of (X_filtered, y_filtered, n_classes_filtered)
        
    Raises:
        ValueError: If no samples remain after filtering
    """
    n_classes_orig = len(np.unique(y))
    n_samples_orig = len(y)
    
    # Filter classes with insufficient samples
    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= min_samples_per_class]
    mask = np.isin(y, valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    if len(y_filtered) == 0:
        raise ValueError(f"No samples remain after filtering classes with <{min_samples_per_class} samples")
    
    # Log filtering summary if requested
    if log_info:
        kept_names = [class_names[i] for i in valid_classes]
        if len(kept_names) > 10:
            names_display = ", ".join(kept_names[:10]) + f", ... (+{len(kept_names)-10} more)"
        else:
            names_display = ", ".join(kept_names)
        
        prefix = f"[{target_name}] " if target_name else ""
        if len(valid_classes) < n_classes_orig:
            print(f"{prefix}Filtered: {len(valid_classes)}/{n_classes_orig} classes, {len(y_filtered)}/{n_samples_orig} samples")
        else:
            print(f"{prefix}{n_classes_orig} classes, {n_samples_orig} samples")
        print(f"  Classes: {names_display}")
    
    return X_filtered, y_filtered, len(valid_classes)


def train_classifier(
    X_pca: np.ndarray,
    labels: np.ndarray | list[str],
    target_name: str,
    class_names: list[str] = None,
    seed: int = 42,
    test_size: float = 0.2,
    min_samples_per_class: int = 2,
    verbose: bool = True,
    log_class_info: bool = False,
) -> ClassificationResult:
    """
    Train a LightGBM classifier on PC features for any target.
    
    Args:
        X_pca: PC-projected features (n_samples, n_components)
        labels: Target labels (integers or strings)
        target_name: Name of target variable for logging
        class_names: Optional class names for display
        seed: Random seed
        test_size: Test fraction
        min_samples_per_class: Minimum samples per class (filters rare classes)
        verbose: Print detailed results
        log_class_info: Print class summary and filtering info
        
    Returns:
        ClassificationResult with metrics and feature importances
        
    Raises:
        ValueError: If labels is empty or no samples remain after filtering
    """
    if len(labels) == 0:
        raise ValueError(f"labels cannot be empty for target '{target_name}'")
    
    # Encode labels if strings
    if isinstance(labels[0], str):
        le = LabelEncoder()
        y = le.fit_transform(labels)
        class_names = class_names or list(le.classes_)
    else:
        y = np.asarray(labels)
        class_names = class_names or [str(i) for i in sorted(np.unique(y))]
    
    n_components = X_pca.shape[1]
    
    # Filter classes with insufficient samples
    X_pca, y, n_classes = filter_rare_classes(
        X=X_pca,
        y=y,
        class_names=class_names,
        min_samples_per_class=min_samples_per_class,
        target_name=target_name,
        log_info=log_class_info,
    )
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Train model
    model = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    if verbose:
        print(f"\n[{target_name}] n_classes={n_classes}, acc={accuracy:.3f}, f1={f1_macro:.3f}")
    
    return ClassificationResult(
        target_name=target_name,
        n_components=n_components,
        n_classes=n_classes,
        accuracy=accuracy,
        f1_macro=f1_macro,
        class_names=class_names,
        feature_importances=model.feature_importances_,
    )


@dataclass
class PCScalingResult:
    """Results from PC scaling analysis for one target."""
    target_name: str
    n_classes: int
    n_components_list: list[int]
    accuracies: list[float]
    f1_scores: list[float]

def analyze_pc_predictivity(
    pca: PCA,
    data_matrix: np.ndarray,
    sequences: list[str],
    species: list[str],
    n_bins: int = 5,
    seed: int = 42,
    output_dir: str = None,
) -> dict[str, PCScalingResult]:
    """
    Analyze how classification performance scales with number of PCs.
    
    For each target (species, GC, repeats, k-mer entropy, dinucleotides),
    trains classifiers using increasing numbers of PCs to reveal which
    targets are captured by early vs. late components.
    
    Args:
        pca: Fitted PCA object
        data_matrix: Embedding matrix
        sequences: Raw DNA sequences (for feature extraction)
        species: Species labels
        n_bins: Number of bins for discretizing continuous features
        seed: Random seed
        output_dir: Directory to save plots (if None, plots are not saved)
        
    Returns:
        Dict mapping target_name -> PCScalingResult
    """
    n_components_list = [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # Cap at available components (0 is always allowed as random baseline)
    max_available = min(pca.n_components_, data_matrix.shape[0])
    n_components_list = [n for n in n_components_list if n == 0 or n <= max_available]
    
    # Always include max_available at the end if not already present
    if max_available not in n_components_list and max_available > 0:
        n_components_list.append(max_available)
    
    print("\n" + "=" * 70)
    print("PC SCALING ANALYSIS")
    print(f"Testing n_components: {n_components_list}")
    print("=" * 70)
    
    # Project data to max components needed
    max_n = max(n for n in n_components_list if n > 0) if any(n > 0 for n in n_components_list) else 1
    X_pca_full = pca.transform(data_matrix)[:, :max_n]
    
    # Random baseline feature for n_components=0
    rng = np.random.default_rng(seed)
    X_random = rng.random((data_matrix.shape[0], 1))
    
    # Extract sequence features
    seq_features = extract_sequence_features(sequences)
    
    # Build targets dict: name -> (labels, class_names)
    targets = {}
    le = LabelEncoder()
    y_species = le.fit_transform(species)
    targets['species'] = (y_species, list(le.classes_))
    
    for feat_name, values in seq_features.items():
        labels, bin_names = discretize_feature(values, n_bins=n_bins)
        targets[feat_name] = (labels, bin_names)
    
    # Train classifiers for each (target, n_components) combination
    results = {}
    for target_name, (labels, class_names) in targets.items():
        accs, f1s, n_classes_list = [], [], []
        for i, n_comp in enumerate(n_components_list):
            # Use random baseline for n_components=0
            X_subset = X_random if n_comp == 0 else X_pca_full[:, :n_comp]
            res = train_classifier(
                X_pca=X_subset,
                labels=labels,
                target_name=f"{target_name}[{n_comp}]",
                class_names=class_names,
                seed=seed,
                verbose=False,
                log_class_info=(i == 0),  # Only log for first n_comp
            )
            accs.append(res.accuracy)
            f1s.append(res.f1_macro)
            n_classes_list.append(res.n_classes)
        
        # Verify n_classes is consistent across all n_components
        if len(set(n_classes_list)) > 1:
            raise ValueError(f"Inconsistent n_classes for {target_name}: {n_classes_list}")
        
        results[target_name] = PCScalingResult(
            target_name=target_name,
            n_classes=n_classes_list[0],
            n_components_list=n_components_list,
            accuracies=accs,
            f1_scores=f1s,
        )
        # Print compact summary
        f1_str = " ".join(f"{f:.2f}" for f in f1s)
        print(f"  {target_name:<20} F1: {f1_str}")
    
    # Plot scaling curves
    if output_dir:
        plot_pc_scaling_curves(results, output_dir)
    
    return results


def plot_pc_scaling_curves(
    results: dict[str, PCScalingResult],
    output_dir: str,
):
    """
    Plot F1 score vs number of PCs for each target.
    
    Creates a 3-panel figure with:
    - 'kmers' targets (kmer_entropy_1, kmer_entropy_3, kmer_entropy_9)
    - 'dinucleotides' targets
    - 'other' targets (species, GC, repeats)
    Also prints a summary table and exports data to CSV.
    
    Args:
        results: Dict mapping target_name -> PCScalingResult
        output_dir: Directory to save the plot (saved as 'pc_scaling_curves.png')
        
    Returns:
        matplotlib Figure object
    """
    # Group targets by type for cleaner visualization
    groups = {
        'primary_tasks': ['species', 'gc_content', 'repeat_fraction'],
        'kmer_tasks': ['kmer_entropy_1', 'kmer_entropy_3', 'kmer_entropy_9'],
        'dinucleotide_tasks': [k for k in results.keys() if k.startswith('dinuc_')],
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (group_name, target_names) in zip(axes, groups.items(), strict=True):
        # Filter to targets that exist in results
        available_targets = [t for t in target_names if t in results]
        if not available_targets:
            ax.set_title(f'PC Scaling: {group_name.replace("_", " ").title()} (no data)')
            continue
        
        for target_name in available_targets:
            res = results[target_name]
            label = target_name.replace('dinuc_', '').replace('_', ' ').replace('kmer entropy ', 'k=')
            # Use sequential x positions for even spacing, label with actual n_components
            x_positions = list(range(len(res.n_components_list)))
            ax.plot(x_positions, res.f1_scores, 'o-', label=label, markersize=4)
        
        ax.set_xlabel('Number of Principal Components')
        ax.set_ylabel('F1 Score (Macro)')
        ax.set_title(f'PC Scaling: {group_name.replace("_", " ").title()}')
        # Use actual n_components values as tick labels
        n_comp_list = results[available_targets[0]].n_components_list
        ax.set_xticks(range(len(n_comp_list)))
        ax.set_xticklabels([str(n) for n in n_comp_list])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save PNG and PDF
    png_path = os.path.join(output_dir, 'pc_scaling_curves.png')
    pdf_path = os.path.join(output_dir, 'pc_scaling_curves.pdf')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"\nPC scaling curves saved to: {png_path}")
    print(f"PC scaling curves PDF saved to: {pdf_path}")
    
    # Export data to CSV
    csv_path = os.path.join(output_dir, 'pc_scaling_curves.csv')
    rows = []
    for target_name, res in results.items():
        for n_comp, acc, f1 in zip(res.n_components_list, res.accuracies, res.f1_scores, strict=True):
            rows.append({
                'target': target_name,
                'n_classes': res.n_classes,
                'n_components': n_comp,
                'accuracy': acc,
                'f1_macro': f1,
            })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"PC scaling data exported to: {csv_path}")
    
    plt.show()
    
    # Also create summary table
    print("\n" + "=" * 70)
    print("SUMMARY: F1 by n_components")
    print("=" * 70)
    n_list = list(results.values())[0].n_components_list
    header = f"{'Target':<20}" + "".join(f"{n:>8}" for n in n_list)
    print(header)
    print("-" * len(header))
    for name in sorted(results.keys()):
        res = results[name]
        row = f"{name:<20}" + "".join(f"{f:>8.3f}" for f in res.f1_scores)
        print(row)
    
    return fig


def train_species_classifier(
    pca: PCA,
    data_matrix: np.ndarray,
    species: list[str],
    n_components: int = 10,
    output_path: str = None,
    seed: int = 42,
    test_size: float = 0.2,
    min_samples_per_class: int = 2,
):
    """
    Train species classifier and plot confusion matrix.
    
    Args:
        pca: Fitted PCA object
        data_matrix: Embedding matrix (n_samples, n_features)
        species: Species labels (assembly names)
        n_components: Number of PCs to use as features
        output_path: Path to save confusion matrix plot (if None, not saved)
        seed: Random seed for train/test split
        test_size: Fraction of data to use for testing
        min_samples_per_class: Minimum samples per class (filters rare classes)
        
    Returns:
        Tuple of (figure, metrics_dict) where metrics_dict contains 'accuracy' and 'f1_macro'
    """
    print(f"\nTraining species classifier with n_components={n_components} PCs...")
    X_pca = pca.transform(data_matrix)[:, :n_components]
    
    le = LabelEncoder()
    y = le.fit_transform(species)
    class_names = list(le.classes_)
    
    # Filter classes with insufficient samples
    X_pca, y, n_classes = filter_rare_classes(
        X=X_pca,
        y=y,
        class_names=class_names,
        min_samples_per_class=min_samples_per_class,
        target_name="Species",
        log_info=True,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    model = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n[Species] n_classes={n_classes}, acc={accuracy:.3f}, f1={f1_macro:.3f}")
    
    # Confusion matrix plot
    labels_present = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    display_labels = le.classes_[labels_present]
    fig, ax = plt.subplots(figsize=(max(12, len(labels_present) * 0.4), max(10, len(labels_present) * 0.35)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)
    ax.set_title(f'Species Classification (acc={accuracy:.3f}, f1={f1_macro:.3f})')
    plt.tight_layout()
    
    if output_path:
        # Save PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
        
        # Save PDF at 300 dpi
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix PDF saved to: {pdf_path}")
        
        # Export confusion matrix data to CSV
        csv_path = output_path.replace('.png', '.csv')
        # Create a DataFrame with confusion matrix values
        cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
        cm_df.index.name = 'true_label'
        cm_df.to_csv(csv_path)
        print(f"Confusion matrix data exported to: {csv_path}")
        
        # Also export per-class metrics
        metrics_csv_path = output_path.replace('.png', '_metrics.csv')
        per_class_correct = cm.diagonal()
        per_class_total = cm.sum(axis=1)
        per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
        metrics_df = pd.DataFrame({
            'species': display_labels,
            'correct': per_class_correct,
            'total': per_class_total,
            'accuracy': per_class_acc,
        })
        # Add overall metrics as first row
        overall_row = pd.DataFrame([{
            'species': '_OVERALL_',
            'correct': int(per_class_correct.sum()),
            'total': int(per_class_total.sum()),
            'accuracy': accuracy,
        }])
        metrics_df = pd.concat([overall_row, metrics_df], ignore_index=True)
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Classification metrics exported to: {metrics_csv_path}")
    plt.show()
    
    return fig, {"accuracy": accuracy, "f1_macro": f1_macro}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eigenspectrum analysis of DNA sequences")
    parser.add_argument("--source", type=str, choices=list(SOURCE_ABBREVS.keys()), default="sequence",
                        help="Feature source: 'sequence' for one-hot, 'plantcad'/'marin' for pretrained, '*_rand' for random init")
    parser.add_argument("--n_samples", type=int, nargs='+', default=[500, 1000, 2000, 5000],
                        help="List of sample sizes to analyze (PCA computed for each)")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length to crop to")
    parser.add_argument("--pooling_method", type=str, choices=["single_token", "mean", "max"], default="mean",
                        help="Pooling method: 'single_token' (center for MaskedLM, last for CausalLM), 'mean', or 'max'")
    parser.add_argument("--model_path", type=str, default=None,
                        help=f"Model path (default: {PLANTCAD_MODEL_PATH} for plantcad, {MARIN_MODEL_PATH} for marin)")
    parser.add_argument("--model_subfolder", type=str, default="",
                        help="Subfolder within model path (default: empty string)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for model inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model inference")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if checkpoint exists")
    parser.add_argument("--species_filter", type=str, default=None, help="Filter data to a single species (assembly name). Default: None (no filtering)")
    parser.add_argument("--species_filter_file", type=str, default=None, help="Path to file containing species names to filter (one per line). Mutually exclusive with --species_filter. Default: None (no filtering)")
    parser.add_argument("--species_column", type=str, default="assembly",
                        help="Name of the column containing species/assembly identifiers (default: 'assembly')")
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"], default="train",
                        help="Dataset split to use (default: train)")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH,
                        help=f"HuggingFace dataset path (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration name (e.g., 'wikitext-103-raw-v1' for Salesforce/wikitext)")
    parser.add_argument("--dataset_revision", type=str, default=DEFAULT_DATASET_REVISION,
                        help=f"Dataset git revision/commit hash (default: {DEFAULT_DATASET_REVISION})")
    parser.add_argument("--text_column", type=str, default="seq",
                        help="Name of the column containing text/sequences (default: 'seq' for DNA, use 'text' for most text datasets)")
    parser.add_argument("--tokenization_mode", type=str, choices=["strict", "lenient"], default="strict",
                        help="Tokenization mode: 'strict' (no padding/truncation, validate exact length) or 'lenient' (pad/truncate to target length). Default: strict")
    parser.add_argument("--dataset_sample_size", type=int, default=None,
                        help="Number of records to take from dataset using streaming. If None (default), load full dataset normally.")
    return parser.parse_args()


def get_output_basename(source: FeatureSource, n_samples_list: list[int], seq_len: int, pooling_method: PoolingMethod, split: str) -> str:
    """Generate output filename base."""
    src_abbrev = SOURCE_ABBREVS[source]
    max_n = max(n_samples_list)
    pool_suffix = f"_p{pooling_method}" if source in MODEL_SOURCES else ""
    split_suffix = "" if split == "train" else f"_{split}"
    return f"eigenspectrum_s{src_abbrev}_n{max_n}_l{seq_len}{pool_suffix}{split_suffix}"


def main():
    """Main entry point."""
    args = parse_args()
    
    # Sort sample sizes
    n_samples_list = sorted(args.n_samples)
    
    # Set output paths
    os.makedirs(args.output_dir, exist_ok=True)
    basename = get_output_basename(
        source=args.source,
        n_samples_list=n_samples_list,
        seq_len=args.seq_len,
        pooling_method=args.pooling_method,
        split=args.split,
    )
    log_path = os.path.join(args.output_dir, f"{basename}.log")
    
    # Run with logging to file
    with TeeLogger(log_path):
        _run_main(args, n_samples_list, basename)
    
    print(f"Log saved to: {log_path}")


def _run_main(args: argparse.Namespace, n_samples_list: list[int], basename: str):
    """Main logic wrapped for logging."""
    # Log run info
    print(f"Run started: {datetime.now().isoformat()}")
    print(f"Arguments: {args}")
    print()
    
    checkpoint_path = get_checkpoint_path(
        output_dir=args.output_dir,
        source=args.source,
        seq_len=args.seq_len,
        n_samples_list=n_samples_list,
        pooling_method=args.pooling_method,
        split=args.split,
    )
    output_path = os.path.join(args.output_dir, f"{basename}.png")
    confusion_matrix_path = os.path.join(args.output_dir, f"{basename}_confusion_matrix.png")
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_path) and not args.force:
        print(f"Found existing checkpoint: {checkpoint_path}")
        print("Use --force to recompute")
        results = load_checkpoint(checkpoint_path)
    else:
        # Run full analysis
        results = run_analysis(
            n_samples_list=n_samples_list,
            args=args,
        )
        # Save checkpoint
        save_checkpoint(results, checkpoint_path)
    
    # Plot eigenspectrum (all sample sizes, colored by n_samples)
    plot_eigenspectrum(results.pca_results, output_path=output_path)
    
    # Analyze how performance scales with number of PCs
    analyze_pc_predictivity(
        pca=results.max_pca,
        data_matrix=results.max_data_matrix,
        sequences=results.max_sequences,
        species=results.max_species,
        n_bins=5,
        seed=results.seed,
        output_dir=args.output_dir,
    )
    
    # Train species classifier and plot confusion matrix
    # Use all available PCs
    max_available_pcs = min(results.max_pca.n_components_, results.max_data_matrix.shape[0])
    train_species_classifier(
        pca=results.max_pca,
        data_matrix=results.max_data_matrix,
        species=results.max_species,
        n_components=max_available_pcs,
        output_path=confusion_matrix_path,
        seed=results.seed,
    )
    
    print(f"\nRun completed: {datetime.now().isoformat()}")
    print("\nDone!")


if __name__ == "__main__":
    main()
