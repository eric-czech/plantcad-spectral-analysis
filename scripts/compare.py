"""
Generalized Eigendecomposition (GEP) analysis of DNA sequence activations from two models.

This script:
1. Samples records from the train split of Angiosperm_65_genomes_8192bp
2. Crops sequences to the specified length
3. Extracts hidden states from two DNA language models
4. Computes eigenspectra for each model separately
5. Computes generalized eigendecomposition (GEP) between the two covariance matrices
6. Plots all three spectra (model1, model2, and relative/GEP)
"""

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from scipy import linalg
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


# Default model paths
PLANTCAD_MODEL_PATH = "kuleshov-group/PlantCAD2-Small-l24-d0768"
MARIN_MODEL_PATH = "plantcad/marin_exp1729__pcv1_600m_c512__checkpoints"
MARIN_SUBFOLDER = "local_store/checkpoints/plantcad-train-600m-r16-a1bc43/hf/step-26782"


# =============================================================================
# Generalized Eigendecomposition
# =============================================================================

def compute_relative_spectrum(X_real, X_baseline, reg=0.0):
    """
    Computes Generalized Eigenvalues solving: sigma_real * v = lambda * sigma_baseline * v
    
    Args:
        X_real (np.array): (N, D) activations from the trained model.
        X_baseline (np.array): (N, D) activations from the random/k-mer baseline.
        reg (float): Diagonal regularization to prevent singularities if baseline is low-rank (default: 0.0).

    Returns:
        evals (np.array): Generalized eigenvalues sorted descending.
        evecs (np.array): Generalized eigenvectors (columns).
    """
    
    # 1. Compute Sample Covariance (assuming mean-centering is desired)
    # rowvar=False assumes shape is (Samples, Features)
    sigma_real = np.cov(X_real, rowvar=False)
    sigma_baseline = np.cov(X_baseline, rowvar=False)
    
    # 2. Regularize if requested (Whitening safety)
    # Can be useful for stability in the "tail" where eigenvalues approach 0
    if reg > 0:
        dim = sigma_real.shape[0]
        jitter = np.eye(dim) * reg
        sigma_real += jitter
        sigma_baseline += jitter
    
    # 3. Solve Generalized Hermitian Eigenvalue Problem
    # Uses 'b' parameter: A*v = lambda*B*v
    # Returns eigenvalues in ascending order
    evals, evecs = linalg.eigh(sigma_real, b=sigma_baseline)
    
    # 4. Sort Descending (Signal-to-Noise Ratio)
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


def compute_standard_spectrum(X, reg=0.0):
    """
    Computes standard eigendecomposition of the covariance matrix.
    
    Args:
        X (np.array): (N, D) activations from a model.
        reg (float): Diagonal regularization for numerical stability (default: 0.0).
    
    Returns:
        evals (np.array): Eigenvalues sorted descending.
        evecs (np.array): Eigenvectors (columns).
    """
    # Compute covariance matrix (assumes mean-centering)
    sigma = np.cov(X, rowvar=False)
    
    # Regularize if requested
    if reg > 0:
        dim = sigma.shape[0]
        jitter = np.eye(dim) * reg
        sigma += jitter
    
    # Compute eigendecomposition
    evals, evecs = linalg.eigh(sigma)
    
    # Sort descending
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]


# =============================================================================
# Model Loading
# =============================================================================

def _get_hidden_size(config) -> int:
    """Get hidden size from model config (handles d_model or hidden_size)."""
    if hasattr(config, 'd_model'):
        return config.d_model
    elif hasattr(config, 'hidden_size'):
        return config.hidden_size
    else:
        raise AttributeError(f"Config has neither 'd_model' nor 'hidden_size': {config}")


def load_model_and_tokenizer(
    model_path: str,
    model_type: str,
    subfolder: str = "",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    random_init: bool = False,
) -> tuple:
    """Load a model and tokenizer.
    
    Args:
        model_path: HuggingFace model path
        model_type: "masked" for AutoModelForMaskedLM, "causal" for AutoModelForCausalLM
        subfolder: Optional subfolder within model_path
        device: Device for inference
        dtype: Data type for model weights
        random_init: If True, use random weights (architecture from config only)
        
    Returns:
        Tuple of (model, tokenizer, is_plantcad)
    """
    if model_type == "masked":
        model_class = AutoModelForMaskedLM
        is_plantcad = True
    elif model_type == "causal":
        model_class = AutoModelForCausalLM
        is_plantcad = False
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'masked' or 'causal'")
    
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
    
    return model, tokenizer, is_plantcad


def extract_model_embeddings(
    sequences: list[str],
    model,
    tokenizer: AutoTokenizer,
    is_plantcad: bool,
    is_causal: bool,
    batch_size: int = 64,
    device: str = "cuda",
    pooling_method: str = "mean",
) -> np.ndarray:
    """
    Extract hidden states from a DNA language model.
    
    For PlantCAD models (Caduceus): hidden_dim = 2 * d_model, we extract
    only the forward half. For other models: use full hidden_dim.
    
    Args:
        sequences: List of DNA sequences
        model: Model (AutoModelForMaskedLM or AutoModelForCausalLM)
        tokenizer: Tokenizer
        is_plantcad: Whether model is PlantCAD (extract forward half only)
        is_causal: Whether model is causal (affects token selection for single_token pooling)
        batch_size: Batch size for inference
        device: Device for inference
        pooling_method: How to pool sequence positions:
            - "single_token": Center token for MaskedLM, last token for CausalLM
            - "mean": Average over all positions
            - "max": Max-pool over all positions
        
    Returns:
        Array of shape (n_samples, output_dim) with embeddings
    """
    # Get expected hidden size from config
    if is_plantcad:
        d_model = model.config.d_model
        expected_hidden_dim = 2 * d_model  # PlantCAD (Caduceus): forward + backward
        output_dim = d_model
    else:
        output_dim = _get_hidden_size(model.config)
        expected_hidden_dim = output_dim
    
    embeddings = []
    n_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Extracting embeddings"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(sequences))
        batch_seqs = sequences[batch_start:batch_end]
        batch_size_actual = len(batch_seqs)
        
        # Tokenize batch
        input_ids = torch.tensor(
            [tokenizer.encode(seq, add_special_tokens=False) for seq in batch_seqs],
            dtype=torch.long,
            device=device,
        )
        
        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            seq_len = hidden_states.shape[1]
            hidden_dim = hidden_states.shape[2]
            
            assert hidden_dim == expected_hidden_dim, (
                f"Expected hidden_dim={expected_hidden_dim}, got {hidden_dim}"
            )
            
            # For PlantCAD models: extract forward representation only
            if is_plantcad:
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


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_data(
    n_samples: int,
    seq_len: int,
    seed: int = 42,
    split: str = "train",
) -> list[str]:
    """
    Load and sample raw sequences from the dataset.
    
    Args:
        n_samples: Number of samples to draw
        seq_len: Length to crop sequences to
        seed: Random seed for reproducibility
        split: Dataset split to use ("train", "validation", or "test")
        
    Returns:
        List of DNA sequences
    """
    print(f"Loading dataset from HuggingFace (split={split})...")
    dataset = load_dataset(
        "plantcad/Angiosperm_65_genomes_8192bp",
        split=split,
        revision="4a444fff5520b992aa978d92a5af509a81977098"
    )
    
    print(f"Dataset size: {len(dataset):,} records")
    
    # Check if we have enough samples
    if n_samples > len(dataset):
        raise ValueError(
            f"Requested n_samples={n_samples:,} but only {len(dataset):,} records available"
        )
    
    print(f"Sampling {n_samples} records...")
    
    # Shuffle and select samples
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    
    # Extract sequences
    print("Extracting sequences...")
    sequences = [dataset[int(idx)]['seq'][:seq_len] for idx in indices]
    
    return sequences


# =============================================================================
# Analysis & Results
# =============================================================================

@dataclass
class GEPResult:
    """Results from generalized eigendecomposition analysis."""
    n_samples: int
    seq_len: int
    seed: int
    pooling_method: str
    split: str
    # Activations
    X1: np.ndarray
    X2: np.ndarray
    # Individual spectra
    evals1: np.ndarray
    evecs1: np.ndarray
    evals2: np.ndarray
    evecs2: np.ndarray
    # Generalized spectrum
    gep_evals: np.ndarray
    gep_evecs: np.ndarray
    # Model info
    model1_path: str
    model1_subfolder: str
    model2_path: str
    model2_subfolder: str


def run_gep_analysis(
    model1_path: str,
    model1_subfolder: str,
    model1_type: str,
    model1_random_init: bool,
    model2_path: str,
    model2_subfolder: str,
    model2_type: str,
    model2_random_init: bool,
    n_samples: int,
    seq_len: int,
    seed: int,
    batch_size: int,
    device: str,
    pooling_method: str,
    split: str,
    reg: float = 0.0,
) -> GEPResult:
    """
    Run generalized eigendecomposition analysis on two models.
    
    Args:
        model1_path: Path to first model
        model1_subfolder: Subfolder for first model
        model1_type: Type of first model ("masked" or "causal")
        model1_random_init: If True, use random weights for model 1
        model2_path: Path to second model
        model2_subfolder: Subfolder for second model
        model2_type: Type of second model ("masked" or "causal")
        model2_random_init: If True, use random weights for model 2
        n_samples: Number of samples to analyze
        seq_len: Sequence length
        seed: Random seed
        batch_size: Batch size for model inference
        device: Device for model inference
        pooling_method: Pooling method for embeddings
        split: Dataset split to use
        reg: Regularization parameter for eigendecomposition (default: 0.0, no regularization)
        
    Returns:
        GEPResult with all eigendecomposition results
    """
    print("=" * 60)
    print("Generalized Eigendecomposition Analysis of DNA Sequences")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  - Model 1: {model1_path}")
    if model1_subfolder:
        print(f"    Subfolder: {model1_subfolder}")
    print(f"    Type: {model1_type}")
    print(f"    Random init: {model1_random_init}")
    print(f"  - Model 2: {model2_path}")
    if model2_subfolder:
        print(f"    Subfolder: {model2_subfolder}")
    print(f"    Type: {model2_type}")
    print(f"    Random init: {model2_random_init}")
    print(f"  - Sample size: {n_samples}")
    print(f"  - Sequence length: {seq_len} bp")
    print(f"  - Pooling method: {pooling_method}")
    print(f"  - Split: {split}")
    print(f"  - Regularization: {reg}")
    print()
    
    # Load data
    sequences = load_raw_data(
        n_samples=n_samples,
        seq_len=seq_len,
        seed=seed,
        split=split,
    )
    
    # Load model 1 and extract embeddings
    print("\n" + "=" * 60)
    print("MODEL 1")
    print("=" * 60)
    model1, tokenizer1, is_plantcad1 = load_model_and_tokenizer(
        model_path=model1_path,
        model_type=model1_type,
        subfolder=model1_subfolder,
        device=device,
        random_init=model1_random_init,
    )
    X1 = extract_model_embeddings(
        sequences=sequences,
        model=model1,
        tokenizer=tokenizer1,
        is_plantcad=is_plantcad1,
        is_causal=(model1_type == "causal"),
        batch_size=batch_size,
        device=device,
        pooling_method=pooling_method,
    )
    print(f"  Embedding matrix 1 shape: {X1.shape}")
    del model1
    torch.cuda.empty_cache()
    
    # Load model 2 and extract embeddings
    print("\n" + "=" * 60)
    print("MODEL 2")
    print("=" * 60)
    model2, tokenizer2, is_plantcad2 = load_model_and_tokenizer(
        model_path=model2_path,
        model_type=model2_type,
        subfolder=model2_subfolder,
        device=device,
        random_init=model2_random_init,
    )
    X2 = extract_model_embeddings(
        sequences=sequences,
        model=model2,
        tokenizer=tokenizer2,
        is_plantcad=is_plantcad2,
        is_causal=(model2_type == "causal"),
        batch_size=batch_size,
        device=device,
        pooling_method=pooling_method,
    )
    print(f"  Embedding matrix 2 shape: {X2.shape}")
    del model2
    torch.cuda.empty_cache()
    
    # Compute individual eigenspectra
    print("\n" + "=" * 60)
    print("COMPUTING EIGENSPECTRA")
    print("=" * 60)
    print("Computing standard eigendecomposition for Model 1...")
    evals1, evecs1 = compute_standard_spectrum(X1, reg=reg)
    print(f"  Computed {len(evals1)} eigenvalues")
    print(f"  Top 10 eigenvalues: {evals1[:10]}")
    
    print("\nComputing standard eigendecomposition for Model 2...")
    evals2, evecs2 = compute_standard_spectrum(X2, reg=reg)
    print(f"  Computed {len(evals2)} eigenvalues")
    print(f"  Top 10 eigenvalues: {evals2[:10]}")
    
    # Compute generalized eigendecomposition
    print("\nComputing generalized eigendecomposition (Model 1 vs Model 2)...")
    gep_evals, gep_evecs = compute_relative_spectrum(X1, X2, reg=reg)
    print(f"  Computed {len(gep_evals)} generalized eigenvalues")
    print(f"  Top 10 generalized eigenvalues: {gep_evals[:10]}")
    print(f"  Bottom 10 generalized eigenvalues: {gep_evals[-10:]}")
    
    return GEPResult(
        n_samples=n_samples,
        seq_len=seq_len,
        seed=seed,
        pooling_method=pooling_method,
        split=split,
        X1=X1,
        X2=X2,
        evals1=evals1,
        evecs1=evecs1,
        evals2=evals2,
        evecs2=evecs2,
        gep_evals=gep_evals,
        gep_evecs=gep_evecs,
        model1_path=model1_path,
        model1_subfolder=model1_subfolder,
        model2_path=model2_path,
        model2_subfolder=model2_subfolder,
    )


# =============================================================================
# Checkpointing
# =============================================================================

def get_checkpoint_path(
    output_dir: str,
    model1_name: str,
    model2_name: str,
    n_samples: int,
    seq_len: int,
    pooling_method: str,
    split: str,
) -> str:
    """Generate checkpoint filename."""
    split_suffix = "" if split == "train" else f"_{split}"
    return os.path.join(
        output_dir,
        f"gep_{model1_name}_vs_{model2_name}_n{n_samples}_l{seq_len}_p{pooling_method}{split_suffix}.pkl"
    )


def save_checkpoint(result: GEPResult, path: str):
    """Save analysis results to disk."""
    print(f"\nSaving checkpoint to: {path}")
    with open(path, 'wb') as f:
        pickle.dump(result, f)


def load_checkpoint(path: str) -> GEPResult:
    """Load analysis results from disk."""
    print(f"\nLoading checkpoint from: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Plotting
# =============================================================================

def plot_gep_spectra(
    result: GEPResult,
    output_path: str = None,
    model1_label: str = "Model 1",
    model2_label: str = "Model 2",
):
    """
    Plot the three eigenspectra: model1, model2, and generalized.
    
    Creates a 3-panel figure showing:
    1. Model 1 eigenspectrum
    2. Model 2 eigenspectrum
    3. Generalized eigenspectrum (Model 1 vs Model 2)
    
    Args:
        result: GEPResult with all eigenvalues
        output_path: Path to save the figure (if None, not saved)
        model1_label: Label for model 1 in plots
        model2_label: Label for model 2 in plots
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Model 1 eigenspectrum
    ax = axes[0]
    ranks1 = np.arange(1, len(result.evals1) + 1)
    ax.loglog(ranks1, result.evals1, 'o-', color='steelblue', linewidth=1.5, markersize=3)
    ax.set_xlabel('Eigenvalue Rank')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'{model1_label} Eigenspectrum (n={result.n_samples})')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Model 2 eigenspectrum
    ax = axes[1]
    ranks2 = np.arange(1, len(result.evals2) + 1)
    ax.loglog(ranks2, result.evals2, 'o-', color='coral', linewidth=1.5, markersize=3)
    ax.set_xlabel('Eigenvalue Rank')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'{model2_label} Eigenspectrum (n={result.n_samples})')
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Generalized eigenspectrum
    ax = axes[2]
    ranks_gep = np.arange(1, len(result.gep_evals) + 1)
    ax.loglog(ranks_gep, result.gep_evals, 'o-', color='forestgreen', linewidth=1.5, markersize=3)
    # Add reference line at y=1
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='λ = 1')
    ax.set_xlabel('Eigenvalue Rank')
    ax.set_ylabel('Generalized Eigenvalue')
    ax.set_title(f'Generalized Spectrum ({model1_label} vs {model2_label})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if output_path:
        # Save PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {output_path}")
        
        # Save PDF at 300 dpi
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"PDF saved to: {pdf_path}")
        
        # Export data to CSV
        csv_path = output_path.replace('.png', '.csv')
        rows = []
        for rank, eig1 in enumerate(result.evals1, start=1):
            rows.append({
                'spectrum': 'model1',
                'rank': rank,
                'eigenvalue': eig1,
            })
        for rank, eig2 in enumerate(result.evals2, start=1):
            rows.append({
                'spectrum': 'model2',
                'rank': rank,
                'eigenvalue': eig2,
            })
        for rank, gep_eig in enumerate(result.gep_evals, start=1):
            rows.append({
                'spectrum': 'generalized',
                'rank': rank,
                'eigenvalue': gep_eig,
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Eigenvalue data exported to: {csv_path}")
    
    plt.show()
    
    return fig


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generalized eigendecomposition analysis of DNA sequence activations from two models"
    )
    
    # Model 1 arguments
    parser.add_argument("--model1_path", type=str, default=PLANTCAD_MODEL_PATH,
                        help=f"Path to first model (default: {PLANTCAD_MODEL_PATH})")
    parser.add_argument("--model1_subfolder", type=str, default="",
                        help="Subfolder within first model path")
    parser.add_argument("--model1_type", type=str, choices=["masked", "causal"], default="masked",
                        help="Type of first model: 'masked' or 'causal' (default: masked)")
    parser.add_argument("--model1_random_init", action="store_true",
                        help="Use random initialization for first model (default: False)")
    parser.add_argument("--model1_label", type=str, default="Model 1",
                        help="Label for first model in plots (default: 'Model 1')")
    
    # Model 2 arguments
    parser.add_argument("--model2_path", type=str, default=MARIN_MODEL_PATH,
                        help=f"Path to second model (default: {MARIN_MODEL_PATH})")
    parser.add_argument("--model2_subfolder", type=str, default=MARIN_SUBFOLDER,
                        help=f"Subfolder within second model path (default: {MARIN_SUBFOLDER})")
    parser.add_argument("--model2_type", type=str, choices=["masked", "causal"], default="causal",
                        help="Type of second model: 'masked' or 'causal' (default: causal)")
    parser.add_argument("--model2_random_init", action="store_true",
                        help="Use random initialization for second model (default: False)")
    parser.add_argument("--model2_label", type=str, default="Model 2",
                        help="Label for second model in plots (default: 'Model 2')")
    
    # Data arguments
    parser.add_argument("--n_samples", type=int, default=8192,
                        help="Number of samples to analyze (default: 8192)")
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="Sequence length to crop to (default: 4096)")
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"], default="train",
                        help="Dataset split to use (default: train)")
    
    # Model inference arguments
    parser.add_argument("--pooling_method", type=str, choices=["single_token", "mean", "max"], default="mean",
                        help="Pooling method for embeddings (default: mean)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for model inference (default: 64)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model inference (default: cuda)")
    
    # Analysis arguments
    parser.add_argument("--reg", type=float, default=0.0,
                        help="Regularization parameter for eigendecomposition (default: 0.0, no regularization)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Output directory (default: current directory)")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if checkpoint exists")
    
    return parser.parse_args()


def get_model_name(model_path: str) -> str:
    """Extract a short name from model path for filenames."""
    # Take last part of path
    name = model_path.rstrip('/').split('/')[-1]
    # Truncate if too long
    if len(name) > 30:
        name = name[:27] + "..."
    return name


def get_output_basename(
    model1_name: str,
    model2_name: str,
    n_samples: int,
    seq_len: int,
    pooling_method: str,
    split: str,
) -> str:
    """Generate output filename base."""
    split_suffix = "" if split == "train" else f"_{split}"
    return f"gep_{model1_name}_vs_{model2_name}_n{n_samples}_l{seq_len}_p{pooling_method}{split_suffix}"


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate short names for models
    model1_name = get_model_name(args.model1_path)
    model2_name = get_model_name(args.model2_path)
    
    # Set output paths
    basename = get_output_basename(
        model1_name=model1_name,
        model2_name=model2_name,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        pooling_method=args.pooling_method,
        split=args.split,
    )
    log_path = os.path.join(args.output_dir, f"{basename}.log")
    
    # Run with logging to file
    with TeeLogger(log_path):
        _run_main(args, model1_name, model2_name, basename)
    
    print(f"Log saved to: {log_path}")


def _run_main(args: argparse.Namespace, model1_name: str, model2_name: str, basename: str):
    """Main logic wrapped for logging."""
    # Log run info
    print(f"Run started: {datetime.now().isoformat()}")
    print(f"Arguments: {args}")
    print()
    
    checkpoint_path = get_checkpoint_path(
        output_dir=args.output_dir,
        model1_name=model1_name,
        model2_name=model2_name,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        pooling_method=args.pooling_method,
        split=args.split,
    )
    output_path = os.path.join(args.output_dir, f"{basename}.png")
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_path) and not args.force:
        print(f"Found existing checkpoint: {checkpoint_path}")
        print("Use --force to recompute")
        result = load_checkpoint(checkpoint_path)
    else:
        # Run full analysis
        result = run_gep_analysis(
            model1_path=args.model1_path,
            model1_subfolder=args.model1_subfolder,
            model1_type=args.model1_type,
            model1_random_init=args.model1_random_init,
            model2_path=args.model2_path,
            model2_subfolder=args.model2_subfolder,
            model2_type=args.model2_type,
            model2_random_init=args.model2_random_init,
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
            pooling_method=args.pooling_method,
            split=args.split,
            reg=args.reg,
        )
        # Save checkpoint
        save_checkpoint(result, checkpoint_path)
    
    # Plot eigenspectra
    plot_gep_spectra(
        result=result,
        output_path=output_path,
        model1_label=args.model1_label,
        model2_label=args.model2_label,
    )
    
    print(f"\nRun completed: {datetime.now().isoformat()}")
    print("\nDone!")


if __name__ == "__main__":
    main()

