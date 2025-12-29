import argparse
import os
import random
import numpy as np
import torch
import wandb
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
    DataCollatorForLanguageModeling,
)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_wsd_schedule(optimizer, num_warmup_steps, num_stable_steps, num_decay_steps):
    """
    Warmup-Stable-Decay (WSD) scheduler.
    1. Linear warmup from 0 to peak LR.
    2. Constant peak LR.
    3. Linear decay from peak LR to 0.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        
        # Decay phase
        decay_progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        return max(0.0, 1.0 - decay_progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate_simulated_input(original_sequences, simulated_sequences):
    """
    Validates that simulated sequences maintain the same batch size and individual sequence lengths.
    
    Args:
        original_sequences (list of str): Original input sequences.
        simulated_sequences (list of str): Simulated output sequences.
    
    Raises:
        ValueError: If batch sizes don't match or any sequence length differs.
    """
    if len(original_sequences) != len(simulated_sequences):
        raise ValueError(
            f"Batch size mismatch: original has {len(original_sequences)} sequences, "
            f"but simulated has {len(simulated_sequences)} sequences"
        )
    
    for i, (orig, sim) in enumerate(zip(original_sequences, simulated_sequences, strict=True)):
        if len(orig) != len(sim):
            raise ValueError(
                f"Sequence length mismatch at index {i}: original length {len(orig)}, "
                f"simulated length {len(sim)}"
            )


def simulate_high_homology(sequences, attractor_ratio=0.1, injection_rate=0.3):
    """
    Increases batch homology by pulling sequences toward a few random 'attractor' sequences.
    
    Args:
        sequences (list of str): Input batch.
        attractor_ratio (float): Fraction of batch to use as templates (0.0-1.0).
        injection_rate (float): % of base pairs to overwrite with attractor data (0.0-1.0).
    """
    # 1. Convert to Numpy char array for vectorized editing
    # We use 'S1' (one-byte characters) for efficiency
    batch_size = len(sequences)
    seq_len = len(sequences[0])
    arr = np.array([list(s) for s in sequences], dtype='S1')

    # 2. Select Attractors
    # We randomly pick indices to serve as 'attractors' for the whole batch
    num_attractors = max(1, int(batch_size * attractor_ratio))
    attractor_indices = np.random.choice(batch_size, num_attractors, replace=False)
    attractors = arr[attractor_indices]

    # 3. Assign every sequence to a random attractor
    # (For higher fidelity, you could calculate Hamming distance and pick the closest, 
    # but random assignment creates 'forced' homology well enough for pretraining)
    assigned_attractor_ids = np.random.randint(0, num_attractors, size=batch_size)
    target_templates = attractors[assigned_attractor_ids]

    # 4. Generate Mutation Mask
    # True = overwrite this position with the attractor's nucleotide
    mask = np.random.random((batch_size, seq_len)) < injection_rate

    # 5. Apply Homology Injection
    # Where mask is True, use template; otherwise keep original
    # We exclude the attractors themselves from mutation to preserve the 'centers'
    out_arr = np.where(mask, target_templates, arr)
    out_arr[attractor_indices] = attractors 

    # 6. Decode back to list of strings
    result = [b"".join(row).decode('utf-8') for row in out_arr]
    validate_simulated_input(sequences, result)
    return result


def simulate_low_homology(sequences, k=3):
    """
    Destroys long-range alignment and repeats while preserving local k-mer spectra.
    
    Args:
        sequences (list of str): Input batch.
        k (int): Size of the token to shuffle. k=1 is random shuffle (preserves GC). 
                 k=3 preserves codon/trinucleotide frequencies but destroys genes.
    """
    batch_size = len(sequences)
    seq_len = len(sequences[0])
    
    # Ensure clean divisibility for reshaping
    if seq_len % k != 0:
        trim_len = seq_len - (seq_len % k)
        sequences = [s[:trim_len] for s in sequences]
        seq_len = trim_len

    # 1. Convert to Numpy array
    arr = np.array([list(s) for s in sequences], dtype='S1')

    # 2. Reshape into (Batch, Num_Chunks, k)
    # This treats each k-mer as a single unit
    num_chunks = seq_len // k
    arr_reshaped = arr.reshape(batch_size, num_chunks, k)

    # 3. Generate random permutations for the chunks
    # This creates a unique shuffle order for EACH sequence in the batch
    # argsort on random noise is the fastest way to get random indices
    permutations = np.argsort(np.random.random((batch_size, num_chunks)), axis=1)

    # 4. Apply Shuffle
    # We use advanced indexing to reorder the chunks according to permutations
    # dim expansion is needed to broadcast the permutation index across the k dimension
    expanded_perms = permutations[:, :, np.newaxis].repeat(k, axis=2)
    shuffled_arr = np.take_along_axis(arr_reshaped, expanded_perms, axis=1)

    # 5. Flatten and Decode
    flat_arr = shuffled_arr.reshape(batch_size, -1)
    result = [b"".join(row).decode('utf-8') for row in flat_arr]
    validate_simulated_input(sequences, result)
    return result


def simulate_random(sequences, random_probability=0.5):
    """
    Replaces sequences with entirely random ACTG strings based on a probability.
    
    Args:
        sequences (list of str): Input batch.
        random_probability (float): Probability that each sequence becomes random (0.0-1.0).
    """
    nucleotides = np.array([b'A', b'C', b'T', b'G'], dtype='S1')
    result = []
    
    for seq in sequences:
        if np.random.random() < random_probability:
            # Replace with random sequence of same length
            random_indices = np.random.randint(0, 4, size=len(seq))
            random_seq = b"".join(nucleotides[random_indices]).decode('utf-8')
            result.append(random_seq)
        else:
            # Keep original
            result.append(seq)
    
    validate_simulated_input(sequences, result)
    return result


def save_checkpoint(model, tokenizer, args, global_step=None):
    """
    Saves model and tokenizer checkpoint to disk.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        args: Command-line arguments
        global_step: Optional step number for checkpoint naming
    """
    os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb_run_name:
        save_dir = os.path.join(args.output_dir, args.wandb_run_name)
    else:
        save_dir = os.path.join(
            args.output_dir,
            f"plantcad_qwen_step{global_step if global_step is not None else args.steps}_{args.dtype}",
        )
    
    print(f"Saving Hugging Face checkpoint to {save_dir} ...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Checkpoint saved to {save_dir}")


def parse_args():
    # Settings taken from second config at lowest FLOP budget in:
    # https://github.com/marin-community/marin/issues/2101#issuecomment-3657313935
    parser = argparse.ArgumentParser(description="PlantCAD Causal LM Pretraining (Qwen Architecture)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["bfloat16", "float32", "float64"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5696)
    parser.add_argument("--lr", type=float, default=0.004861)
    parser.add_argument("--beta2", type=float, default=0.994962)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    # WSD proportions (default to 10% warmup, 70% stable, 20% decay)
    parser.add_argument("--warmup_fraction", type=float, default=0.1, help="Fraction of steps for warmup")
    parser.add_argument("--decay_fraction", type=float, default=0.2, help="Fraction of steps for decay")
    parser.add_argument("--eval_fraction", type=float, default=0.333, help="Fraction of steps between evaluations")
    parser.add_argument("--eval_batches", type=int, default=128, help="Number of batches to use for evaluation")
    parser.add_argument("--wandb_project", type=str, default="plantcad-eigenanalysis")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument("--num_tokenizer_workers", type=int, default=96, help="Number of workers for tokenization/cropping map")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulation_mode", type=str, default=None, choices=[None, "low_homology", "high_homology", "random"],
                        help="Apply sequence simulation: None (default), low_homology, high_homology, or random")
    parser.add_argument("--simulation_batch_size", type=int, default=32, help="Batch size for simulation map steps")
    parser.add_argument("--num_simulation_workers", type=int, default=96, help="Number of workers for simulation map")
    parser.add_argument("--simulation_attractor_ratio", type=float, default=0.1, help="Fraction of batch to use as attractors for high_homology mode (0.0-1.0)")
    parser.add_argument("--simulation_injection_rate", type=float, default=0.3, help="Fraction of base pairs to overwrite with attractor data for high_homology mode (0.0-1.0)")
    parser.add_argument("--simulation_k", type=int, default=3, help="K-mer size for shuffling in low_homology mode")
    parser.add_argument("--simulation_random_probability", type=float, default=0.5, help="Probability that each sequence becomes random for random mode (0.0-1.0)")
    parser.add_argument("--no_train", action="store_true", help="Skip training and export untrained model (negative control)")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length in base pairs for training")
    parser.add_argument("--dataset_path", type=str, default="plantcad/Angiosperm_65_genomes_8192bp", help="Path to the Hugging Face dataset")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration name (e.g., 'wikitext-103-raw-v1' for Salesforce/wikitext)")
    parser.add_argument("--dataset_revision", type=str, default="4a444fff5520b992aa978d92a5af509a81977098", help="Revision/commit hash for the dataset")
    parser.add_argument("--text_column", type=str, default="seq", help="Name of the column containing text/sequences (default: 'seq' for DNA, use 'text' for most text datasets)")
    parser.add_argument("--tokenizer_path", type=str, default="kuleshov-group/PlantCAD2-Small-l24-d0768", help="Path to the Hugging Face tokenizer")
    parser.add_argument("--tokenizer_revision", type=str, default=None, help="Revision/commit hash for the tokenizer (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    
    # Precision configuration
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    target_dtype = dtype_map[args.dtype]
    device = torch.device("cuda")
    
    # CRITICAL: Ensures all new tensors (parameters, buffers) are in the target precision
    torch.set_default_dtype(target_dtype)

    # 1. Tokenizer
    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, 
        revision=args.tokenizer_revision,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Qwen Model Configuration
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=4,
        num_attention_heads=3,
        num_key_value_heads=3,
        max_position_embeddings=args.sequence_length,
        use_cache=False,
    )
    print("Initializing Qwen architecture...")
    model = Qwen2ForCausalLM(config).to(device=device, dtype=target_dtype)

    # Early exit for no_train mode
    if args.no_train:
        print("--no_train enabled: Saving untrained model and exiting...")
        save_checkpoint(model, tokenizer, args, global_step=0)
        print("Untrained model saved. Exiting without training.")
        return

    # 3. Dataset (Angiosperm 65 genomes)
    def tokenize_and_crop(examples):
        # Isolate the first sequence_length bp before tokenization
        seqs = [s[:args.sequence_length] for s in examples[args.text_column]]
        return tokenizer(seqs, truncation=True, max_length=args.sequence_length, add_special_tokens=False)

    dataset_config_msg = f" (config: {args.dataset_config})" if args.dataset_config else ""
    print(f"Loading dataset: {args.dataset_path}{dataset_config_msg} (revision {args.dataset_revision[:8] if args.dataset_revision != 'main' else 'main'})...")
    train_ds = load_dataset(args.dataset_path, args.dataset_config, split="train", revision=args.dataset_revision)
    val_ds = load_dataset(args.dataset_path, args.dataset_config, split="validation", revision=args.dataset_revision)

    # Apply simulation if specified
    if args.simulation_mode is not None:
        print(f"Applying {args.simulation_mode} simulation with batch size {args.simulation_batch_size} and {args.num_simulation_workers} workers...")
        
        def apply_simulation(examples):
            sequences = examples[args.text_column]
            
            if args.simulation_mode == "low_homology":
                simulated_seqs = simulate_low_homology(sequences, k=args.simulation_k)
            elif args.simulation_mode == "high_homology":
                simulated_seqs = simulate_high_homology(
                    sequences,
                    attractor_ratio=args.simulation_attractor_ratio,
                    injection_rate=args.simulation_injection_rate
                )
            elif args.simulation_mode == "random":
                simulated_seqs = simulate_random(
                    sequences,
                    random_probability=args.simulation_random_probability
                )
            else:
                simulated_seqs = sequences
            
            return {args.text_column: simulated_seqs}
        
        train_ds = train_ds.map(
            apply_simulation,
            batched=True,
            batch_size=args.simulation_batch_size,
            num_proc=args.num_simulation_workers,
        )
        val_ds = val_ds.map(
            apply_simulation,
            batched=True,
            batch_size=args.simulation_batch_size,
            num_proc=args.num_simulation_workers,
        )

    print(f"Mapping tokenizer and cropping to {args.sequence_length}bp...")
    train_ds = train_ds.map(
        tokenize_and_crop,
        batched=True,
        remove_columns=train_ds.column_names,
        num_proc=args.num_tokenizer_workers,
    )
    val_ds = val_ds.map(
        tokenize_and_crop,
        batched=True,
        remove_columns=val_ds.column_names,
        num_proc=args.num_tokenizer_workers,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Adjust batch size and grad accumulation for fp64 mode
    if args.dtype == "float64":
        train_batch_size = max(1, args.batch_size // 2)
        grad_accum_steps = 2
    else:
        train_batch_size = args.batch_size
        grad_accum_steps = 1

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_ds, batch_size=train_batch_size, collate_fn=data_collator)

    # 4. Optimizer & WSD Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, args.beta2)
    )
    
    # Compute warmup/decay/eval steps from fractions
    warmup_steps = int(args.steps * args.warmup_fraction)
    decay_steps = int(args.steps * args.decay_fraction)
    stable_steps = args.steps - warmup_steps - decay_steps
    eval_steps = max(1, int(args.steps * args.eval_fraction))
    
    if stable_steps < 0:
        raise ValueError(f"Warmup fraction ({args.warmup_fraction}) + Decay fraction ({args.decay_fraction}) exceed 1.0.")
    
    print(f"LR Schedule: warmup={warmup_steps}, stable={stable_steps}, decay={decay_steps}")
    print(f"Eval every {eval_steps} steps")
    
    scheduler = get_wsd_schedule(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_stable_steps=stable_steps, 
        num_decay_steps=decay_steps
    )

    # 5. Weights & Biases
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # 6. Training Loop
    model.train()
    global_step = 0  # counts optimizer steps
    train_iter = iter(train_loader)
    accum_counter = 0
    
    pbar = tqdm(total=args.steps, desc="Training")
    optimizer.zero_grad()
    
    while global_step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass (no autocast)
        outputs = model(**batch)
        loss = outputs.loss
        
        # Gradient accumulation (scaled loss)
        (loss / grad_accum_steps).backward()
        accum_counter += 1

        if accum_counter % grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            pbar.update(1)
            
            if global_step % 10 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            # Periodic Validation (on optimizer step boundary)
            if global_step % eval_steps == 0 or global_step == args.steps:
                model.eval()
                val_loss = 0
                val_batches_counted = 0
                with torch.no_grad():
                    for i, v_batch in enumerate(val_loader):
                        if i >= args.eval_batches:
                            break 
                        v_batch = {k: v.to(device) for k, v in v_batch.items()}
                        v_loss = model(**v_batch).loss
                        val_loss += v_loss.item()
                        val_batches_counted += 1
                
                avg_val_loss = val_loss / max(1, val_batches_counted)
                wandb.log({"val/loss": avg_val_loss}, step=global_step)
                print(f"Step {global_step} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f}")
                model.train()

    # 7. Final Checkpoint (Hugging Face compatible)
    save_checkpoint(model, tokenizer, args, global_step=global_step)
    print("Pretraining complete.")
    wandb.finish()

if __name__ == "__main__":
    main()

