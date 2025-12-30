# PlantCAD Spectral Analysis Experiments

Abbreviations: 
- SEP = Standard Eigenvalue Problem
- GEP = Generalized Eigenvalue Problem

## Standard Eigenanalysis (SEP)

These experiments all decompose activations from a single model using various initializations, context lengths, sample sizes, and pooling methods.

### Experiment Summary

| Version | Title | Group | Key Features |
|---------|-------|-------|--------------|
| v11 | PlantCAD2-Small validation split | plantcad2_baseline | MLM, 8192bp, validation split |
| v12 | PlantCAD2-Small random init 4096bp | plantcad2_context_ablation | MLM, random_init, 4096bp |
| v13 | PlantCAD2-Small baseline 8192bp | plantcad2_baseline | MLM, trained, full context |
| v14 | PlantCAD2-Small random init 8192bp | plantcad2_baseline | MLM, random_init, full context |
| v15 | PlantCAD2-Small baseline 4096bp | plantcad2_context_ablation | MLM, trained, reduced context |
| v16 | Marin PoC 600M | marin_poc | CLM, 512bp, 1408 hidden dim |
| v17 | IsoFLOP large width, high FLOP | marin_isoflop_large_width | 8.2e16 FLOP, 25M params |
| v18 | IsoFLOP large width, low FLOP | marin_isoflop_large_width | 3.3e16 FLOP, 9.4M params |
| v19 | IsoFLOP small width, low FLOP | marin_isoflop_small_width | 2.0e15 FLOP, 805K params |
| v20 | IsoFLOP small width, smallest | marin_isoflop_small_width | 2.0e15 FLOP, 74.6K params |
| v21 | IsoFLOP small width, high FLOP | marin_isoflop_small_width | 1.0e16 FLOP, 805K params |
| v22 | Marin Qwen pretrain baseline | marin_simulation | simulation_mode=none |
| v23 | Marin Qwen high homology | marin_simulation | simulation_mode=high_homology |
| v24 | Marin Qwen k-mer shuffle | marin_simulation | simulation_mode=low_homology |
| v25 | Marin Qwen random simulation | marin_simulation | simulation_mode=random |
| v26 | Marin Qwen random init | marin_simulation | random_init baseline |
| v27 | Marin Qwen Zmays only | marin_species_filter | single_species=Zmays |
| v28 | Animal promoter dataset | animals_promoter | animals, 512bp, 65 species |
| v29 | Wikitext small pretrain | text_baseline | text, 384 hidden dim |
| v30 | Qwen2-1.5B DCLM-Edu | text_baseline | text, HF model, 1536 hidden dim |

See `results/sep/experiment_schema.json` for the full metadata schema and each `results/sep/v*/experiment.json` for complete experiment metadata.

### DNA/Plants/PlantCAD2

```bash
# PlantCAD2-S at full context length
python scripts/plantcad_eigenanalysis.py --source plantcad --n_samples 128 256 512 1024 4096 16384 65536 262144 --device cuda --output_dir results/sep/v13 --pooling_method mean --batch_size 32 --seq_len 8192 --force

# Random init of PlantCAD2-S at full context length
python scripts/plantcad_eigenanalysis.py --source plantcad_rand --n_samples 128 256 512 1024 4096 16384 65536 262144 --device cuda --output_dir results/sep/v14 --pooling_method mean --batch_size 32 --seq_len 8192 --force

# PlantCAD2-S at 4096bp context length
python scripts/plantcad_eigenanalysis.py --source plantcad --n_samples 128 256 512 1024 4096 16384 65536 262144 --device cuda --output_dir results/sep/v15 --pooling_method mean --batch_size 32 --seq_len 4096 --force

# PlantCAD2-S at 4096bp context length with random init
python scripts/plantcad_eigenanalysis.py --source plantcad_rand --n_samples 128 256 512 1024 4096 16384 65536 262144 --device cuda --output_dir results/sep/v12 --pooling_method mean --batch_size 32 --seq_len 4096 --force

# PlantCAD2-S at full context length using validation split rather than train split
python scripts/plantcad_eigenanalysis.py --source plantcad --n_samples 128 256 512 1024 4096 16384 65536 262144 --device cuda --output_dir results/sep/v11 --pooling_method mean --batch_size 32 --seq_len 8192 --split validation --force
```

### DNA/Plants/Marin

```bash
# ------------------------------------------------------------------------------------------------
# PoC model
# ------------------------------------------------------------------------------------------------
# For PoC 512bp model from https://github.com/marin-community/marin/issues/1729:
python scripts/plantcad_eigenanalysis.py --source marin --n_samples 512 1024 4096 16384 65536 262144 \
  --device cuda --output_dir results/sep/v16 --pooling_method mean \
  --model_path plantcad/marin_exp1729__pcv1_600m_c512__checkpoints \
  --model_subfolder local_store/checkpoints/plantcad-train-600m-r16-a1bc43/hf/step-26782 \
  --batch_size 32 --seq_len 512 --force

# ------------------------------------------------------------------------------------------------
# IsoFLOP models
# ------------------------------------------------------------------------------------------------
# For models used in https://github.com/marin-community/marin/issues/2101

## Large width (original)
### Largest FLOP budget
python scripts/plantcad_eigenanalysis.py --source marin --n_samples 512 1024 4096 16384 65536 262144 \
  --device cuda --output_dir results/sep/v17 \
  --model_path  plantcad/marin_exp2101__pcv2_isoflop_c4096__checkpoints \
  --model_subfolder checkpoints/plantcad_isoflop_v1.0-A_qwen-F8.2e+16-P25M-T810M-E1-56b128/hf/step-6184 \
  --pooling_method mean --batch_size 32 --seq_len 4096 --force 
### Smallest FLOP budget
python scripts/plantcad_eigenanalysis.py --source marin --n_samples 512 1024 4096 16384 65536 262144 \
  --device cuda --output_dir results/sep/v18 \
  --model_path  plantcad/marin_exp2101__pcv2_isoflop_c4096__checkpoints \
  --model_subfolder checkpoints/plantcad_isoflop_v1.0-A_qwen-F3.3e+16-P9.4M-T746M-E1-c98aae/hf \
  --pooling_method mean --batch_size 32 --seq_len 4096 --force

## Small width (tiny models)
### Smallest FLOP budget
python scripts/plantcad_eigenanalysis.py --source marin --n_samples 512 1024 4096 16384 65536 262144 \
  --device cuda --output_dir results/sep/v19 \
  --model_path  plantcad/marin_exp2101__pcv2_isoflop_c4096__checkpoints \
  --model_subfolder checkpoints/plantcad_isoflop_v1.5-A_qwen-F2.0e+15-P805.3K-T207M-E1-96174b/hf/step-6333 \
  --pooling_method mean --batch_size 32 --seq_len 4096 --force
### Smallest FLOP budget + smallest params
python scripts/plantcad_eigenanalysis.py --source marin --n_samples 512 1024 4096 16384 65536 262144 \
  --device cuda --output_dir results/sep/v20 \
  --model_path  plantcad/marin_exp2101__pcv2_isoflop_c4096__checkpoints \
  --model_subfolder checkpoints/plantcad_isoflop_v1.5-A_qwen-F2.0e+15-P74.6K-T1.1B-E1-b7ce45/hf/step-8166 \
  --pooling_method mean --batch_size 32 --seq_len 4096 --force
  
### Largest FLOP budget
python scripts/plantcad_eigenanalysis.py --source marin --n_samples 512 1024 4096 16384 65536 262144 \
  --device cuda --output_dir results/sep/v21 \
  --model_path  plantcad/marin_exp2101__pcv2_isoflop_c4096__checkpoints \
  --model_subfolder checkpoints/plantcad_isoflop_v1.5-A_qwen-F1.0e+16-P805.3K-T1.0B-E1-4d9c41/hf/step-7917 \
  --pooling_method mean --batch_size 32 --seq_len 4096 --force


# --------------------------------------------------------------------------------------------------
# Qwen pre-training w/ simulation
# --------------------------------------------------------------------------------------------------

checkpoints=(
  plantcad_pretrain_fp32_v0.3
  plantcad_pretrain_fp32_sim_ar0.2_ir0.8_v0.3
  plantcad_pretrain_fp32_sim_k3_v0.3
  plantcad_pretrain_fp32_sim_rand_p1.0_v0.3
  plantcad_pretrain_fp32_no_train_v0.3
)

versions=(22 23 24 25 26)

for i in "${!checkpoints[@]}"; do
  checkpoint="${checkpoints[$i]}"
  version="${versions[$i]}"

  python scripts/plantcad_eigenanalysis.py --source marin --n_samples 512 1024 4096 16384 65536 262144 \
    --device cuda --output_dir "results/sep/v${version}" \
    --model_path plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints \
    --model_subfolder "checkpoints/${checkpoint}" \
    --pooling_method mean --batch_size 32 --seq_len 4096 --force
done

# Species filtering experiments, cf. https://github.com/plantcad/plantcad-dev/issues/39
# Athaliana, Zmays, Gmax all show the same patterns, so only Zmays is used:
python scripts/plantcad_eigenanalysis.py --source marin --n_samples 512 1024 4096 8192 16384 32768 \
  --device cuda --output_dir results/sep/v27 \
  --model_path plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints \
  --model_subfolder checkpoints/plantcad_pretrain_fp32_v0.3 \
  --species_filter Zmays \
  --pooling_method mean --batch_size 32 --seq_len 4096 --force
```

### DNA/Animals

Experiments from previously trained or re-trained models on [songlab/gpn-animal-promoter-dataset](https://huggingface.co/datasets/songlab/gpn-animal-promoter-dataset).

Species were mapped to this dataset in [eczech/gpn-animal-promoter-dataset](https://huggingface.co/datasets/eczech/gpn-animal-promoter-dataset) via the pipeline and data in [eric-czech/gpn-animal-promoter-dataset-metadata](https://huggingface.co/datasets/eric-czech/gpn-animal-promoter-dataset-metadata).

```bash

# Fetch 65 most prevalent species to be used for both decomposition and species classification:
curl -o /tmp/gpn_promoter_species_filter.txt https://gist.githubusercontent.com/eric-czech/c2e65ecee9d89c7dc479306ddc895585/raw/4b70e7a511231345569b4598378b7f3c6d3bdae0/species_filter_gpn_promoter.txt

# Run on species subset:
python scripts/plantcad_eigenanalysis.py \
  --source marin --n_samples 512 1024 4096 8192 16384 32768 65536 \
  --device cuda --output_dir results/sep/v28 \
  --dataset_path eczech/gpn-animal-promoter-dataset \
  --dataset_revision f009db443a914d4113922e3028de0666b85c24d6 \
  --species_column organism \
  --species_filter_file /tmp/gpn_promoter_species_filter.txt \
  --model_path plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints \
  --model_subfolder checkpoints/gpn_promoter_pretrain_c512_v0.2 \
  --pooling_method mean --batch_size 512 --seq_len 512

# Run on all species
python scripts/plantcad_eigenanalysis.py \
  --source marin --n_samples 512 1024 4096 8192 16384 32768 65536 \
  --device cuda --output_dir results/sep/v31 \
  --dataset_path eczech/gpn-animal-promoter-dataset \
  --dataset_revision f009db443a914d4113922e3028de0666b85c24d6 \
  --species_column organism \
  --model_path plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints \
  --model_subfolder checkpoints/gpn_promoter_pretrain_c512_v0.2 \
  --pooling_method mean --batch_size 512 --seq_len 512
```

NTv2 experiments:


```bash
# NTv2 + multi-species genomes
python scripts/plantcad_eigenanalysis.py \
  --source ntv2 --n_samples 4096 8192 16384 \
  --device cuda --output_dir results/sep/v32 \
  --model_path InstaDeepAI/nucleotide-transformer-v2-50m-multi-species \
  --model_revision 81b29e5786726d891dbf929404ef20adca5b36f1 \
  --dataset_path InstaDeepAI/multi_species_genomes \
  --dataset_revision b7326579fa3528b5896f5f4aa1554ad69ad5f105 \
  --dataset_config 12kbp \
  --dataset_sample_size 16384 \
  --dataset_shuffle \
  --text_column sequence \
  --dtype float32 --tokenization_mode lenient \
  --split train \
  --pooling_method mean --batch_size 16 --seq_len 8192

# NTv2 + animal promoter dataset
python scripts/plantcad_eigenanalysis.py \
  --source ntv2 --n_samples 4096 8192 16384 \
  --device cuda --output_dir results/sep/v33 \
  --model_path InstaDeepAI/nucleotide-transformer-v2-50m-multi-species \
  --model_revision 81b29e5786726d891dbf929404ef20adca5b36f1 \
  --dataset_path eczech/gpn-animal-promoter-dataset \
  --dataset_revision f009db443a914d4113922e3028de0666b85c24d6 \
  --species_column organism \
  --species_filter_file /tmp/gpn_promoter_species_filter.txt \
  --dtype float32 --tokenization_mode lenient \
  --split train \
  --pooling_method mean --batch_size 64 --seq_len 512

# AgroNT + PlantCAD2 dataset; use 6144bp sequences for 1024 tokens, see:
# https://huggingface.co/InstaDeepAI/agro-nucleotide-transformer-1b
python scripts/plantcad_eigenanalysis.py \
  --source ntv2 --n_samples 4096 8192 16384 \
  --device cuda --output_dir results/sep/v34 \
  --model_path InstaDeepAI/agro-nucleotide-transformer-1b \
  --model_revision b0e1ea1f53a2bf5bb29f8eab7a7e553bf06c1ab1 \
  --dataset_path plantcad/Angiosperm_65_genomes_8192bp \
  --dataset_revision 4a444fff5520b992aa978d92a5af509a81977098 \
  --dtype float32 --tokenization_mode lenient \
  --split train \
  --pooling_method mean --batch_size 64 --seq_len 6144

```

Hyena experiments:

```bash
python scripts/plantcad_eigenanalysis.py \
  --source hyena --n_samples 16384 65536 262144 \
  --device cuda --output_dir results/sep/v37 \
  --model_path LongSafari/hyenadna-tiny-1k-seqlen-hf \
  --model_revision e8c1effa8673814e257e627d2e1eda9ea5a373f6 \
  --dataset_path eczech/gpn-animal-promoter-dataset \
  --dataset_revision f009db443a914d4113922e3028de0666b85c24d6 \
  --species_column organism \
  --species_filter_file /tmp/gpn_promoter_species_filter.txt \
  --dtype float32 --tokenization_mode lenient \
  --split train \
  --pooling_method mean --batch_size 256 --seq_len 512
```

### Text

Experiments on [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) and [HuggingFaceTB/dclm-edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu).

```bash
# Wikitext + very small-scale Qwen local pretrain
python scripts/plantcad_eigenanalysis.py \
  --source marin --n_samples 512 1024 4096 8192 16384 \
  --device cuda --output_dir results/sep/v29 \
  --dataset_path Salesforce/wikitext \
  --dataset_revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --dataset_config wikitext-2-v1 \
  --text_column text \
  --tokenization_mode lenient \
  --split train \
  --model_path plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints \
  --model_subfolder checkpoints/wikitext_pretrain_c4096_v0.1 \
  --pooling_method mean --batch_size 64 --seq_len 4096
  
# Qwen2-1.5B + DCLM-Edu
python scripts/plantcad_eigenanalysis.py \
--source marin --n_samples 512 1024 4096 8192 16384 \
  --device cuda --output_dir results/sep/v30 \
  --model_path Qwen/Qwen2-1.5B \
  --dataset_path HuggingFaceTB/dclm-edu \
  --dataset_revision dbad8ad71224482740cd9c9d353591adbf62fe04 \
  --dataset_sample_size 262144 \
  --text_column text \
  --tokenization_mode lenient \
  --split train \
  --pooling_method mean --batch_size 16 --seq_len 4096

# Qwen2-1.5B + DCLM-Baseline
python scripts/plantcad_eigenanalysis.py \
  --source marin --n_samples 16384 \
  --device cuda --output_dir results/sep/v35 \
  --model_path Qwen/Qwen2-1.5B \
  --dataset_path mlfoundations/dclm-baseline-1.0 \
  --dataset_revision a3b142c183aebe5af344955ae20836eb34dcf69b \
  --dataset_sample_size 262144 \
  --text_column text \
  --tokenization_mode lenient \
  --split train \
  --pooling_method mean --batch_size 16 --seq_len 4096

# Qwen2-1.5B + StarCoder 
python scripts/plantcad_eigenanalysis.py \
  --source marin --n_samples 16384 \
  --device cuda --output_dir results/sep/v36 \
  --model_path Qwen/Qwen2-1.5B \
  --dataset_path bigcode/starcoderdata \
  --dataset_revision 9fc30b578cedaec69e47302df72cf00feed7c8c4 \
  --dataset_sample_size 262144 \
  --dataset_dir python \
  --text_column content \
  --tokenization_mode lenient \
  --split train \
  --pooling_method mean --batch_size 16 --seq_len 4096

```

### Relative Eigenanalysis (GEP)

Experiments for pair-wise eigenspectra decompositions between two models.

```bash
# PlantCAD2-S vs random init on validation split
python scripts/plantcad_relative_eigenanalysis.py \
  --model1_type masked --model1_path kuleshov-group/PlantCAD2-Small-l24-d0768 --model1_subfolder "" \
  --model1_label pc2-small \
  --model2_type masked --model2_path kuleshov-group/PlantCAD2-Small-l24-d0768 --model2_subfolder "" --model2_random_init \
  --model2_label pc2-small-randinit \
  --n_samples 16384 --seq_len 8192 --split validation --batch_size 32 --output_dir results/gep/v1
  
# PlantCAD2-S vs random init on train split
python scripts/plantcad_relative_eigenanalysis.py \
  --model1_type masked --model1_path kuleshov-group/PlantCAD2-Small-l24-d0768 --model1_subfolder "" \
  --model1_label pc2-small \
  --model2_type masked --model2_path kuleshov-group/PlantCAD2-Small-l24-d0768 --model2_subfolder "" --model2_random_init \
  --model2_label pc2-small-randinit \
  --n_samples 16384 --seq_len 8192 --split train --batch_size 32 --output_dir results/gep/v2
  
# Qwen pretrain vs random init on train split
python scripts/plantcad_relative_eigenanalysis.py \
  --model1_type causal --model1_path plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints \
  --model1_subfolder checkpoints/plantcad_pretrain_fp32_v0.3 \
  --model1_label marin-small \
  --model2_type causal --model2_path plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints \
  --model2_subfolder checkpoints/plantcad_pretrain_fp32_no_train_v0.3 \
  --model2_label marin-small-randinit \
  --n_samples 16384 --seq_len 4096 --split train --batch_size 64 --output_dir results/gep/v3
  
# Qwen pretrain vs the same model with kmer-shuffled sequences
python scripts/plantcad_relative_eigenanalysis.py \
  --model1_type causal --model1_path plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints \
  --model1_subfolder checkpoints/plantcad_pretrain_fp32_v0.3 \
  --model1_label marin-small \
  --model2_type causal --model2_path plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints \
  --model2_subfolder checkpoints/plantcad_pretrain_fp32_sim_k3_v0.3 \
  --model2_label marin-small-kmer-shuffle \
  --n_samples 16384 --seq_len 4096 --split train --batch_size 64 --output_dir results/gep/v4
```

## Model Pretraining

Commands to generate models for decomposition analyses above.

### DNA/Plants

```bash
# ------------------------------------------------------------------------------------------------
# Baseline models
# ------------------------------------------------------------------------------------------------
# https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/ogmaf8mr
python scripts/plantcad_pretrain.py --dtype float32 --wandb_run_name plantcad_pretrain_fp32_v0.2
# https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/ib431jty
python scripts/plantcad_pretrain.py --dtype float64 --wandb_run_name plantcad_pretrain_fp64_v0.2

# New runs following correction for W&B metric logging
# https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/ej4xxsvc
python scripts/plantcad_pretrain.py --dtype float32 --wandb_run_name plantcad_pretrain_fp32_v0.3
hf upload plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints /work/10459/eczech/vista/analysis/pcad_eigenspectrum/checkpoints/plantcad_pretrain_fp32_v0.3 checkpoints/plantcad_pretrain_fp32_v0.3 --repo-type model

# https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/iodqcytt
python scripts/plantcad_pretrain.py --dtype float64 --wandb_run_name plantcad_pretrain_fp64_v0.3
hf upload plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints /work/10459/eczech/vista/analysis/pcad_eigenspectrum/checkpoints/plantcad_pretrain_fp64_v0.3 checkpoints/plantcad_pretrain_fp64_v0.3 --repo-type model

# Random init baseline
python scripts/plantcad_pretrain.py --dtype float32 --no_train --wandb_run_name plantcad_pretrain_fp32_no_train_v0.3
hf upload plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints /work/10459/eczech/vista/analysis/pcad_eigenspectrum/checkpoints/plantcad_pretrain_fp32_no_train_v0.3 checkpoints/plantcad_pretrain_fp32_no_train_v0.3 --repo-type model

# ------------------------------------------------------------------------------------------------
# Simulation models
# ------------------------------------------------------------------------------------------------

python scripts/plantcad_pretrain.py --dtype float32 \
  --simulation_mode high_homology \
  --simulation_attractor_ratio .2 \
  --simulation_injection_rate .8 \
  --wandb_run_name plantcad_pretrain_fp32_sim_ar0.2_ir0.8_v0.3
#  https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/4e4pd44s  
hf upload plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints /work/10459/eczech/vista/analysis/pcad_eigenspectrum/checkpoints/plantcad_pretrain_fp32_sim_ar0.2_ir0.8_v0.3 checkpoints/plantcad_pretrain_fp32_sim_ar0.2_ir0.8_v0.3 --repo-type model

python scripts/plantcad_pretrain.py --dtype float32 \
  --simulation_mode low_homology \
  --simulation_k 3 \
  --wandb_run_name plantcad_pretrain_fp32_sim_k3_v0.3
# https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/daqyb5zp
hf upload plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints /work/10459/eczech/vista/analysis/pcad_eigenspectrum/checkpoints/plantcad_pretrain_fp32_sim_k3_v0.3 checkpoints/plantcad_pretrain_fp32_sim_k3_v0.3 --repo-type model

python scripts/plantcad_pretrain.py --dtype float32 \
  --simulation_mode random \
  --simulation_random_probability 1.0 \
  --wandb_run_name plantcad_pretrain_fp32_sim_rand_p1.0_v0.3
# https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/y8woce47
hf upload plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints /work/10459/eczech/vista/analysis/pcad_eigenspectrum/checkpoints/plantcad_pretrain_fp32_sim_rand_p1.0_v0.3 checkpoints/plantcad_pretrain_fp32_sim_rand_p1.0_v0.3 --repo-type model

```

### DNA/Animals

```bash
python scripts/plantcad_pretrain.py \
  --dtype float32 \
  --dataset_path songlab/gpn-animal-promoter-dataset \
  --dataset_revision 09d363c86202374986c4a7ed6d39073aa1ac2e23 \
  --tokenizer_path songlab/gpn-animal-promoter \
  --tokenizer_revision 7cf3276a03b5e243efd421b8939ed3d1e7dcf9cc \
  --sequence_length 512 \
  --batch_size 32 \
  --wandb_run_name gpn_promoter_pretrain_c512_v0.1
# https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/y107yujz
hf upload plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints /work/10459/eczech/vista/analysis/pcad_eigenspectrum/checkpoints/gpn_promoter_pretrain_c512_v0.1 checkpoints/gpn_promoter_pretrain_c512_v0.1 --repo-type model

python scripts/plantcad_pretrain.py \
  --dtype float32 \
  --dataset_path songlab/gpn-animal-promoter-dataset \
  --dataset_revision 09d363c86202374986c4a7ed6d39073aa1ac2e23 \
  --tokenizer_path songlab/gpn-animal-promoter \
  --tokenizer_revision 7cf3276a03b5e243efd421b8939ed3d1e7dcf9cc \
  --sequence_length 512 \
  --batch_size 32 \
  --steps 68352 \
  --wandb_run_name gpn_promoter_pretrain_c512_v0.2
# https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/4ihb7bbj
hf upload plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints /work/10459/eczech/vista/analysis/pcad_eigenspectrum/checkpoints/gpn_promoter_pretrain_c512_v0.2 checkpoints/gpn_promoter_pretrain_c512_v0.2 --repo-type model
```

### Text

```bash
python scripts/plantcad_pretrain.py \
  --dataset_path Salesforce/wikitext \
  --dataset_revision main \
  --dataset_config wikitext-2-v1 \
  --tokenizer_path gpt2 \
  --tokenizer_revision main \
  --sequence_length 4096 \
  --batch_size 32 \
  --text_column text \
  --wandb_run_name wikitext_pretrain_c4096_v0.1
# https://wandb.ai/eric-czech/plantcad-eigenanalysis/runs/96vk25uq
hf upload plantcad/marin_exp2101__pcv2_pretrain_c4096__checkpoints /work/10459/eczech/vista/analysis/pcad_eigenspectrum/checkpoints/wikitext_pretrain_c4096_v0.1 checkpoints/wikitext_pretrain_c4096_v0.1 --repo-type model
```