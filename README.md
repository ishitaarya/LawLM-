# LawLM — Legal Language Model from Scratch

LawLM is a B.Tech AI/ML project that builds a small decoder-only Transformer language model for legal-text generation.

## Core project principle

LawLM is trained from randomly initialized weights on a legal dataset. It does not fine-tune GPT, Llama, Mistral, Qwen, or any other pretrained LLM.

## Dataset

The project uses the Open India Law legislation subset published by Vaquill AI. The corpus is normalized from Indian primary-law sources and the legislation portion is sourced from India Code. Legislation records include provenance such as title, section, status, jurisdiction, and source URL.

For the student-scale prototype, LawLM streams and keeps 5,000 legislation provisions locally instead of downloading the entire corpus.

- Dataset: vaquill/open-india-law
- Configuration: legislation
- Dataset license: CC BY 4.0
- Scripts in this repository: Apache-2.0
- Source/provenance: official government legal sources as documented by Open India Law
- Dataset snapshot: use the current snapshot available when the data is downloaded
- Attribution: Vaquill AI / Open India Law, with required CC BY 4.0 attribution

Source: https://github.com/Vaquill-AI/open-india-law

The dataset is a point-in-time archive. Legal content changes over time, so generated output must be verified against the current official source before any real-world use.

## Planned pipeline

1. Environment setup
2. Legal dataset preparation
3. Train our own tokenizer
4. Create next-token prediction datasets
5. Implement a decoder-only Transformer from scratch
6. Train with PyTorch
7. Generate legal-style text
8. Evaluate loss and perplexity
9. Run model/data-size experiments
10. Optional API layer
11. Frontend added only after the LLM is complete

## Current status

### Phase 1 — Environment Setup

- Python 3.13
- PyTorch 2.14.0 CPU build
- AMD Radeon graphics detected; CUDA is unavailable
- Virtual environment: .venv
- Basic ML/data dependencies configured
- Project directory structure created
- Environment test available at tests/test_environment.py

### Phase 2 — Dataset Pipeline

The repository now contains a one-command dataset pipeline:

1. Stream Open India Law legislation
2. Keep 5,000 provisions
3. Clean and normalize text
4. Remove duplicate provisions/text
5. Create 90/5/5 train/validation/test splits
6. Print dataset statistics

The raw dataset is intentionally excluded from Git via .gitignore.

## Project structure

LawLM/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/
│   │   ├── download_dataset.py
│   │   ├── clean_dataset.py
│   │   ├── split_dataset.py
│   │   ├── dataset_stats.py
│   │   └── run_pipeline.py
│   ├── tokenizer/
│   ├── model/
│   ├── training/
│   ├── generation/
│   └── evaluation/
├── tokenizer/
├── checkpoints/
├── experiments/
├── graphs/
├── tests/
├── configs/
├── notebooks/
├── .gitignore
├── requirements.txt
└── README.md

## Important limitation

LawLM is an educational language-model project. Generated text must not be treated as legal advice or as a substitute for a qualified legal professional.

## License

The LawLM source code is licensed separately from the external dataset. The Open India Law dataset remains subject to its stated CC BY 4.0 terms and attribution requirements.

## Phase 1 — Project, Hardware & Storage Architecture

Phase 1 now defines the scalable architecture for the LawLM project.

### Scaling target

200 GB → 500 GB → 1 TB → 2 TB → 5 TB

The corpus target refers to the source reservoir. It is not loaded into RAM or automatically used as one training run.

### Initial from-scratch model

- ~50M parameters
- 8 Transformer layers
- 512 embedding dimension
- 8 attention heads
- 2048 FFN dimension
- 512-token context
- 16K vocabulary
- Random initialization
- No pretrained LLM

### Device-agnostic design

The project supports:

- CPU development and preprocessing
- NVIDIA CUDA training where supported
- AMD ROCm where the exact hardware/software stack is supported
- Apple MPS where supported
- Later cloud/distributed GPU training

### Storage architecture

The data pipeline is separated into:

`data/raw` → `data/extracted` → `data/cleaned` → `data/deduplicated` → `data/structured` → `data/tokenized` → train/validation/test

For a 200 GB source corpus, Phase 1 recommends at least 500 GB of free workspace and prefers approximately 1 TB. The pipeline is designed to process data incrementally when local storage is insufficient.

### Phase 1 files

- `configs/project.yaml` — project-wide architecture and scaling configuration
- `configs/hardware_profiles.yaml` — portable CPU/CUDA/ROCm/MPS profiles
- `src/utils/system_info.py` — runtime, CPU, storage, and PyTorch-device inspection
- `tests/test_phase1_config.py` — Phase 1 configuration tests
- `docs/PHASE_01_ARCHITECTURE.md` — detailed Phase 1 architecture

Run the environment inspection with:

```bash
python src/utils/system_info.py
```

### Next phase

**Phase 2 — Legal Data Collection:** build a resumable, logged ingestion pipeline before attempting the large legal corpus download.
