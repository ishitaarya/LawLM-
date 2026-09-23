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