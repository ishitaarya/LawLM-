# LawLM — Legal Language Model from Scratch

LawLM is a B.Tech AI/ML project that builds a small decoder-only Transformer language model for legal-text generation.

## Core project principle

LawLM is trained **from randomly initialized weights** on a legal dataset. It does not fine-tune GPT, Llama, Mistral, Qwen, or any other pretrained LLM.

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
- Virtual environment: `.venv`
- Basic ML/data dependencies configured
- Project directory structure created
- Environment test available at `tests/test_environment.py`

The first prototype will be intentionally small so it can be developed and tested on a student laptop.

## Project structure

```
LawLM/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── tokenizer/
├── src/
│   ├── data/
│   ├── tokenizer/
│   ├── model/
│   ├── training/
│   ├── generation/
│   └── evaluation/
├── checkpoints/
├── experiments/
├── graphs/
├── tests/
├── configs/
├── notebooks/
├── .gitignore
├── requirements.txt
└── README.md
```

## Important limitation

LawLM is an educational language-model project. Generated text must not be treated as legal advice or as a substitute for a qualified legal professional.

## License

To be decided as the project develops.
