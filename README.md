# LawSuit LLM — Legal Language Model From Scratch

LawSuit LLM is an educational B.Tech AI/ML project that builds a small legal language model from randomly initialized weights and progressively adds legal document understanding.

The repository name remains `LawLM-`, but the project direction is now **LawSuit LLM**.

## Project goal

LawSuit LLM explores how an LLM works under the hood instead of starting with a pretrained model.

The long-term system is intended to help lawyers and non-lawyers understand legal language and legal documents by combining:

- a from-scratch decoder-only Transformer;
- a legal text corpus;
- our own tokenizer;
- next-token prediction training;
- PDF/document extraction;
- legal document analysis;
- retrieval/evidence-aware processing;
- evaluation and experiments.

This is a learning and research project, not a production legal-advice system.

## Core principles

- **From scratch:** model weights start from random initialization.
- **No pretrained LLM:** no GPT, Llama, Mistral, Qwen, etc. are used as the base model.
- **Understand the internals:** attention, embeddings, masking, transformer blocks, loss, backpropagation, optimization, and generation are implemented and studied explicitly.
- **Progressive engineering:** foundation first, model second, legal document capabilities later.
- **Device-aware:** code is portable across CPU, CUDA, ROCm, and MPS where the relevant PyTorch stack is available.
- **Scalable experiments:** corpus size and model parameter count are configurable independently.

## Seven phases

| Phase | Focus | Main outcome |
|---|---|---|
| 1 | Foundation | Repository, configuration, environment, tests, hardware inspection |
| 2 | Legal corpus | Collection, cleaning, normalization, deduplication, splitting, statistics |
| 3 | Tokenizer | Train and inspect our own BPE-style tokenizer |
| 4 | Transformer | Implement the decoder-only Transformer from scratch |
| 5 | Training | Next-token prediction, checkpoints, logging, validation, experiments |
| 6 | Generation + documents | Text generation and legal PDF/document understanding |
| 7 | Enhancement + evaluation | Benchmarks, experiments, evidence-aware workflows, research improvements |

## Initial model experiment

The first substantive model target is approximately **10 million parameters**.

Current configuration target:

- Architecture: decoder-only Transformer
- Target parameters: ~10M
- Vocabulary: 10,000
- Context length: 256 tokens
- Embedding dimension: 384
- Transformer layers: 4
- Attention heads: 6
- FFN dimension: 1,536
- Dropout: 0.1
- Initialization: random

These are configuration values for the initial experiment. They are deliberately kept editable so later experiments can change one variable at a time.

## Repository layout

```
LawLM-/
├── configs/
│   ├── project.yaml
│   ├── project_config.json
│   └── hardware_profiles.yaml
│
├── data/
│   ├── raw/
│   ├── extracted/
│   ├── cleaned/
│   ├── deduplicated/
│   ├── structured/
│   ├── tokenized/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data/
│   ├── tokenizer/
│   ├── model/
│   ├── training/
│   ├── generation/
│   ├── evaluation/
│   └── utils/
│
├── tokenizer/
├── checkpoints/
├── experiments/
├── graphs/
├── notebooks/
├── docs/
├── tests/
├── requirements.txt
└── README.md
```

## Current implementation status

This repository has progressed beyond the initial Phase 1 foundation.

| Component | Current status |
|---|---|
| Python/PyTorch environment and API smoke test | Implemented |
| Legal corpus ingestion and initial processing | Initial version implemented; source quality still needs review |
| 10,000-token tokenizer | Implemented |
| From-scratch decoder-only Transformer | Implemented; approximately 11.04M parameters |
| Initial next-token training | Completed for earlier corpus experiments; checkpoints exist |
| PDF page extraction | Implemented |
| Legal section parser and cross-page assembler | Implemented on the legal-document feature branch |
| Structural validation and page/line provenance | Implemented; automated tests are configured |
| Section-aware, source-traceable chunking | Implemented on the legal-document feature branch |
| Full source-document validation | Pending actual PDF/page JSONL review |
| Legal QA evaluation and retrieval integration | In progress |
| Frontend and final project packaging | Pending |

The current Transformer configuration uses a 10,000-token vocabulary, 256-token context, 384-dimensional embeddings, 4 layers, 6 attention heads, and a 1,536-dimensional feed-forward layer. It starts from random weights; it does not use a pretrained LLM.

The legal document pipeline is documented in [docs/LEGAL_DOCUMENT_PIPELINE.md](docs/LEGAL_DOCUMENT_PIPELINE.md). It creates new outputs only and refuses to overwrite existing files. Structural validation does not establish legal accuracy or training readiness.

**Important:** the actual Indian Contract Act PDF and page-level JSONL are user data and are not assumed to be stored in this repository. Synthetic unit tests can test parser behavior, but full-document validation still requires the real source document and manual legal review.

## Legal data

The repository may use public legal datasets with explicit provenance and licensing. Any external corpus must retain its source, version/snapshot information, license requirements, and provenance metadata.

Legal material can change over time. A trained model or generated response must therefore not be treated as a current statement of law without checking the relevant authoritative source.

## Development philosophy

Every major component should be understandable independently:

```
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Embeddings + Positions
  ↓
Masked Self-Attention
  ↓
Feed Forward Network
  ↓
Residual + LayerNorm
  ↓
Transformer Blocks
  ↓
Logits
  ↓
Cross-Entropy Loss
  ↓
Backpropagation
  ↓
Parameter Update
  ↓
Next Token
```

The project will build this pipeline incrementally and test each stage before moving on.

## License

The project source code and external datasets are governed by their respective licenses. Dataset-specific attribution and license terms must be preserved in project documentation.
