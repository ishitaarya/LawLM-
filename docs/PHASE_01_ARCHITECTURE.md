# LawSuit LLM — Phase 1: Foundation

## Objective

Phase 1 establishes the engineering foundation for the seven-phase LawSuit LLM project. It does not train the model yet.

## Seven-phase contract

1. Foundation
2. Legal corpus collection and preparation
3. Tokenizer from scratch
4. Decoder-only Transformer from scratch
5. Training and experiment framework
6. Generation and legal document understanding
7. Evaluation, enhancement, and research experiments

## Project principles

- Model weights start from random initialization.
- No pretrained LLM is used as the base model.
- Core LLM components are implemented explicitly so their behavior can be studied.
- The project is educational/research-oriented rather than production deployment.
- Corpus size and model size remain independently configurable.
- Data is processed incrementally instead of assuming the whole corpus belongs in RAM.

## Initial model target

The first model experiment targets approximately 10M parameters.

| Setting | Initial value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Target parameters | ~10M |
| Vocabulary size | 8,000 |
| Context length | 256 |
| Embedding dimension | 384 |
| Transformer layers | 4 |
| Attention heads | 6 |
| FFN dimension | 1,536 |
| Dropout | 0.1 |
| Initialization | Random |

The exact parameter count will be measured programmatically once the model implementation is introduced in Phase 4.

## Data layers

`raw → extracted → cleaned → deduplicated → structured → tokenized → train/validation/test`

Each representation has a separate purpose. Raw legal sources are not automatically considered training-ready.

## Runtime policy

The codebase supports portable device selection:

1. CUDA when available;
2. MPS when available;
3. CPU otherwise.

Explicit device overrides remain possible for controlled experiments.

AMD systems are handled through the actual PyTorch backend available on the machine rather than by assuming CUDA from the hardware name.

## Phase 1 deliverables

- Seven-phase project specification.
- Central project/model configuration.
- Portable hardware profiles.
- Standard data directory layout.
- Existing environment inspection retained.
- Configuration tests retained and updated as needed.
- Documentation describing the project contract and exit criteria.

## Phase 1 exit criteria

Phase 1 is complete when:

- project configuration loads successfully;
- model configuration is centralized;
- hardware inspection runs;
- configuration tests pass;
- repository structure matches the seven-phase plan;
- README and Phase 1 documentation describe the same architecture.

## Next phase

Phase 2 will implement legal-corpus ingestion, provenance tracking, cleaning, normalization, deduplication, and dataset statistics.
