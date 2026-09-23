# LawLM — Phase 1: Project, Hardware & Storage Architecture

## Objective
Phase 1 establishes a portable project architecture before the large legal corpus is collected.

The project supports local CPU development, NVIDIA CUDA systems, supported AMD ROCm systems, Apple Silicon/MPS systems, and later cloud/distributed GPU training.

## Scaling strategy
LawLM starts with a source-corpus target of approximately 200 GB and is designed to scale to:

200 GB → 500 GB → 1 TB → 2 TB → 5 TB

The source-corpus size is not the same as the amount of data loaded into RAM or used in a single training run.

## Storage rule
For a 200 GB source corpus, storage must account for raw files, extracted text, cleaned data, deduplicated data, structured/chunked data, tokenized shards, train/validation/test shards, temporary files, and checkpoints.

Phase 1 uses 300 GB free space as a safety gate and recommends 500 GB+ free workspace for the first large corpus. A 1 TB workspace is preferred.

If local storage is insufficient, process incrementally or use external/cloud storage rather than keeping every representation locally.

## Data layers
Internet/legal sources → data/raw → data/extracted → data/cleaned → data/deduplicated → data/structured → data/tokenized → train/validation/test.

Raw data is never treated as training data automatically.

## Initial model boundary
| Setting | Initial value |
|---|---:|
| Parameters | ~50M |
| Transformer layers | 8 |
| Embedding dimension | 512 |
| Attention heads | 8 |
| FFN dimension | 2048 |
| Context length | 512 |
| Vocabulary | 16K |
| Initialization | Random |
| Pretrained model | No |

These are the starting architecture values, not a claim that a laptop can efficiently train on the entire 200 GB source reservoir.

## Device policy
- CPU: ingestion, extraction, OCR orchestration, cleaning, deduplication, metadata processing, tokenizer preparation, and small model tests.
- NVIDIA CUDA: substantially larger training experiments when a compatible GPU is available.
- AMD ROCm: only where the exact GPU/OS/PyTorch combination is supported.
- Apple MPS: development and compatible training experiments.

## Portable runtime
Run `python src/utils/system_info.py` to inspect the current operating system, Python version, CPU, disk capacity/free space, and detected PyTorch device.

## Phase 1 completion criteria
- Scalable configuration exists.
- Storage boundaries are documented.
- Hardware profiles are device-agnostic.
- Runtime inspection works without downloading the corpus.
- Phase 2 can build on the same structure without redesign.

## Next phase
Phase 2 — Legal Data Collection. It will implement resumable, logged ingestion and validate storage, source metadata, document IDs, checksums, retry/resume behavior, and output locations before large downloads.