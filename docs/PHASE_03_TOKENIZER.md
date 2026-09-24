# Phase 3 — Tokenizer from Scratch

## Objective

Build the tokenizer used by LawSuit LLM from the project legal corpus. The tokenizer is trained only on the training split and uses a 10,000-token BPE vocabulary.

## Configuration

- Algorithm: BPE
- Vocabulary size: 10,000
- Training data: `data/splits/train.jsonl`
- Validation/test data: used only after tokenizer training
- Special tokens:
  - `<pad>` = 0
  - `<unk>` = 1
  - `<bos>` = 2
  - `<eos>` = 3
- Output directory: `data/tokenizer`

## Workflow

1. Read legal text from the training split.
2. Train the BPE tokenizer.
3. Save the tokenizer model and vocabulary.
4. Test encode/decode behavior.
5. Encode train, validation, and test splits using the same trained tokenizer.
6. Record token counts and average tokens per provision.

## Local commands

```powershell
python src/tokenizer/train_tokenizer.py
python src/tokenizer/test_tokenizer.py
python src/tokenizer/encode_dataset.py
```

## Output

The tokenizer model is stored under:

`data/tokenizer/lawsuit_bpe.model`

The encoded datasets are stored under:

`data/tokenized/`

## Data-leakage rule

The tokenizer learns its vocabulary only from `train.jsonl`. Validation and test records are never used to learn BPE merges.

## Phase 4 handoff

Phase 4 will consume these token IDs and construct fixed-length 256-token training sequences for the randomly initialized decoder-only Transformer.
