# Phase 6 — Text Generation

## Objective
Load the trained LawSuit LLM checkpoint and generate legal-style text using the tokenizer trained from the project corpus.

## Inputs
- Model: `checkpoints/lawsuit_llm_epoch_03.pt`
- Tokenizer: `data/tokenizer/lawsuit_bpe.model`
- Architecture: 11,036,928-parameter decoder-only Transformer
- Device: CPU when CUDA is unavailable

## Generation flow

```
Prompt
  ↓
Project BPE tokenizer
  ↓
Token IDs
  ↓
Trained LawSuit LLM
  ↓
Next-token sampling
  ↓
Generated token IDs
  ↓
Project BPE decoder
  ↓
Generated text
```

## Sampling controls
- Temperature controls randomness.
- Top-k limits sampling to the k highest-probability tokens.
- Max-new-tokens controls generated length.

## Run

From the repository root:

```powershell
python -m src.generation.generate
```

Custom prompt:

```powershell
python -m src.generation.generate --prompt "The court may grant relief under the applicable law because"
```

## What to evaluate

Generation is an experiment, not proof of legal correctness. Record whether outputs show:
1. Legal vocabulary and syntax.
2. Coherent continuation of the prompt.
3. Repeated or broken text.
4. Invented sections, cases, or legal claims.
5. Sensitivity to temperature and top-k.

The model is trained on a small corpus and should not be used as legal advice. Any legal statement must be checked against authoritative legal sources.
