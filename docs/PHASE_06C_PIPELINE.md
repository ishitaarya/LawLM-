# Phase 6C — End-to-End Legal QA

Phase 6C connects the Phase 6B retriever and context builder to the trained LawSuit LLM.

## Pipeline

Question -> TF-IDF/legal concept retrieval -> context builder -> LawSuit LLM generation.

The model is loaded from the locally trained checkpoint:
- `checkpoints/lawsuit_llm_epoch_03.pt`
- `data/tokenizer/lawsuit_bpe.model`

No pretrained LLM or pretrained embedding model is used.

## Run

```powershell
python -m src.document.pipeline "When is an agreement without consideration valid?"
```

## Test

```powershell
python -m src.document.test_pipeline
```

The generated text is a research/educational demonstration. Retrieval supplies the legal source context; the 11M-parameter model should not be treated as a reliable legal advisor.
