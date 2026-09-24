# Phase 2 — Legal Corpus Collection and Preparation

## Objective

Build the first reproducible legal-text corpus for LawSuit LLM without jumping directly to large-scale training.

## Dataset

The initial source is Open India Law, maintained by Vaquill AI. Its legislation collection contains Indian primary-law provisions normalized to section-level records and retains links back to source publishers. The published dataset is CC BY 4.0.

The repository uses snapshot v2026.08.1 for reproducibility.

## Initial experiment

The first run collects at most **5,000 provisions** from:

- Central legislation
- Madhya Pradesh legislation

The source is streamed from jurisdiction-specific Parquet files. We do not load the entire corpus into memory.

The full Open India Law corpus is much larger, with more than one million legislation sections, so later experiments can expand the jurisdiction list and record cap without redesigning the pipeline.

## Pipeline

Open India Law → Streaming Parquet → Raw JSONL → Cleaning → Deduplication → Clean corpus → 90/5/5 split → Statistics

## Provenance

Each retained record keeps act_id, title, chapter, section_number, section_title, state, year, amendment_count, act_status, section_status, source_url, source_publisher, source_snapshot, source_dataset, and text_sha256.

This allows later model outputs to be connected to source material rather than treating generated text as authoritative legal information.

## Output

data/raw/open_india_law_legislation.jsonl

data/cleaned/legal_corpus.jsonl

data/splits/train.jsonl

data/splits/validation.jsonl

data/splits/test.jsonl

## Run Phase 2

From the repository root:

git pull origin main

python -m src.data.run_pipeline

The pipeline prints collection counts, filtering counts, split sizes, and corpus statistics.

## What we do NOT do yet

Phase 2 does not train the Transformer, train the tokenizer, create embeddings, use a pretrained language model, build a frontend, or build a production API.

## Expansion path

5K → 25K → 100K+ → larger experiments

We will increase the corpus only after verifying data quality and statistics at each stage.

## Important legal-data note

The dataset is a point-in-time archive. The Open India Law documentation states that legal material changes over time and should be checked against the authoritative source before real-world reliance.