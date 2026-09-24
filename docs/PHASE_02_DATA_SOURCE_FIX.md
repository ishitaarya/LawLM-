# Phase 2 Data Source Fix

The first Phase 2 ingestion attempt streamed the Open India Law Parquet mirror directly from `oss-data-in.vaquill.ai` and the connection was repeatedly dropped before records could be read.

The project therefore uses the published Hugging Face dataset endpoint for the legislation corpus. The upstream Open India Law documentation explicitly supports loading the `legislation` configuration through `datasets.load_dataset(...)`, while the same project also publishes the `oss-data-in.vaquill.ai` mirror. The corpus remains the same Open India Law source and the pipeline still preserves source URL, publisher, snapshot, and dataset provenance.

The Phase 2 record cap remains 5,000 for the first reproducible run. This is intentionally a small ingestion sample before scaling to 25,000 and later 100,000+ provisions.

## Run

```powershell
python -m src.data.run_pipeline
```

## Expected stages

1. Collect legal provisions from the Open India Law Hugging Face legislation configuration.
2. Preserve provenance metadata.
3. Clean and deduplicate text.
4. Create deterministic 90/5/5 train/validation/test splits.
5. Print corpus statistics.

The model is not trained in Phase 2.
