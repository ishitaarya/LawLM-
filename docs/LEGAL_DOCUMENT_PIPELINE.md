# Legal PDF processing pipeline

This pipeline converts extracted PDF pages into provision-bounded legal chunks. It preserves page/line references, keeps amendments and footnotes separate, records known OCR anomalies, and does not claim extracted text is legally correct or current.

## Processing flow

```text
PDF
 |
 v
pdf_loader.py
 |  page JSONL (one record per PDF page)
 v
legal_document_pipeline.py
 |
 +--> legal_section_assembler.py
 |      section heading + body + page/line provenance
 |
 +--> legal_validation.py
 |      structural checks only; legal accuracy remains unverified
 |
 +--> legal_chunker.py
        section-bounded chunks with page/line citations
```

## Windows PowerShell commands

Run from the repository root after activating the project virtual environment.

1. Extract a PDF to a **new** page JSONL file:

```powershell
python -m src.document.pdf_loader "data/raw/Indian_Contract_Act_1872.pdf" --output "data/documents/Indian_Contract_Act_1872_pages_v2.jsonl"
```

2. Assemble, validate, and chunk it into **new output files**:

```powershell
python -m src.document.legal_document_pipeline --source "data/documents/Indian_Contract_Act_1872_pages_v2.jsonl" --sections-output "data/processed/Indian_Contract_Act_1872_sections_v2.jsonl" --report-output "reports/Indian_Contract_Act_1872_validation_v2.json" --chunks-output "data/processed/Indian_Contract_Act_1872_legal_chunks_v2.jsonl"
```

Use a new suffix (for example, `_v3`) for another run. The extractor and artifact pipeline refuse to overwrite existing outputs.\n\nIf the enactment/preamble marker is not detected automatically, verify the first Act-body page from the PDF and pass it explicitly with `--body-start-page 10` (replace 10 with the verified page number). The pipeline refuses to treat an unknown table-of-contents boundary as legal text.

## Output contracts

### Assembled section record

Each section keeps the existing downstream fields:

- `section_number`, `section_title`, `page_number`, `end_page_number`
- `source_pages`, `text`, `chapter`
- `amendments`, `footnotes`, `ocr_anomalies`
- `section_status`, `review_required`

It also adds:

- `heading_source`: exact heading page, line, and raw extracted line.
- `source_spans`: each body text line with its source page and line number.

### Chunk record

Each chunk belongs to one section only. It carries section metadata, page range, source pages, source line excerpts, review status, and flags indicating whether amendments, footnotes, or OCR anomalies were associated with the parent section.

### Validation report

The report checks duplicate/missing section numbers, empty text, page ranges, page references, heading provenance, body-line provenance, and annotation counts. It always keeps:

- `legal_accuracy_verified: false`
- `ready_for_training: false`

A structurally clean extraction is not proof of legal accuracy, completeness, amendment currency, or suitability for training.


## Retrieval and model QA

After the new chunk file has been reviewed, it can be passed to the sparse
retriever without replacing the legacy chunk file:

```powershell
python -m src.document.retriever "What makes an agreement a contract?" --input "data/processed/Indian_Contract_Act_1872_legal_chunks_v2.jsonl"
```

The from-scratch model QA pipeline also accepts an explicit chunk path:

```powershell
python -m src.document.pipeline "Explain Section 10 in simple language." --chunks "data/processed/Indian_Contract_Act_1872_legal_chunks_v2.jsonl"
```

Explicitly marked repealed sections are excluded from normal retrieval. The
retriever has an opt-in `--include-repealed` flag for historical research.
Unknown or review-required sections are not silently treated as verified law;
their status is carried into the model context. The QA pipeline still depends
on the configured tokenizer and checkpoint being available locally.

## Manual review rules

- Do not silently turn OCR strings such as `1151.` or `1161.` into Section 151 or 161. They are recorded as anomalies for human review.
- Do not infer repeal status. Supply a verified repeal list only when it has been checked against an authoritative legal source. The CLI accepts repeated `--repealed-section 76` arguments for verified entries; omit this option when repeal status is not established.
- Treat amendment text separately from the central Act text.
- Keep footnotes and explanations as annotations until their relationship to the provision has been reviewed.
- Review every section marked `review_required` before training or publishing.
- If a PDF page has no extractable text, this text-based pipeline cannot recover it. OCR is a separate step and OCR output must be reviewed.

## Current repository limitations

The source PDF and page-level JSONL are user data and are not assumed to be present in GitHub. The automated tests use small synthetic fixtures. Full Indian Contract Act validation requires the actual PDF/page JSONL and manual comparison with an authoritative publication.

## Test suite

```powershell
python -m unittest discover -s tests -p "test_legal_parser.py" -v
python -m unittest discover -s tests -p "test_legal_section_assembler.py" -v
python -m unittest discover -s tests -p "test_legal_validation.py" -v
python -m unittest discover -s tests -p "test_legal_chunker.py" -v
python -m unittest discover -s tests -p "test_legal_document_pipeline.py" -v
```
