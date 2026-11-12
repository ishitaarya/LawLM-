# Legal PDF processing pipeline

This pipeline converts extracted PDF pages into provision-bounded legal chunks. It preserves page/line references, keeps amendments and footnotes separate, records known OCR anomalies, and does not claim extracted text is legally correct or current.

## Step 0: audit extracted pages before section assembly

Run this read-only audit immediately after PDF extraction and before building
sections. It checks duplicate/missing page numbers, blank and unusually short
pages, inconsistent source filenames, replacement characters and control
characters. It never repairs extracted text or overwrites the source JSONL.

```powershell
python -m src.document.legal_source_audit --source "data/documents/Indian_Contract_Act_1872_pages_v2.jsonl" --output "reports/Indian_Contract_Act_1872_page_audit_v2.json"
```

Inspect every reported short/empty/OCR-warning page against the original PDF.
A clean report is only a structural extraction check; it cannot prove OCR
accuracy or detect every missing word, column, table, or legal amendment.

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



## Retrieval evaluation

The evaluator measures section retrieval with Hit Rate@1/3/5 and Mean
Reciprocal Rank (MRR). It requires a **human-reviewed** benchmark JSONL file;
the project does not invent expected legal sections.

Each benchmark line must follow this shape:

```json
{"query": "What makes an agreement a contract?", "expected_sections": ["10"]}
```

Run it after creating the reviewed benchmark:

```powershell
python -m src.evaluation.legal_retrieval_evaluation --chunks "data/processed/Indian_Contract_Act_1872_legal_chunks_v2.jsonl" --benchmark "data/evaluation/legal_retrieval_benchmark.jsonl" --output "reports/legal_retrieval_v2.json"
```

These metrics evaluate retrieval ranking only. They do not evaluate the
generated model answer or prove that a legal interpretation is correct.

## Supervised QA dataset audit

The QA audit is read-only. It reports missing/empty fields, suspiciously short
contexts or answers, answers that may be sentence-truncated, explicit hard
token-truncation flags, duplicate questions, and JSON parsing errors. It does
not delete records or rewrite the training dataset.

```powershell
python -m src.training.qa_dataset_audit --source "data/qa_v2/train.jsonl" --output "reports/qa_v2_train_audit.json"
```

Repeat for validation and test splits, using a different new report filename
for each. The audit expects the canonical `question`, `context`, `answer`,
and `section` fields. If the local QA v2 files use a different schema, inspect
their field names before adapting the audit; do not rename or drop source data
blindly. A clean structural report is not legal verification and never sets
`ready_for_training` to true.

## Test suite

```powershell
python -m unittest discover -s tests -p "test_legal_parser.py" -v
python -m unittest discover -s tests -p "test_legal_section_assembler.py" -v
python -m unittest discover -s tests -p "test_legal_validation.py" -v
python -m unittest discover -s tests -p "test_legal_chunker.py" -v
python -m unittest discover -s tests -p "test_legal_document_pipeline.py" -v
```
