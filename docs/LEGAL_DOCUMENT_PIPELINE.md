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


## Human extraction-review manifest

Create a separate review checklist after assembling sections. This command reads
the section JSONL and creates a **new** manifest; it never edits the source
sections or chunks:

```powershell
python -m src.document.legal_review --source "data/processed/Indian_Contract_Act_1872_sections_v2.jsonl" --output "data/review/Indian_Contract_Act_1872_review_v2.jsonl"
```

Every section starts with `review_status: "pending"`. The manifest preserves
section identity, page/line provenance, amendments, footnotes, OCR anomalies,
and a SHA-256 hash of the exact assembled section text. Review reasons include
short text, duplicate/missing section numbers, missing provenance, unverified
section status, and annotations requiring attention.

A reviewer approval must be recorded through
`approve_review_record` in `src.document.legal_review`, supplying a non-empty
reviewer identity, review timestamp, and the exact `source_text_sha256`
observed during review. A mismatched hash is rejected. The validation helper
checks every manifest entry against the current sections and reports changed
text, missing records, duplicate indices, or invalid approval metadata.

This approval means only that a human reviewed the extracted text and its
structure against the source. It does **not** certify legal interpretation,
currentness, completeness, or permission to train. Both
`legal_accuracy_verified` and `ready_for_training` remain false. If source
text changes, the previous approval no longer validates and a fresh pending
manifest should be generated. Keep the review manifest as a separate artifact;
do not overwrite source sections, chunks, QA datasets, or checkpoints.


## Controlled legal training-candidate gate

The review manifest has two distinct human attestations:
1. Extraction/structure review, tied to the exact section-text SHA-256.
2. Legal-content review, tied to the same hash and requiring a named reviewer,
   timestamp, authoritative-source reference, and review notes.

Use `approve_legal_content_record` in `src.document.legal_review` only after
checking the provision and its legal status against the cited authoritative
source. Any review reasons on the manifest must be explicitly listed as
resolved; missing or duplicate section identities cannot pass the gate.

The candidate gate can be called from Python after loading the current section
JSONL and review manifest:

```python
from pathlib import Path
from src.document.legal_review import load_jsonl
from src.training.legal_training_gate import evaluate_training_candidates

sections = load_jsonl(Path("data/processed/Indian_Contract_Act_1872_sections_v2.jsonl"))
manifest = load_jsonl(Path("data/review/Indian_Contract_Act_1872_review_v2.jsonl"))
report = evaluate_training_candidates(sections, manifest)
print(report)
```

Only sections with both valid attestations, matching current text hashes,
explicit `in_force`/ `active` status, and resolved review flags are listed as
candidates. Unknown and repealed sections are blocked. A manifest integrity
failure blocks all sections.

**Candidate does not mean training-ready.** This gate does not independently
verify the cited authority, export or filter a dataset, modify QA splits, or
start training. It always reports `ready_for_training: false` and
`legal_accuracy_verified: false`. Separate QA dataset, benchmark, and final
training approvals remain necessary. No training is triggered automatically.


## Chunk provenance and review-history audit

Generate chunks with the current review manifest attached so each chunk preserves the extraction reviewer, legal-content reviewer, approval timestamps, text hashes, authoritative-source reference, resolution notes, and resolved review flags. The chunker validates manifest coverage before attaching it.

```python
from pathlib import Path
from src.document.legal_review import load_jsonl
from src.document.legal_chunker import create_legal_chunks, write_jsonl_new

sections = load_jsonl(Path("data/processed/Indian_Contract_Act_1872_sections_v2.jsonl"))
manifest = load_jsonl(Path("data/review/Indian_Contract_Act_1872_review_v2.jsonl"))
chunks = create_legal_chunks(sections, review_manifest=manifest)
write_jsonl_new(chunks, Path("data/processed/Indian_Contract_Act_1872_reviewed_chunks_v2.jsonl"))
```

Then run the read-only audit against those three matching artifacts:

```powershell
python -m src.training.legal_chunk_provenance_audit --sections "data/processed/Indian_Contract_Act_1872_sections_v2.jsonl" --chunks "data/processed/Indian_Contract_Act_1872_reviewed_chunks_v2.jsonl" --review-manifest "data/review/Indian_Contract_Act_1872_review_v2.jsonl" --output "reports/Indian_Contract_Act_1872_chunk_provenance_v2.json"
```

The audit checks source section hashes, page ranges, page/line coordinates, source excerpts, raw source lines, exact chunk-text reconstruction, amendment/footnote/OCR annotations, section identity/status, and carried review-history fields. It also confirms that every section passing the two-stage review gate has at least one chunk. Ambiguous or unreviewed sections are blocked from the candidate set.

The report is created at a new path and never overwrites existing data. A passing provenance report is still not legal accuracy verification, QA dataset approval, or permission to train. It never exports a training dataset or starts model training.


## Human-reviewed legal QA dataset builder

The QA builder does **not** generate questions or answers automatically. A
human must write each question and answer, identify an exact supporting quote
from a reviewed legal chunk, and approve that exact QA payload. The approval
hash covers the question, answer, evidence quote, and chunk ID; changing any
of them invalidates the approval.

Keep three separate draft files. Every record must contain:

- `question`, `answer`, `evidence_quote`, and `chunk_id`
- `qa_review_status: "qa_review_approved"`
- `qa_reviewed_by` and `qa_reviewed_at`
- `qa_reviewed_content_sha256`, calculated with
  `qa_review_hash(record)` from `src.training.legal_qa_dataset_builder`

Example approval preparation (the human reviewer must review the content
before recording this attestation):

```python
from src.training.legal_qa_dataset_builder import qa_review_hash

draft = {
    "question": "When is an agreement a contract?",
    "answer": "A valid agreement becomes a contract when it is enforceable by law.",
    "evidence_quote": "A valid agreement is enforceable by law.",
    "chunk_id": "legal_000001",
    "qa_review_status": "qa_review_approved",
    "qa_reviewed_by": "Reviewer name or role",
    "qa_reviewed_at": "2026-11-12T14:00:00Z",
}
draft["qa_reviewed_content_sha256"] = qa_review_hash(draft)
```

Build three splits without random splitting or overwriting existing files:

```powershell
python -m src.training.legal_qa_dataset_builder --sections "data/processed/Indian_Contract_Act_1872_sections_v2.jsonl" --chunks "data/processed/Indian_Contract_Act_1872_reviewed_chunks_v2.jsonl" --review-manifest "data/review/Indian_Contract_Act_1872_review_v2.jsonl" --train-drafts "data/qa_reviewed/train_drafts.jsonl" --validation-drafts "data/qa_reviewed/validation_drafts.jsonl" --test-drafts "data/qa_reviewed/test_drafts.jsonl" --output-dir "data/qa_reviewed/built_v2"
```

Before building, it rechecks the two-stage section gate and chunk provenance.
It rejects unreviewed or modified QA drafts, evidence quotes not found exactly
in the referenced chunk, non-candidate/repealed sections, duplicate questions,
and any section appearing in more than one split. All examples from one legal
section stay together to reduce section-level leakage. Output records preserve
the source context, section, page/line citations, authority reference, and
reviewer history.

This is a controlled dataset-construction step, not an automatic legal QA
generator. A passing build still does not independently prove that an answer
is legally correct or authorize training. The builder always reports
`ready_for_training: false`; it does not touch existing QA v2 splits or
model checkpoints.
