import sys
from pathlib import Path

import sentencepiece as spm
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lawlm_model import LawLM


TOKENIZER_FILE = PROJECT_ROOT / "tokenizer/lawlm.model"
CHECKPOINT_FILE = PROJECT_ROOT / "experiments/experiment_10_epoch_best.pt"

app = FastAPI(
    title="LawLM API",
    description="Legal-domain language model trained from scratch.",
    version="1.0.0",
)

model = None
tokenizer = None


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=60, ge=1, le=200)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=8000)


def load_model():
    global model, tokenizer

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_FILE}")

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(f"Final checkpoint not found: {CHECKPOINT_FILE}")

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_FILE)
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    model = LawLM(
        vocab_size=checkpoint.get("vocab_size", 8000),
        block_size=checkpoint.get("block_size", 128),
        embed_dim=checkpoint.get("embed_dim", 256),
        num_heads=checkpoint.get("num_heads", 4),
        num_layers=checkpoint.get("num_layers", 4),
        ff_hidden_dim=checkpoint.get("ff_hidden_dim", 1024),
        dropout=checkpoint.get("dropout", 0.1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
def root():
    return {
        "name": "LawLM",
        "status": "running",
        "model": "5.24M parameter Transformer trained from scratch",
        "disclaimer": "Generated text is for research and educational purposes only and is not legal advice.",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/generate")
def generate(request: GenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    prompt_ids = tokenizer.encode(
        request.prompt,
        out_type=int,
    )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

    generated_text = tokenizer.decode(
        output_ids[0].tolist()
    )

    return {
        "prompt": request.prompt,
        "generated_text": generated_text,
        "temperature": request.temperature,
        "top_k": request.top_k,
        "disclaimer": "Generated text is for research and educational purposes only and is not legal advice.",
    }
