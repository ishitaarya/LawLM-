import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lawlm_model import LawSuitLLM

app = FastAPI(
    title="LawSuit LLM API",
    description="Development API for the from-scratch LawSuit Transformer.",
    version="0.2.0",
)

model = LawSuitLLM()
model.eval()


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=40, ge=1, le=200)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=8000)


@app.get("/")
def root():
    return {
        "name": "LawSuit LLM",
        "status": "running",
        "model": "randomly initialized ~10M-parameter target Transformer",
        "parameters": model.parameter_count(),
        "device": next(model.parameters()).device.type,
        "pretrained": False,
        "phase": 1,
        "disclaimer": (
            "Development/educational API. Output is not legal advice "
            "and the model is not yet trained."
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "parameters": model.parameter_count(),
    }


@app.post("/generate")
def generate(request: GenerationRequest):
    if request.top_k > model.vocab_size:
        request.top_k = model.vocab_size

    # Phase 1 smoke-test API: generation uses the model's random weights.
    # A real tokenizer/checkpoint is introduced in later phases.
    token_ids = torch.tensor([[0]], dtype=torch.long)

    with torch.no_grad():
        output_ids = model.generate(
            token_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

    return {
        "prompt": request.prompt,
        "token_ids": output_ids[0].tolist(),
        "trained": False,
        "parameters": model.parameter_count(),
        "message": (
            "Phase 1 smoke-test output. Legal training and tokenizer "
            "integration are implemented in later phases."
        ),
    }
