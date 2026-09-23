import sys
from pathlib import Path

import sentencepiece as spm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lawlm_model import LawLM


MODEL_FILE = Path("tokenizer/lawlm.model")
CHECKPOINT_FILE = Path("checkpoints/best.pt")

TEMPERATURE = 0.8
TOP_K = 40
MAX_NEW_TOKENS = 80


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True,
    )

    model = LawLM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def generate_text(model, tokenizer, prompt):
    prompt_ids = tokenizer.encode(prompt, out_type=int)

    if not prompt_ids:
        raise ValueError("Prompt produced no tokens.")

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
    )

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    text = tokenizer.decode(generated_ids[0].tolist())

    return text


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found: {MODEL_FILE}"
        )

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_FILE}"
        )

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(MODEL_FILE)
    )

    model = load_model()

    prompts = [
        "The court may",
        "Under the applicable law,",
        "A person who commits an offence",
    ]

    print("=" * 60)
    print("LawLM - Legal Text Generation")
    print("=" * 60)
    print(f"Temperature:    {TEMPERATURE}")
    print(f"Top-k:          {TOP_K}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    for prompt in prompts:
        print("-" * 60)
        print(f"Prompt: {prompt}")
        print()
        print("Generated:")
        print(generate_text(model, tokenizer, prompt))
        print()

    print("=" * 60)
    print("Generation completed.")
    print("Disclaimer: Generated text is for research and")
    print("educational purposes only and is NOT legal advice.")
    print("=" * 60)


if __name__ == "__main__":
    main()
