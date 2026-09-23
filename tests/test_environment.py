import torch
import numpy as np
import pandas as pd
import sentencepiece
import sklearn

def main():
    print("=" * 50)
    print("LawLM Environment Test")
    print("=" * 50)
    print("PyTorch:", torch.__version__)
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("SentencePiece: OK")
    print("Scikit-learn:", sklearn.__version__)
    print("Device: CPU")
    print("CUDA available:", torch.cuda.is_available())
    print("=" * 50)
    print("LawLM Phase 1 environment is ready!")
    print("=" * 50)

if __name__ == "__main__":
    main()
