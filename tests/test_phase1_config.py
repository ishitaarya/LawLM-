from pathlib import Path

def test_phase1_directories_are_documented():
    config = Path("configs/project.yaml")
    assert config.exists()
    text = config.read_text(encoding="utf-8")
    for directory in (
        "data/raw", "data/extracted", "data/cleaned",
        "data/deduplicated", "data/structured", "data/tokenized",
        "data/train", "data/validation", "data/test",
    ):
        assert directory in text

def test_initial_model_is_from_scratch():
    text = Path("configs/project.yaml").read_text(encoding="utf-8")
    assert 'pretrained_model: false' in text
    assert 'initialization: "random"' in text