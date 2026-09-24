from pathlib import Path


CONFIG = Path("configs/project.yaml")


def test_phase1_config_exists():
    assert CONFIG.exists()


def test_seven_phases_are_defined():
    text = CONFIG.read_text(encoding="utf-8")
    for phase in (
        "phase_1:",
        "phase_2:",
        "phase_3:",
        "phase_4:",
        "phase_5:",
        "phase_6:",
        "phase_7:",
    ):
        assert phase in text


def test_initial_model_target_is_10m():
    text = CONFIG.read_text(encoding="utf-8")
    assert "parameters_target: 10000000" in text
    assert "pretrained_llm: false" in text


def test_model_configuration_is_present():
    text = CONFIG.read_text(encoding="utf-8")
    for value in (
        "vocabulary_size: 8000",
        "context_length: 256",
        "embedding_dim: 384",
        "layers: 4",
        "attention_heads: 6",
        "ffn_dim: 1536",
        'initialization: "random"',
    ):
        assert value in text


def test_data_layers_are_defined():
    text = CONFIG.read_text(encoding="utf-8")
    for directory in (
        "data/raw",
        "data/extracted",
        "data/cleaned",
        "data/deduplicated",
        "data/structured",
        "data/tokenized",
        "data/train",
        "data/validation",
        "data/test",
    ):
        assert directory in text
