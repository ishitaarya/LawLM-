"""Tests for the Phase 7.5 legal QA dataset seed."""

from src.training.qa_dataset import build_section25_examples


def main() -> None:
    examples = build_section25_examples()

    assert len(examples) == 3
    assert all(example.section == "25" for example in examples)
    assert all(example.question.strip() for example in examples)
    assert all(example.context.strip() for example in examples)
    assert all(example.answer.strip() for example in examples)

    print("=" * 72)
    print("LawSuit LLM - Phase 7.5 QA Dataset Tests")
    print("=" * 72)
    print(f"Seed examples: {len(examples)}")
    print("Question/context/answer fields: passed")
    print("Section labels: passed")
    print("All Phase 7.5 dataset tests passed!")
    print("=" * 72)


if __name__ == "__main__":
    main()
