from src.strategy.selection.model import (
    Concept,
    SelectionInput,
    SelectionRequest,
)
from src.strategy.selection.prompt import PromptFactory

input = SelectionInput(image_path="data/raw/synthetic/1/whole-without-test.png")
request = SelectionRequest(
    concepts=[
        Concept(left="l1", right="r1", label=1),
        Concept(left="l2", right="r2", label=2),
    ],
)


def test_prompt_factory():
    factory = PromptFactory(prompt_version="v1")
    prompt = factory.select(input, request)
    assert len(prompt) == 5
