from src.strategy.d1s.model import (
    BongardImage,
    DescriptionToSideRequest,
)
from src.strategy.d1s.prompt import PromptFactory

request = DescriptionToSideRequest(
    left_descriptions=["left description 1", "left description 2"],
    right_descriptions=["right description 1", "right description 2"],
    test_description="description",
)


def test_prompt_factory_predict():
    factory = PromptFactory(prompt_version="v1")
    prompt = factory.predict(request)
    assert len(prompt) == 3
