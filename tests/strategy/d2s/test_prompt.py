from src.messenger.content import ImageContent
from src.strategy.d2s.model import (
    BongardImage,
    DescriptionsToSidesRequest,
)
from src.strategy.d2s.prompt import PromptFactory

request = DescriptionsToSidesRequest(
    left_descriptions=["left description 1", "left description 2"],
    right_descriptions=["right description 1", "right description 2"],
    first="first description",
    second="second description",
    whole_image="./test_image.png",
)


def test_prompt_factory_predict():
    factory = PromptFactory(prompt_version="v1")
    prompt = factory.predict(request)
    assert len(prompt) == 6
    assert all([not isinstance(content, ImageContent) for content in prompt])


def test_prompt_with_image_factory_predict():
    factory = PromptFactory(prompt_version="v3")
    prompt = factory.predict(request)
    assert len(prompt) == 6
    assert any([isinstance(content, ImageContent) for content in prompt])
