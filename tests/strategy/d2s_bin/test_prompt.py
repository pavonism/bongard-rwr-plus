from src.strategy.d2s_bin.model import (
    BongardImage,
    DescriptionsToSidesRequest,
)
from src.strategy.d2s_bin.prompt import PromptFactory

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
    assert len(prompt) == 5
