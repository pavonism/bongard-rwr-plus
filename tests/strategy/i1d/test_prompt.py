from src.strategy.i1d.model import BongardImage
from src.strategy.i1d.prompt import PromptFactory


image = BongardImage(
    image_id=1,
    path="path/to/image.png",
)


def test_prompt_factory_describe():
    factory = PromptFactory(prompt_version="v1")
    prompt = factory.describe(image)
    assert len(prompt) == 3
