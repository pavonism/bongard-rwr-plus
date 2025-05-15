from src.strategy.i2s.model import BongardProblemImages
from src.strategy.i2s.prompt import PromptFactory

images = BongardProblemImages(
    whole_image_path="data/raw/synthetic/1/whole-without-test.png",
    first_test_image_path="data/raw/synthetic/1/left/5.png",
    second_test_image_path="data/raw/synthetic/1/right/5.png",
)


def test_prompt_factory():
    factory = PromptFactory(prompt_version="v1")
    prompt = factory.predict(images)
    assert len(prompt) > 5
