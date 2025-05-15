import os
from typing import List

from jinja2 import Environment, FileSystemLoader
from pydantic.json import pydantic_encoder

from src.messenger.content import Content, TextContent
from src.strategy.i2s.model import BongardProblemImages


class PromptFactory:
    def __init__(self, prompt_version: str, source_dir: str = "resources/prompts/i2s"):
        env = Environment(
            loader=FileSystemLoader(os.path.join(source_dir, prompt_version))
        )
        self._shared_prompt = env.get_template("shared.jinja").render()
        self._predict_prompt_1 = env.get_template("i2s-1.jinja").render()
        self._predict_prompt_2 = env.get_template("i2s-2.jinja").render()
        self._retry_prompt_1 = env.get_template("i2s-retry-1.jinja").render()
        self._retry_prompt_2 = env.get_template("i2s-retry-2.jinja").render()

    def predict(self, images: BongardProblemImages) -> List[Content]:
        return [
            TextContent(self._shared_prompt),
            TextContent(self._predict_prompt_1),
            *images.contents(),
            TextContent(self._predict_prompt_2),
        ]

    def retry(self) -> List[Content]:
        return [
            TextContent(self._retry_prompt_1),
            TextContent(self._retry_prompt_2),
        ]
