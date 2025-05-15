import os
from typing import List

from jinja2 import Environment, FileSystemLoader

from src.messenger.content import Content, TextContent
from src.strategy.i1d.model import BongardImage


class PromptFactory:
    def __init__(self, prompt_version: str, source_dir: str = "resources/prompts/i1d"):
        env = Environment(
            loader=FileSystemLoader(os.path.join(source_dir, prompt_version))
        )

        self._describe_prompt_1 = env.get_template("i1d-describe-1.jinja").render()
        self._describe_prompt_2 = env.get_template("i1d-describe-2.jinja").render()

    def describe(
        self,
        image: BongardImage,
    ) -> List[Content]:
        return [
            TextContent(self._describe_prompt_1),
            image.content(),
            TextContent(self._describe_prompt_2),
        ]
