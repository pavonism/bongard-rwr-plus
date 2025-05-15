import json
import os
from typing import List

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

from src.messenger.content import Content, TextContent
from src.strategy.selection.model import SelectionInput, SelectionRequest


class PromptFactory:
    def __init__(
        self,
        prompt_version: str,
        source_dir: str = "resources/prompts/selection",
    ):
        env = Environment(
            loader=FileSystemLoader(os.path.join(source_dir, prompt_version))
        )
        self._shared_prompt = env.get_template("selection-shared.jinja").render()
        self._selection_prompt_1 = env.get_template("selection-1.jinja").render()
        self._selection_prompt_2 = env.get_template("selection-2.jinja")

    def select(self, input: SelectionInput, request: SelectionRequest) -> List[Content]:
        return [
            TextContent(self._shared_prompt),
            TextContent(self._selection_prompt_1),
            *input.contents(),
            TextContent(self._selection_prompt_2.render(request=self._to_str(request))),
        ]

    def _to_str(self, model: BaseModel) -> str:
        return json.dumps(model.model_dump(), default=pydantic_encoder, indent=4)
