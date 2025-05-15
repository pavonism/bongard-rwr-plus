import os
from typing import List

from jinja2 import Environment, FileSystemLoader

from src.messenger.content import Content, TextContent
from src.strategy.d2s_bin.model import DescriptionsToSidesRequest


class PromptFactory:
    def __init__(
        self, prompt_version: str, source_dir: str = "resources/prompts/d2s-bin"
    ):
        env = Environment(
            loader=FileSystemLoader(os.path.join(source_dir, prompt_version))
        )

        self._shared_prompt = env.get_template("shared.jinja").render()
        self._predict_prompt_1 = env.get_template("d2s-bin-predict-1.jinja").render()
        self._predict_prompt_2 = env.get_template("d2s-bin-predict-2.jinja").render()
        self._predict_prompt_3 = env.get_template("d2s-bin-predict-3.jinja").render()

    def predict(
        self,
        request: DescriptionsToSidesRequest,
    ) -> List[Content]:
        return [
            TextContent(self._shared_prompt),
            TextContent(self._predict_prompt_1),
            TextContent(self._predict_prompt_2),
            TextContent(request.model_dump_json(indent=4)),
            TextContent(self._predict_prompt_3),
        ]
