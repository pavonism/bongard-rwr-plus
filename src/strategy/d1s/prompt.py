import os
from typing import List

from jinja2 import Environment, FileSystemLoader

from src.messenger.content import Content, TextContent
from src.strategy.d2s.model import DescriptionsToSidesRequest


class PromptFactory:
    def __init__(self, prompt_version: str, source_dir: str = "resources/prompts/d1s"):
        env = Environment(
            loader=FileSystemLoader(os.path.join(source_dir, prompt_version))
        )

        self._shared_prompt = env.get_template("shared.jinja").render()
        self._predict_prompt_1 = env.get_template("d1s-predict-1.jinja").render()
        self._predict_prompt_2 = env.get_template("d1s-predict-2.jinja")

    def predict(
        self,
        request: DescriptionsToSidesRequest,
    ) -> List[Content]:
        return [
            TextContent(self._shared_prompt),
            TextContent(self._predict_prompt_1),
            TextContent(
                self._predict_prompt_2.render(
                    request=request.model_dump_json(indent=4),
                )
            ),
        ]
