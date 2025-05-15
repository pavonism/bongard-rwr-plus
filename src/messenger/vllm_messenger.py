import subprocess
import time
from typing import List, Optional, Dict, Type

import openai
import portpicker
import requests
from pydantic import BaseModel

from src.messenger.content import Content, ImageContent, TextContent
from src.messenger.formatter import OpenaiFormatter
from src.messenger.llm_messenger import LLMMessenger


class VllmMessenger(LLMMessenger):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        has_reasoning_content: bool = False,
        temperature: float = 1.0,
        max_output_tokens: int = 1536,
        log_directory: str = "",
        log_suffix: str = "",
    ):
        super().__init__(model_name, temperature, log_directory, log_suffix)

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.has_reasoning_content = has_reasoning_content

        self.client = openai.AsyncClient(base_url=f"{base_url}/v1", api_key=api_key)
        self.formatter = OpenaiFormatter()
        self._context: Optional[List[Dict]] = None  # set type explicitly

    def __update_context(self, contents: List[Content], model_response: str):
        if self._context is not None:
            if not self._keep_image_history:
                contents = [
                    content
                    for content in contents
                    if not isinstance(content, ImageContent)
                ]
            self._context += [self.formatter.user(contents)]
            self._context += [self.formatter.assistant(model_response)]

    async def ask(
        self,
        contents: List[Content],
        schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        if self._context is None:
            self.log("INFO", [TextContent("Opening new context")])

        self.log("USER", contents)

        context = self.get_context()
        message = self.formatter.user(contents)
        messages = context + [message]

        response = await self.client.chat.completions.create(
            model=self.get_name(),
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            extra_body={"guided_json": schema.model_json_schema()} if schema else None,
        )

        self._log_reasoning_content(response)

        model_response = response.choices[0].message.content
        if model_response:
            model_response = model_response.strip()
        print(model_response)
        self.log("ASSISTANT", [TextContent(model_response)])

        self.__update_context(contents, model_response)
        return model_response

    async def ask_structured(
        self, contents: List[Content], schema: Type[BaseModel]
    ) -> Optional[BaseModel]:
        if self._context is None:
            self.log("INFO", [TextContent("Opening new context")])

        self.log("USER", contents)

        context = self.get_context()
        message = self.formatter.user(contents)
        messages = context + [message]

        response = await self.client.beta.chat.completions.parse(
            model=self.get_name(),
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            response_format=schema,
            extra_body=dict(guided_decoding_backend="outlines"),
        )

        self._log_reasoning_content(response)

        model_response = response.choices[0].message
        if model_response.parsed:
            print(model_response.parsed)
            self.log("ASSISTANT", [TextContent(str(model_response))])
            self.__update_context(contents, str(model_response))
            return model_response.parsed
        else:
            print(f"Failed to parse model response: {model_response}")
            return None

    def _log_reasoning_content(self, response):
        if self.has_reasoning_content and hasattr(
            response.choices[0].message, "reasoning_content"
        ):
            reasoning_content = response.choices[0].message.reasoning_content

            if reasoning_content:
                reasoning_content = reasoning_content.strip()
                print("REASONING_CONTENT: ", reasoning_content)
                self.log("ASSISTANT", [TextContent(reasoning_content)])


class VllmMessengerFactory:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 2048,
        limit_mm_per_prompt: int = 4,
        custom_args: List[str] = [],
    ):
        port = portpicker.pick_unused_port()

        self.api_key = "NOT-USED"
        self.base_url = f"http://localhost:{port}"

        self.model_name = model_name
        self.has_reasoning_content = "--enable-reasoning" in custom_args
        self.max_tokens = max_tokens

        self.process = popen_launch_server(
            model_name,
            self.base_url,
            timeout=7200,
            api_key=self.api_key,
            other_args=(
                "--port",
                str(port),
                "--max-model-len",
                str(max_tokens),
                "--trust-remote-code",
                # classify_picked_images_to_sides uses 4 images (2 per side + 2 test instances)
                *(
                    ("--limit-mm-per-prompt", f"image={limit_mm_per_prompt}")
                    if limit_mm_per_prompt > 0
                    else ()
                ),
                "--guided-decoding-backend",
                "outlines",
                *custom_args,
            ),
        )

    def make_messengers(
        self,
        n: int = 1,
        temperature: float = 1.0,
        max_output_tokens: int = 1536,
    ) -> List[VllmMessenger]:
        assert max_output_tokens < self.max_tokens

        return [
            VllmMessenger(
                base_url=self.base_url,
                api_key=self.api_key,
                model_name=self.model_name,
                has_reasoning_content=self.has_reasoning_content,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                log_suffix=f"-agent-{index}",
            )
            for index in range(n)
        ]


def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
    api_key: str,
    other_args: tuple = (),
    env: Optional[dict] = None,
    return_stdout_stderr: bool = False,
):
    command = ["vllm", "serve", model, "--api-key", api_key, *other_args]
    if return_stdout_stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
    else:
        process = subprocess.Popen(command, stdout=None, stderr=None, env=env)

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {api_key}",
            }
            response = requests.get(f"{base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                return process
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")
