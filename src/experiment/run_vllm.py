import asyncio
from argparse import ArgumentParser

from dotenv import load_dotenv

from src.experiment.base import run
from src.messenger.vllm_messenger import VllmMessengerFactory

with open("resources/chat_templates/deepseek_vl2.jinja", "r") as f:
    DEEPSEEK_CHAT_TEMPLATE = "".join([line.strip() for line in f.readlines()])
with open("resources/chat_templates/llava_v1.6_vicuna.jinja", "r") as f:
    LLAVA_V16_VICUNA_CHAT_TEMPLATE = "".join([line.strip() for line in f.readlines()])

CUSTOM_ARGS = {
    "mistralai/Pixtral-12B-2409": ["--tokenizer_mode", "mistral"],
    "mistralai/Pixtral-Large-Instruct-2411": [
        "--tokenizer_mode",
        "mistral",
        "--config-format",
        "mistral",
        "--load-format",
        "mistral",
    ],
    "llava-hf/llava-v1.6-vicuna-13b-hf": [
        "--chat-template",
        LLAVA_V16_VICUNA_CHAT_TEMPLATE,
        "--max-num-batched-tokens",
        "4096",
    ],
    "llava-hf/llava-next-72b-hf": ["--gpu-memory-utilization", "0.98"],
    "llava-hf/llava-next-110b-hf": ["--gpu-memory-utilization", "0.98"],
    "Qwen/Qwen2-VL-72B-Instruct": [
        "--gpu-memory-utilization",
        "0.95",
        "--max-num-seqs",
        "128",
    ],
    **{
        model: [
            "--enable-reasoning",
            "--reasoning-parser",
            "deepseek_r1",
        ]
        for model in [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        ]
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": [
        "--enable-reasoning",
        "--reasoning-parser",
        "deepseek_r1",
        "--gpu-memory-utilization",
        "0.98",
    ],
    "deepseek-ai/deepseek-vl2-tiny": [
        "--chat-template",
        DEEPSEEK_CHAT_TEMPLATE,
        "--hf-overrides",
        '{"architectures": ["DeepseekVLV2ForCausalLM"]}',
    ],
    "deepseek-ai/deepseek-vl2-small": [
        "--chat-template",
        DEEPSEEK_CHAT_TEMPLATE,
        "--hf-overrides",
        '{"architectures": ["DeepseekVLV2ForCausalLM"]}',
    ],
    "deepseek-ai/deepseek-vl2": [
        "--chat-template",
        DEEPSEEK_CHAT_TEMPLATE,
        "--hf-overrides",
        '{"architectures": ["DeepseekVLV2ForCausalLM"]}',
    ],
    "OpenGVLab/InternVL2_5-38B": [
        "--gpu-memory-utilization",
        "0.95",
    ],
    "OpenGVLab/InternVL2_5-78B": [
        "--gpu-memory-utilization",
        "0.95",
        "--max-num-batched-tokens",
        "98304",
    ],
}

max_tokens_override = {
    "mistralai/Pixtral-12B-2409": 32768,  # max 128k
    **{f"OpenGVLab/InternVL2-{size}": 8192 for size in ["4B", "8B"]},  # max 8k
    **{f"OpenGVLab/InternVL2_5-{size}": 8192 for size in ["4B", "8B"]},  # max 8k
    "llava-hf/llava-v1.6-mistral-7b-hf": 8192,  # not sure, likely 8k
    "llava-hf/llava-v1.6-vicuna-13b-hf": 4096,
    "llava-hf/llava-v1.6-34b-hf": 4096,
    "llava-hf/llava-next-72b-hf": 4096,
    "llava-hf/llava-next-110b-hf": 16384,  # max 32k
    "microsoft/Phi-3.5-vision-instruct": 32768,  # max 128k
    "deepseek-ai/deepseek-vl2-tiny": 4096,
    "deepseek-ai/deepseek-vl2-small": 4096,
    "deepseek-ai/deepseek-vl2": 4096,
    "Qwen/Qwen2-VL-72B-Instruct": 16384,
    **{
        model: 32768
        for model in [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ]  # max 128k
    },
}


def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--descriptor-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v1",
    )
    parser.add_argument("--data-dir", type=str, default="/app/data")
    parser.add_argument("--labels-file", type=str, default="/app/data/labels.csv")
    parser.add_argument("--problem-ids-range-start", type=int, default=1)
    parser.add_argument("--problem-ids-range-end", type=int, default=101)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--num-choices",
        type=int,
        default=[2],
        nargs="+",
        help="List of class counts for multiclass classification (e.g. 2 4 6).",
    )
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--reevaluate", action="store_true")
    return parser


async def main(args):
    max_tokens = (
        max_tokens_override[args.model] if args.model in max_tokens_override else 8192
    )
    custom_args = CUSTOM_ARGS[args.model] if args.model in CUSTOM_ARGS else []
    if args.tensor_parallel_size > 1:
        custom_args.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    print(f"custom_args: {custom_args}")

    limit_mm_per_prompt = 0 if "R1" in args.model else 4

    messengers = VllmMessengerFactory(
        model_name=args.model,
        max_tokens=max_tokens,
        limit_mm_per_prompt=limit_mm_per_prompt,
        custom_args=custom_args,
    ).make_messengers(
        args.agents,
        max_output_tokens=16384 if "R1" in args.model else 2048,
        temperature=args.temperature,
    )

    problem_ids_range_start = args.problem_ids_range_start
    problem_ids_range_end = args.problem_ids_range_end
    problem_ids = list(range(problem_ids_range_start, problem_ids_range_end))

    await run(
        args,
        messengers,
        problem_ids,
        reevaluate=args.reevaluate,
    )

    print("All experiments finished.")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(args)
    load_dotenv()
    asyncio.run(main(args))
