import os
from typing import List
from tqdm import tqdm

from src.dataset.model import (
    BongardDatasetInfo,
    BongardImage,
)
from src.image import is_image_supported
from src.messenger.vllm_messenger import VllmMessenger
from src.strategy.i1d.model import (
    BongardDatasetDescriptions,
    ImageDescription,
)
from src.strategy.i1d.prompt import PromptFactory
import traceback


async def generate_descriptions(
    problem_ids: List[int],
    dataset_path: str,
    descriptions_file: str,
    prompt_version: str,
    messengers: List[VllmMessenger],
    reevaluate: bool = False,
):
    dataset = BongardDatasetInfo.from_directory(dataset_path).subset(problem_ids)

    if os.path.exists(descriptions_file) and not reevaluate:
        descriptions = BongardDatasetDescriptions.from_file(descriptions_file)
    else:
        descriptions = BongardDatasetDescriptions(
            descriptor_name=messengers[0].get_name()
        )

    prompt_factory = PromptFactory(prompt_version)
    images = dataset.collect_images()
    descriptions_dict = descriptions.to_dict()

    for image in tqdm(images):
        if not is_image_supported(image.path):
            print(f"Unsupported extension: {image.path}. Skipping.")
            continue
        if descriptions_dict.has_image_description(image) and not reevaluate:
            continue

        description = await get_description(
            prompt_factory=prompt_factory,
            image=image,
            # TODO: Use all messengers
            model=messengers[0],
        )

        descriptions_dict.add_description(
            ImageDescription(
                image_id=image.image_id,
                path=image.path,
                description=description,
            )
        )

    descriptions_dict.flatten().to_file(descriptions_file)


async def get_description(
    prompt_factory: PromptFactory,
    image: BongardImage,
    model: VllmMessenger,
) -> str:
    try:
        description = await model.ask(
            contents=prompt_factory.describe(image),
        )
        return description
    except Exception:
        print(f"Failed to get description for image: {image.path}")
        print(traceback.format_exc())
        return ""
