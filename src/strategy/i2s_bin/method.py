import asyncio
import os
from pathlib import Path
import random
from typing import List, Optional, Tuple

from tqdm import tqdm

from src.dataset.model import BongardDatasetInfo, BongardProblem
from src.messenger.vllm_messenger import VllmMessenger
from src.strategy.i2s_bin.model import (
    ImagesToSidesExperiment,
    BongardProblemImages,
    ImagesToSidesResponse,
    ImagesToSidesExperimentInstance,
    Answer,
)
from src.strategy.i2s_bin.prompt import PromptFactory


async def resolve_images_to_sides_binary(
    problem_ids: List[int],
    dataset_path: str,
    experiment_file: str,
    prompt_version: str,
    messengers: List[VllmMessenger],
    reevaluate: bool = False,
):
    if os.path.exists(experiment_file) and not reevaluate:
        experiment = ImagesToSidesExperiment.from_file(experiment_file)
    else:
        experiment = ImagesToSidesExperiment(solver_name=messengers[0].get_name())

    dataset = BongardDatasetInfo.from_directory(dataset_path).subset(problem_ids)

    prompt_factory = PromptFactory(prompt_version)
    progress_bar = tqdm(total=len(dataset.problems))

    current_tasks = [
        asyncio.create_task(
            resolve_problem(
                problem=dataset.problems[i],
                experiment=experiment,
                messenger=messenger,
                prompt_factory=prompt_factory,
                reevaluate=reevaluate,
            )
        )
        for i, messenger in enumerate(messengers)
        if i < len(dataset.problems)
    ]

    current_problem_index = len(messengers)

    while len(current_tasks) > 0:
        done, pending = await asyncio.wait(
            current_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        available_messengers: List[VllmMessenger] = []

        for done_task in done:
            messenger, instance = done_task.result()
            available_messengers.append(messenger)
            progress_bar.update(1)

            if instance is not None:
                experiment.add_solution(instance)

        current_tasks = list(pending)

        if current_problem_index < len(dataset.problems):
            current_tasks = current_tasks + [
                asyncio.create_task(
                    resolve_problem(
                        problem=dataset.problems[current_problem_index + i],
                        experiment=experiment,
                        messenger=messenger,
                        prompt_factory=prompt_factory,
                        reevaluate=reevaluate,
                    )
                )
                for i, messenger in enumerate(available_messengers)
                if current_problem_index + i < len(dataset.problems)
            ]

            current_problem_index += len(available_messengers)

    os.makedirs(Path(experiment_file).parent, exist_ok=True)
    experiment.to_file(experiment_file)


async def resolve_problem(
    problem: BongardProblem,
    experiment: ImagesToSidesExperiment,
    messenger: VllmMessenger,
    prompt_factory: PromptFactory,
    reevaluate: bool,
) -> Tuple[VllmMessenger, Optional[ImagesToSidesExperimentInstance]]:
    last_left_image = problem.left_images[-1].path
    last_right_image = problem.right_images[-1].path

    if (
        experiment.has_solution(
            BongardProblemImages(
                whole_image_path=problem.whole_image,
                first_test_image_path=last_left_image,
                second_test_image_path=last_right_image,
            )
        )
        and not reevaluate
    ):
        print(f"Problem {problem.id} already has solution. Skipping.")
        return messenger, None

    test_image_paths = [last_left_image, last_right_image]
    answers = [Answer.LEFT_RIGHT, Answer.RIGHT_LEFT]
    expected_answer_index = random.choice(range(len(answers)))

    images = BongardProblemImages(
        whole_image_path=problem.whole_image,
        first_test_image_path=test_image_paths[expected_answer_index],
        second_test_image_path=test_image_paths[1 - expected_answer_index],
    )

    try:
        response: ImagesToSidesResponse = await messenger.ask_structured(
            prompt_factory.predict(images),
            schema=ImagesToSidesResponse,
        )

        return messenger, ImagesToSidesExperimentInstance(
            problem_id=problem.id,
            images=images,
            response=response,
            expected_answer=answers[expected_answer_index],
        )

    except Exception as e:
        print(f"problem_id: {problem.id} - Failed to parse json from model response:")
        print(f"Exception: {e}")

    return messenger, None
