import asyncio
import os
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from src.dataset.model import BongardDatasetInfo, BongardProblem
from src.messenger.vllm_messenger import VllmMessenger
from src.strategy.i1s.model import (
    ImageToSideExperiment,
    BongardProblemImages,
    ImageToSideResponse,
    ImageToSideExperimentInstance,
    Answer,
)
from src.strategy.i1s.prompt import PromptFactory


async def resolve_image_to_side(
    problem_ids: List[int],
    dataset_path: str,
    experiment_file: str,
    prompt_version: str,
    messengers: List[VllmMessenger],
    reevaluate: bool = False,
):
    if os.path.exists(experiment_file) and not reevaluate:
        experiment = ImageToSideExperiment.from_file(experiment_file)
    else:
        experiment = ImageToSideExperiment(solver_name=messengers[0].get_name())

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
            messenger, instances = done_task.result()
            available_messengers.append(messenger)
            progress_bar.update(1)
            for instance in instances:
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
    experiment: ImageToSideExperiment,
    messenger: VllmMessenger,
    prompt_factory: PromptFactory,
    reevaluate: bool,
) -> Tuple[VllmMessenger, List[ImageToSideExperimentInstance]]:
    test_image_paths = [problem.left_images[-1].path, problem.right_images[-1].path]
    expected_answers = [Answer.LEFT, Answer.RIGHT]
    instances = []
    for test_image_path, expected_answer in zip(test_image_paths, expected_answers):
        images = BongardProblemImages(
            whole_image_path=problem.whole_image,
            test_image_path=test_image_path,
        )
        if experiment.has_solution(images) and not reevaluate:
            print(f"Problem {problem.id} already has solution. Skipping.")
            continue

        try:
            response: ImageToSideResponse = await messenger.ask_structured(
                prompt_factory.predict(images),
                schema=ImageToSideResponse,
            )
            instances.append(
                ImageToSideExperimentInstance(
                    problem_id=problem.id,
                    images=images,
                    response=response,
                    expected_answer=expected_answer,
                )
            )
        except Exception as e:
            print(
                f"problem_id: {problem.id} - Failed to parse json from model response:"
            )
            print(f"Exception: {e}")

    return messenger, instances
