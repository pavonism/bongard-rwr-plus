import asyncio
import os
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from src.dataset.model import (
    BongardDatasetInfo,
    BongardProblem,
)
from src.messenger.vllm_messenger import VllmMessenger
from src.strategy.d2s_bin.model import (
    BongardDatasetDescriptions,
    BongardDescriptionsDictionary,
    DescriptionsToSidesExperiment,
    DescriptionsToSidesExperimentInstance,
    DescriptionsToSidesResponse,
    Answer,
)
from src.strategy.d2s_bin.prompt import PromptFactory
import traceback


async def resolve_descriptions_to_sides_binary(
    problem_ids: List[int],
    dataset_path: str,
    experiment_file: str,
    descriptions_file: str,
    prompt_version: str,
    messengers: List[VllmMessenger],
    reevaluate: bool = False,
):
    if not os.path.exists(descriptions_file):
        raise FileNotFoundError(
            f"Descriptions file does not exist: {descriptions_file}."
        )

    dataset = BongardDatasetInfo.from_directory(dataset_path).subset(problem_ids)
    descriptions: BongardDatasetDescriptions = BongardDatasetDescriptions.from_file(
        descriptions_file
    )

    if os.path.exists(experiment_file) and not reevaluate:
        experiment = DescriptionsToSidesExperiment.from_file(experiment_file)
    else:
        experiment = DescriptionsToSidesExperiment(
            solver_name=messengers[0].get_name(),
            descriptor_name=descriptions.descriptor_name,
        )

    prompt_factory = PromptFactory(prompt_version)
    progress_bar = tqdm(total=len(dataset.problems))
    descriptions_dict = descriptions.to_dict()

    current_tasks = [
        asyncio.create_task(
            resolve_problem(
                problem=dataset.problems[i],
                experiment=experiment,
                messenger=messenger,
                prompt_factory=prompt_factory,
                descriptions_dict=descriptions_dict,
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
                        descriptions_dict=descriptions_dict,
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
    experiment: DescriptionsToSidesExperiment,
    messenger: VllmMessenger,
    prompt_factory: PromptFactory,
    descriptions_dict: BongardDescriptionsDictionary,
    reevaluate: bool,
) -> Tuple[VllmMessenger, Optional[DescriptionsToSidesExperimentInstance]]:
    if not descriptions_dict.has_all_problem_descriptions(problem):
        print(
            f"Problem with id: {problem.id} does not have all descriptions. Skipping prediction."
        )
        return messenger, None

    if experiment.has_problem_solution(problem) and not reevaluate:
        print(f"Problem {problem.id} already has solution. Skipping.")
        return messenger, None

    problem_descriptions = descriptions_dict.get_descriptions_for_problem(problem)
    expected_answer = Answer.random()
    request = problem_descriptions.create_request(problem, expected_answer)

    try:
        response: DescriptionsToSidesResponse = await messenger.ask_structured(
            prompt_factory.predict(request),
            schema=DescriptionsToSidesResponse,
        )

        return messenger, DescriptionsToSidesExperimentInstance(
            problem_id=problem.id,
            descriptions=problem_descriptions,
            response=response,
            expected_answer=expected_answer,
        )

    except Exception as e:
        print(f"Failed to solve problem with id: {problem.id}.")
        print(f"Error: {e}")
        print(traceback.format_exc())

    return messenger, None
