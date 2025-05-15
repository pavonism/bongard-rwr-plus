import asyncio
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset.model import BaseExperiment, BongardDatasetInfo, BongardProblem
from src.evaluation.longest_common_subsequence import (
    get_longest_common_word_subsequence_length,
)
from src.messenger.vllm_messenger import VllmMessenger
from src.strategy.selection.model import (
    Concept,
    SelectionExperiment,
    SelectionExperimentInstance,
    SelectionInput,
    SelectionRequest,
    SelectionResponse,
)
from src.strategy.selection.prompt import PromptFactory


async def resolve_selection(
    problem_ids: List[int],
    dataset_path: str,
    labels_file: str,
    output_file: str,
    prompt_version: str,
    messengers: List[VllmMessenger],
    num_choices: int = 2,
    reevaluate: bool = False,
):
    df = pd.read_csv(labels_file)

    if os.path.exists(output_file) and not reevaluate:
        experiment = SelectionExperiment.from_file(output_file)
    else:
        experiment = SelectionExperiment(solver_name=messengers[0].get_name())

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
                labels_df=df,
                num_choices=num_choices,
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
                        labels_df=df,
                        num_choices=num_choices,
                        reevaluate=reevaluate,
                    )
                )
                for i, messenger in enumerate(available_messengers)
                if current_problem_index + i < len(dataset.problems)
            ]

            current_problem_index += len(available_messengers)

    os.makedirs(Path(output_file).parent, exist_ok=True)
    experiment.to_file(output_file)


async def resolve_problem(
    problem: BongardProblem,
    experiment: SelectionExperiment,
    messenger: VllmMessenger,
    prompt_factory: PromptFactory,
    labels_df: pd.DataFrame,
    num_choices: int,
    reevaluate: bool,
) -> Tuple[VllmMessenger, Optional[SelectionExperimentInstance]]:
    if experiment.has_solution(problem.whole_image) and not reevaluate:
        print(f"Problem {problem.id} already has solution. Skipping.")
        return messenger, None

    idxs = np.random.permutation(num_choices)
    positive_concept_idx = problem_id_to_index(problem.id)
    negative_concept_idxs = get_negative_concept_indices(
        positive_idx=positive_concept_idx,
        num_negatives=num_choices - 1,
        df=labels_df,
    )
    expected_label = None
    concepts = []
    for i, idx in enumerate(idxs):
        label = 1 + i
        if idx == num_choices - 1:
            df_idx = positive_concept_idx
            expected_label = label
        else:
            df_idx = negative_concept_idxs[idx]
        concept = Concept(
            left=labels_df.iloc[df_idx]["Left-side Rule"],
            right=labels_df.iloc[df_idx]["Right-side Rule"],
            label=label,
        )
        concepts.append(concept)
    assert expected_label is not None

    input = SelectionInput(image_path=problem.whole_image)
    request = SelectionRequest(concepts=concepts)

    try:
        response: SelectionResponse = await messenger.ask_structured(
            prompt_factory.select(input, request),
            schema=SelectionResponse,
        )

        return messenger, SelectionExperimentInstance(
            problem_id=problem.id,
            input=input,
            request=request,
            response=response,
            expected_label=expected_label,
        )

    except Exception as e:
        print(f"problem_id: {problem.id} - Failed to parse json from model response:")
        print(f"Exception: {e}")

    return messenger, None


def problem_id_to_index(problem_id: int) -> int:
    return (problem_id % 1000) - 1


def get_negative_concept_indices(
    positive_idx: int, num_negatives: int, df: pd.DataFrame
) -> List[int]:
    positive_rule = " ".join(
        [
            df.iloc[positive_idx]["Left-side Rule"],
            df.iloc[positive_idx]["Right-side Rule"],
        ]
    )
    positive_rule = positive_rule.replace("NOT ", "")
    all_rules = [positive_rule]
    negative_concept_idxs = []
    negative_idx = positive_idx + 1
    while len(negative_concept_idxs) < num_negatives:
        if negative_idx == positive_idx:
            raise ValueError(
                f"Failed to find negative concepts. anchor_idx: {positive_idx}, num_negatives: {num_negatives}"
            )
        if negative_idx >= len(df):
            negative_idx = 0
        negative_rule = " ".join(
            [
                df.iloc[negative_idx]["Left-side Rule"],
                df.iloc[negative_idx]["Right-side Rule"],
            ]
        )
        negative_rule = negative_rule.replace("NOT ", "")
        can_include = True
        for rule in all_rules:
            lcs_length = get_longest_common_word_subsequence_length(rule, negative_rule)
            if lcs_length > 0:
                can_include = False
                break
        if can_include:
            negative_concept_idxs.append(negative_idx)
            all_rules.append(negative_rule)
        negative_idx += 1
    return negative_concept_idxs
