from collections import Counter
import os
from typing import List, Optional, Tuple
import os

from tqdm import tqdm

from src.dataset.model import BongardDatasetInfo
from src.model.similarity_classifier import encode_images, classify_to_sides
from src.strategy.i2s.model import (
    ExpectedAnswer,
    ImagesToSidesExperiment,
    ImagesToSidesExperimentInstance,
    BongardProblemImages,
    ImagesToSidesResponse,
    ImageToSideResponse,
    Answer,
)

DATASETS = [
    "rwr-plus",
    "rwr-plus-2i",
    "rwr-plus-3i",
    "rwr-plus-4i",
    "rwr-plus-5i",
    "rwr-plus-6i",
]

for dataset_name in DATASETS:
    dataset = BongardDatasetInfo.from_directory(f"P:/eden/data/{dataset_name}")
    instances = []

    for problem in tqdm(dataset.problems):
        left = encode_images(problem.left_images)
        right = encode_images(problem.right_images)
        pred_answers = classify_to_sides(left, right, "euclidean", "max")
        instances.append(
            ImagesToSidesExperimentInstance(
                problem_id=problem.id,
                images=BongardProblemImages(
                    whole_image_path=problem.whole_image,
                    first_test_image_path=problem.left_images[-1].path,
                    second_test_image_path=problem.right_images[-1].path,
                ),
                expected_answer=ExpectedAnswer(first=Answer.LEFT, second=Answer.RIGHT),
                response=ImagesToSidesResponse(
                    concept="",
                    first=ImageToSideResponse(explanation="", answer=pred_answers[0]),
                    second=ImageToSideResponse(explanation="", answer=pred_answers[1]),
                ),
            )
        )

    experiment = ImagesToSidesExperiment(
        solver_name="similarity-classifier",
        instances=instances,
    )

    experiment_dir = f"P:/eden/processed/bongard/MM-experiments/{dataset_name}_i2s_v3"
    os.makedirs(parents=True, exist_ok=True)
    experiment.to_file(f"{experiment_dir}/similarity-classifier_experiment.json")
