import os

from tqdm import tqdm

from src.dataset.model import BongardDatasetInfo
from src.model.similarity_classifier import encode_images, classify_to_side
from src.strategy.i1s.model import (
    Answer,
    ImageToSideExperiment,
    ImageToSideExperimentInstance,
    BongardProblemImages,
    ImageToSideResponse,
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

    for problem in tqdm(dataset.problems, desc=dataset_name):
        left = encode_images(problem.left_images)
        right = encode_images(problem.right_images)
        left_test, right_test = left[-1:], right[-1:]
        left, right = left[:-1], right[:-1]

        for test_image, test_image_path, expected_answer in [
            (left_test, problem.left_images[-1].path, Answer.LEFT),
            (right_test, problem.right_images[-1].path, Answer.RIGHT),
        ]:
            pred_answer = classify_to_side(left, right, test_image, "euclidean", "max")

            instances.append(
                ImageToSideExperimentInstance(
                    problem_id=problem.id,
                    images=BongardProblemImages(
                        whole_image_path=problem.whole_image,
                        test_image_path=test_image_path,
                    ),
                    expected_answer=expected_answer,
                    response=ImageToSideResponse(
                        concept="",
                        explanation="",
                        answer=pred_answer.value,
                    ),
                )
            )

    experiment = ImageToSideExperiment(
        solver_name="similarity-classifier",
        instances=instances,
    )
    experiment_dir = f"P:/eden/processed/bongard/MM-experiments/{dataset_name}_i1s_v3"
    os.makedirs(parents=True, exist_ok=True)
    experiment.to_file(f"{experiment_dir}/similarity-classifier_experiment.json")
