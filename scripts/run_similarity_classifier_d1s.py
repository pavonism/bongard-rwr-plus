import os

from tqdm import tqdm

from src.dataset.model import BongardDatasetInfo
from src.model.similarity_classifier import encode_texts, classify_to_side, Answer
from src.strategy.d1s.model import (
    BongardDatasetDescriptions,
    DescriptionToSideExperiment,
    DescriptionToSideExperimentInstance,
    DescriptionToSideResponse,
    Side,
)

DATASETS = [
    "rwr-plus",
    "rwr-plus-2i",
    "rwr-plus-3i",
    "rwr-plus-4i",
    "rwr-plus-5i",
    "rwr-plus-6i",
]
# data_dir = "/app/data"
data_dir = "local-data"
method = "d1s_v2"
descriptor = "OpenGVLab/InternVL2_5-78B"
dataset_dir = "/home/mikmal/Datasets"

for dataset_name in DATASETS:
    dataset = BongardDatasetInfo.from_directory(f"{dataset_dir}/{dataset_name}")
    descriptions: BongardDatasetDescriptions = BongardDatasetDescriptions.from_file(
        f"{data_dir}/{dataset_name}_i1d_v1/{descriptor}_descriptions.json"
    )
    descriptions_dict = descriptions.to_dict()
    instances = []

    for problem in tqdm(dataset.problems):
        problem_descriptions = descriptions_dict.get_descriptions_for_problem(problem)
        for side in [Side.LEFT, Side.RIGHT]:
            request = problem_descriptions.create_request(side)

            left = encode_texts(request.left_descriptions)
            right = encode_texts(request.right_descriptions)
            test = encode_texts([request.test_description])

            pred_answer = classify_to_side(left, right, test, "euclidean", "max")
            instances.append(
                DescriptionToSideExperimentInstance(
                    problem_id=problem.id,
                    descriptions=problem_descriptions,
                    response=DescriptionToSideResponse(
                        concept="",
                        explanation="",
                        answer=Side.LEFT if pred_answer == Answer.LEFT else Side.RIGHT,
                    ),
                    expected_answer=side,
                )
            )

    experiment = DescriptionToSideExperiment(
        descriptor_name=descriptor,
        solver_name="similarity-classifier",
        instances=instances,
    )
    os.makedirs(f"{data_dir}/{dataset_name}_{method}/{descriptor}", exist_ok=True)
    experiment.to_file(f"{data_dir}/{dataset_name}_{method}/{descriptor}/similarity-classifier_experiment.json")
