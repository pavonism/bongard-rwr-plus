import glob
from typing import List
import pandas as pd
import torch
import clip
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List
from common import Record
from tqdm import tqdm

import random
import shutil

DATASET_DIR = "data/bongard-rwr-augmented-fixed"
OUTPUT_PATH = "data/bongard-rwr-plus-stochastic"

MIN_VARIANTS_PER_SIDE = 10
MAX_SIDE_COMBINATIONS = 100
N_REMOVED_IN_EACH_ITERATION = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def load_and_preprocess_images(image_folder: str):
    images = []
    image_paths = []

    for filename in glob.glob(f"{image_folder}/**/*.png", recursive=True):
        image_path = os.path.join(image_folder, filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        images.append(image)
        image_paths.append(image_path)

    return images, image_paths


def load_images(paths: List[str]):
    images = []

    for image_path in paths:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        images.append(image)

    return images


def extract_features(images):
    with torch.no_grad():
        image_features = torch.cat([model.encode_image(img) for img in images])
    return image_features.cpu().numpy()


def compute_similarity_matrix(features):
    return cosine_similarity(features)


def get_records_for_problem_side(problem_id: int, side: str) -> List[Record]:
    return [
        record
        for record in records
        if record.problem_id == str(problem_id) and record.side == side
    ]


def objective_function(x, matrix: np.ndarray):
    return np.sum(matrix[x, :][:, x])  # Sum of similarities for the selected images


def neighbor(solution, matrix: np.ndarray):
    new_solution = solution.copy()
    index = np.random.randint(0, len(solution))
    new_value = np.random.randint(0, matrix.shape[0])
    while new_value in new_solution:
        new_value = np.random.randint(0, matrix.shape[0])
    new_solution[index] = new_value
    return new_solution


def simulated_annealing(
    matrix: np.ndarray, initial_temp=100, cooling_rate=0.99, stopping_temp=1
):
    if matrix.shape[0] == 6:
        return np.arange(6), objective_function(np.arange(6), matrix)

    current_solution = np.random.choice(matrix.shape[0], size=6, replace=False)
    current_value = objective_function(current_solution, matrix)
    best_solution, best_value = current_solution, current_value
    temperature = initial_temp

    while temperature > stopping_temp:
        new_solution = neighbor(current_solution, matrix)
        new_value = objective_function(new_solution, matrix)

        if new_value < current_value or np.random.rand() < np.exp(
            (current_value - new_value) / temperature
        ):
            current_solution, current_value = new_solution, new_value

        if current_value < best_value:
            best_solution, best_value = current_solution, current_value

        temperature *= cooling_rate

    return best_solution, best_value


def save_matrices(
    problem_id: int,
    left_sides: List[List[Record]],
    right_sides: List[List[Record]],
):
    id_counter = 0

    for left_side in left_sides:
        for right_side in right_sides:
            for side_name, records in [("left", left_side), ("right", right_side)]:
                records_without_additional = records[:6]
                random.shuffle(records_without_additional)
                all_records = records_without_additional + records[6:]

                for index, img in enumerate(all_records):
                    img_path = f"{DATASET_DIR}/images/{img.filename}.jpg"
                    img_out_path = (
                        f"{OUTPUT_PATH}/{id_counter * 1000 + problem_id}/{side_name}"
                    )

                    os.makedirs(img_out_path, exist_ok=True)
                    shutil.copy(img_path, f"{img_out_path}/{index}.jpg")

            id_counter += 1

            if id_counter == MAX_SIDE_COMBINATIONS:
                return


def select_top_similar_records_from(
    all_records: List[Record],
    candidate_records: List[Record],
    similarity_matrix: np.ndarray,
    k: int,
):
    similarities = [
        np.sum(similarity_matrix[all_records.index(record), :])
        for record in candidate_records
    ]

    selected_records = []

    for _ in range(k):
        max_index = np.argmax(similarities)
        selected_records.append(candidate_records[max_index])
        similarities[max_index] = -1

    return selected_records


with open(f"{DATASET_DIR}/dataset.json", "r", encoding="utf-8") as dataset_file:
    dataset_dict = json.load(dataset_file)
records = [Record.from_dict(record) for record in dataset_dict]

problem_ids = set([int(record.problem_id) for record in records])


results = []
used_problems = 0
all_generated_matrices = 0

for problem_id in tqdm(problem_ids):
    left_sides = []
    right_sides = []

    for side, collection in [("left", left_sides), ("right", right_sides)]:
        problem_records = get_records_for_problem_side(problem_id, side)

        img_paths = [
            f"{DATASET_DIR}/images/{record.filename}.jpg" for record in problem_records
        ]
        imgs = load_images(img_paths)
        features = extract_features(imgs)
        similarity_matrix = compute_similarity_matrix(features)

        while (
            len(similarity_matrix) > 7
            and np.mean(similarity_matrix) < 0.85
            and len(collection) < MIN_VARIANTS_PER_SIDE
        ):
            special_index = np.argmin(similarity_matrix.sum(axis=1))
            special_record = problem_records[special_index]

            similar_indexes = [
                i
                for i in range(len(similarity_matrix[special_index]))
                if similarity_matrix[special_index][i] > 0.65 and i != special_index
            ]
            similar_records = [problem_records[i] for i in similar_indexes] + [
                record
                for record in problem_records
                if record.group_id == special_record.group_id
                and record != special_record
            ]
            similar_indexes = [
                problem_records.index(record) for record in similar_records
            ]

            temporally_problem_records = problem_records.copy()

            for record in set(similar_records):
                temporally_problem_records.remove(record)
            temporally_problem_records.remove(special_record)

            temporally_similarity_matrix = np.delete(
                similarity_matrix, [*similar_indexes, special_index], axis=0
            )
            temporally_similarity_matrix = np.delete(
                temporally_similarity_matrix, [*similar_indexes, special_index], axis=1
            )

            if (
                len(temporally_similarity_matrix) < 6
                or np.mean(temporally_similarity_matrix) > 0.85
            ):
                break

            solution, _ = simulated_annealing(temporally_similarity_matrix)

            chosen_records = [temporally_problem_records[i] for i in solution]
            collection.append(chosen_records + [special_record])

            records_to_remove = select_top_similar_records_from(
                problem_records,
                chosen_records,
                similarity_matrix,
                N_REMOVED_IN_EACH_ITERATION,
            ) + [special_record]

            record_indexes_to_remove = [
                problem_records.index(record) for record in records_to_remove
            ]

            similarity_matrix = np.delete(
                similarity_matrix, record_indexes_to_remove, axis=0
            )
            similarity_matrix = np.delete(
                similarity_matrix, record_indexes_to_remove, axis=1
            )

            for record in records_to_remove:
                problem_records.remove(record)

    if (
        len(left_sides) < MIN_VARIANTS_PER_SIDE
        or len(right_sides) < MIN_VARIANTS_PER_SIDE
    ):
        print(
            f"Problem {problem_id} has less than {MIN_VARIANTS_PER_SIDE} sides. Skipping."
        )
        continue

    results.append(
        {
            "problem_id": problem_id,
            "left_sides": len(left_sides),
            "right_sides": len(right_sides),
        }
    )

    save_matrices(
        problem_id,
        left_sides,
        right_sides,
    )
    used_problems += 1
    all_generated_matrices += len(left_sides) * len(right_sides)


print("Finished splitting into matrices.")
print(f"Used problems: {used_problems}")
print(f"Generated matrices: {all_generated_matrices}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_PATH}/results.csv", index=False)
