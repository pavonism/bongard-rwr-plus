from concurrent.futures import Future, ProcessPoolExecutor
import glob
import itertools
import multiprocessing
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Dict, List
import torch
import clip
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from tqdm import tqdm
import asyncio

import random
import shutil
from sortedcontainers import SortedList

from src.dataset.model import BongardDatasetInfo, BongardImage, BongardProblem

SUBPROBLEM_SIZE = 6
MIN_VARIANTS_PER_SIDE = 20
N_JOBS = 32

DATASET_DIR = "app/data/rwr-plus"
OUTPUT_PATH = "app/data/rwr-plus-6i"


class FeatureExtractor:
    def __init__(self, index: int = 0):
        self.index = index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

    def load_and_preprocess_images(self, image_folder: str):
        images = []
        image_paths = []

        for filename in glob.glob(f"{image_folder}/**/*.png", recursive=True):
            image_path = os.path.join(image_folder, filename)
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            images.append(image)
            image_paths.append(image_path)

        return images, image_paths

    def load_images(self, paths: List[str]):
        images = []

        for image_path in paths:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            images.append(image)

        return images

    def extract_features(self, images):
        with torch.no_grad():
            image_features = torch.cat([self.model.encode_image(img) for img in images])
        return image_features.cpu().numpy()

    def compute_similarity_matrix(self, bongard_images: List[BongardImage]):
        imgs = self.load_images([img.path for img in bongard_images])
        features = self.extract_features(imgs)
        return cosine_similarity(features)

    def pick_n_least_similar_images(
        self, n: int, images: List[BongardImage]
    ) -> List[BongardImage]:
        similarity_matrix = self.compute_similarity_matrix(images)
        max_similarity = similarity_matrix.max(axis=1)
        min_indexes = np.argsort(max_similarity)[:n]

        return [images[i] for i in min_indexes]


def save_matrices(
    problem_id: int,
    left_sides: List[List[BongardImage]],
    right_sides: List[List[BongardImage]],
) -> List[BongardProblem]:
    id_counter = 0
    problems = []

    for left_side in left_sides:
        for right_side in right_sides:
            left_side_out: List[BongardImage] = []
            right_side_out: List[BongardImage] = []

            for side_name, records, out in [
                ("left", left_side, left_side_out),
                ("right", right_side, right_side_out),
            ]:
                records_without_additional = records[:SUBPROBLEM_SIZE]
                random.shuffle(records_without_additional)
                all_records = records_without_additional + records[SUBPROBLEM_SIZE:]

                for bongard_img in all_records:
                    img_out_path = f"{OUTPUT_PATH}/images/{bongard_img.image_id}{Path(bongard_img.path).suffix}"

                    if not os.path.exists(img_out_path):
                        shutil.copy(
                            bongard_img.path,
                            img_out_path,
                        )

                    out.append(
                        BongardImage(
                            image_id=bongard_img.image_id,
                            path=f"images/{bongard_img.image_id}{Path(bongard_img.path).suffix}",
                        )
                    )

            problems.append(
                BongardProblem(
                    id=1000 * id_counter + problem_id,
                    left_images=left_side_out,
                    right_images=right_side_out,
                )
            )

            id_counter += 1

    return problems


def pick_special_image(
    side: List[BongardImage],
    all_images: List[BongardImage],
    similarity_matrix: np.ndarray,
):
    side_indexes = [all_images.index(img) for img in side]
    side_similarity_matrix = similarity_matrix[side_indexes]

    max_similarity = float("inf")
    image = None

    for index in range(side_similarity_matrix.shape[1]):
        curr_max_similarity = side_similarity_matrix[:, index].max()
        if curr_max_similarity < max_similarity and index not in side_indexes:
            image = all_images[index]
            max_similarity = curr_max_similarity

    side.append(image)


def squeeze_dataset(dataset: BongardDatasetInfo) -> BongardDatasetInfo:
    problem_id_to_problem: Dict[int, BongardProblem] = {}

    for problem in dataset.problems:
        original_problem_id = problem.id % 1000
        if original_problem_id not in problem_id_to_problem:
            problem.id = original_problem_id
            problem_id_to_problem[original_problem_id] = problem
        else:
            existing_problem = problem_id_to_problem[original_problem_id]
            existing_problem.left_images.extend(problem.left_images)
            existing_problem.right_images.extend(problem.right_images)

    for problem in problem_id_to_problem.values():
        problem.left_images = list(
            {img.image_id: img for img in problem.left_images}.values()
        )
        problem.right_images = list(
            {img.image_id: img for img in problem.right_images}.values()
        )

    return BongardDatasetInfo(problems=list(problem_id_to_problem.values()))


class SubsetWithMaxSimilarity:
    def __init__(self, subset: List[BongardImage], max_similarity: float):
        self.subset = subset
        self.max_similarity = max_similarity

    def __lt__(self, other):
        return self.max_similarity < other.max_similarity

    def __eq__(self, other):
        return self.max_similarity == other.max_similarity


def generate_subproblems(
    process_id: int,
    problem: BongardProblem,
    left_sm: np.ndarray,
    right_sm: np.ndarray,
    progress_dict: DictProxy,
) -> List[BongardProblem]:
    left_sides = []
    right_sides = []

    for side, collection, imgs, similarity_matrix in [
        ("left", left_sides, problem.left_images, left_sm),
        ("right", right_sides, problem.right_images, right_sm),
    ]:
        subsets = list(itertools.combinations(imgs, SUBPROBLEM_SIZE))

        best_subsets = SortedList()
        best_subsets_similarities = []
        max_similarity = float("inf")

        progress_dict[process_id]["side"] = side
        progress_dict[process_id]["problem_id"] = problem.id
        progress_dict[process_id]["total_subsets"] = len(subsets)
        progress_dict[process_id]["processed_subsets"] = 0

        for i, subset in enumerate(subsets):
            indexes = [imgs.index(img) for img in subset]
            subset_similarity_matrix = similarity_matrix[indexes][:, indexes]

            lower_vals = np.tril(subset_similarity_matrix, k=-1)

            curr_max_similarity = lower_vals.max()
            subset = SubsetWithMaxSimilarity(
                subset=list(subset), max_similarity=curr_max_similarity
            )

            if curr_max_similarity < max_similarity and len(best_subsets) > 0:
                highest_similarity_subset = best_subsets[-1]
                best_subsets.remove(highest_similarity_subset)
                best_subsets_similarities.remove(
                    highest_similarity_subset.max_similarity
                )

            if len(best_subsets) < MIN_VARIANTS_PER_SIDE:
                best_subsets.add(subset)
                best_subsets_similarities.append(subset.max_similarity)
                max_similarity = np.max(best_subsets_similarities)

            if i % 100_000 == 0:
                progress_dict[process_id]["processed_subsets"] += 100_000

        for subset in best_subsets:
            side = subset.subset
            pick_special_image(side, imgs, similarity_matrix)
            collection.append(side)

    if (
        len(left_sides) < MIN_VARIANTS_PER_SIDE
        or len(right_sides) < MIN_VARIANTS_PER_SIDE
    ):
        print(
            f"Problem {problem.id} has less than {MIN_VARIANTS_PER_SIDE} sides. Skipping."
        )
        return []

    current_problems = save_matrices(
        problem.id,
        left_sides,
        right_sides,
    )

    return current_problems


async def main():
    dataset = BongardDatasetInfo.from_directory(DATASET_DIR)
    dataset = squeeze_dataset(dataset)

    os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)

    fe = FeatureExtractor()
    generated_problems: List[BongardProblem] = []

    p_bars = [tqdm(total=0, position=i + 1) for i in range(N_JOBS)]
    main_p_bar = tqdm(total=len(dataset.problems), position=0, desc="Running threads")

    futures: List[Future] = []
    collected_futures = []

    with multiprocessing.Manager() as manager:
        progress_dict = manager.dict()

        with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
            for i, problem in enumerate(dataset.problems):
                left_similarties = fe.compute_similarity_matrix(problem.left_images)
                right_similarties = fe.compute_similarity_matrix(problem.right_images)

                progress_dict[i] = manager.dict()

                future = executor.submit(
                    generate_subproblems,
                    i,
                    problem,
                    left_similarties,
                    right_similarties,
                    progress_dict,
                )

                futures.append(future)

                update_p_bars(p_bars, progress_dict, futures)
                main_p_bar.update(1)

            main_p_bar.desc = "Collecting results"
            main_p_bar.total = len(futures)
            main_p_bar.n = 0
            main_p_bar.refresh()

            while len(futures) > len(collected_futures):
                await asyncio.sleep(1)
                update_p_bars(p_bars, progress_dict, futures)

                collect_generated_problems(
                    generated_problems,
                    main_p_bar,
                    futures,
                    collected_futures,
                )

    for p_bar in p_bars:
        p_bar.close()

    print("Finished splitting into matrices.")
    print(f"Generated matrices: {len(generated_problems)}")

    new_dataset = BongardDatasetInfo(
        problems=generated_problems,
    )
    new_dataset.to_file(f"{OUTPUT_PATH}/dataset.json")


def collect_generated_problems(
    generated_problems,
    main_p_bar,
    futures,
    collected_futures,
):
    to_collect = [
        future
        for future in futures
        if future.done() and future not in collected_futures
    ]

    if len(to_collect) > 0:
        for future in to_collect:
            collected_futures.append(future)
            generated_problems.extend(future.result())

        main_p_bar.update(len(to_collect))


def update_p_bars(p_bars, progress_dict, futures):
    running_futures = [i for i in range(len(futures)) if futures[i].running()]
    for i, id in enumerate(running_futures):
        current_progress = progress_dict.get(id, {})

        if current_progress:
            side = current_progress.get("side", "unknown")
            problem_id = current_progress.get("problem_id", -1)
            total_subsets = current_progress.get("total_subsets", 1)
            processed_subsets = current_progress.get("processed_subsets", 0)

            if processed_subsets > 0:
                p_bars[i].desc = f"Problem {problem_id} ({side})"
                p_bars[i].total = total_subsets
                p_bars[i].n = processed_subsets
                p_bars[i].last_print_n = processed_subsets
                p_bars[i].refresh()


if __name__ == "__main__":
    asyncio.run(main())
