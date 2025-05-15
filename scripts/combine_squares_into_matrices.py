import os
import PIL.Image
from tqdm import tqdm

from src.dataset.model import BongardDatasetInfo
from src.dataset.rendering.squares import draw_square_bongard_problem

DATASET_PATH = r"P:\eden\data\rwr-plus-6i"

dataset = BongardDatasetInfo.from_directory(DATASET_PATH, adjust_path_prefix=False)

os.makedirs(f"{DATASET_PATH}/lefts", exist_ok=True)
os.makedirs(f"{DATASET_PATH}/rights", exist_ok=True)
os.makedirs(f"{DATASET_PATH}/wholes", exist_ok=True)

key_to_left_side = {}
key_to_right_side = {}

for problem in tqdm(dataset.problems):
    left_canvas, right_canvas, canvas = draw_square_bongard_problem(
        [
            PIL.Image.open(f"{DATASET_PATH}/{bimg.path}")
            for bimg in problem.left_images[:-1]
        ],
        [
            PIL.Image.open(f"{DATASET_PATH}/{bimg.path}")
            for bimg in problem.right_images[:-1]
        ],
        margin=10,
        side_size=(512, 1024),
        background_color="black",
        separator_color="white",
        grid=(1, 3),
    )

    left_key = "_".join(
        [
            str(bimg.image_id)
            for bimg in sorted(problem.left_images[:-1], key=lambda x: x.image_id)
        ]
    )
    right_key = "_".join(
        [
            str(bimg.image_id)
            for bimg in sorted(problem.right_images[:-1], key=lambda x: x.image_id)
        ]
    )

    if left_key not in key_to_left_side:
        img_index = len(key_to_left_side)
        left_canvas.save(f"{DATASET_PATH}/lefts/{img_index}.jpeg")
        key_to_left_side[left_key] = img_index

    if right_key not in key_to_right_side:
        img_index = len(key_to_right_side)
        right_canvas.save(f"{DATASET_PATH}/rights/{img_index}.jpeg")
        key_to_right_side[right_key] = img_index

    canvas.save(f"{DATASET_PATH}/wholes/{problem.id}.jpeg")

    problem.left_side_image = f"lefts/{key_to_left_side[left_key]}.jpeg"
    problem.right_side_image = f"rights/{key_to_right_side[right_key]}.jpeg"
    problem.whole_image = f"wholes/{problem.id}.jpeg"


dataset.to_file(rf"{DATASET_PATH}\dataset.json")
