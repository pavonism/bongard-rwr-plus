import glob
import os

from PIL import Image
from tqdm import tqdm

INPUT_PATH = "data/rwr-plus"
OUTPUT_PATH = "data/rwr-plus-gs"

image_paths = sorted(
    glob.glob(f"{INPUT_PATH}/**/left/*.*", recursive=True)
    + glob.glob(f"{INPUT_PATH}/**/right/*.*", recursive=True)
)


for image_path in tqdm(image_paths):
    relative_path = os.path.relpath(image_path, INPUT_PATH)
    output_file = os.path.join(OUTPUT_PATH, relative_path)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    img = Image.open(image_path).convert("L")
    img.save(output_file, format="JPEG")
