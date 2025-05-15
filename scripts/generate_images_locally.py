import os
from pathlib import Path
import time

from diffusers import FluxPipeline
import pandas as pd
import torch

from common import BongardDataset

model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

dataset = BongardDataset("data/bongard-rwr", labels_file="data/labels.csv")
prompts_df = pd.read_csv("baseline/bonagrd_rwr_prompts_augmented_plus_15.csv")
total_rows = len(prompts_df)
current_row = 0

for problem_id, file_name, side, file_path in dataset.all_fragments():
    img_rows = prompts_df.query(
        f"problem_id == {problem_id} and file == {file_name} and side == '{side}'"
    )
    positive_prompts = img_rows["positive"]
    negative_prompt = img_rows["negative"].iloc[0]
    file_path = file_path.replace("data/", "")
    file_number = Path(file_name).stem
    output_directory = f"./data/bongard-rwr-augmented/{problem_id}/{side}/{file_number}"
    os.makedirs(output_directory, exist_ok=True)

    for index, positive_prompt in enumerate(positive_prompts):
        current_row += 1

        print(f"|{current_row} / {total_rows}|")
        print("Prompt: ", positive_prompt)
        print("Negative prompt: ", negative_prompt)

        image = pipe(
            prompt=f"Sharp, high quality image of {positive_prompt}",
            negative_prompt=negative_prompt,
        ).images[0]

        timestamp = int(time.time())
        image.save(
            f"./data/bongard-rwr-augmented/{problem_id}/{side}/{file_number}/{timestamp}_{index}.png"
        )
