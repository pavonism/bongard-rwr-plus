from typing import List

from pydantic import BaseModel


class BongardProblem(BaseModel):
    id: int
    left_images: List[str]
    right_images: List[str]
    left_side_image: str = ""
    right_side_image: str = ""
    whole_image: str = ""


class BongardDatasetInfo(BaseModel):
    problems: List[BongardProblem]

    def to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json_str = self.model_dump_json(indent=4)
            f.write(json_str)

    @staticmethod
    def from_directory(
        path: str, adjust_path_prefix: bool = True
    ) -> "BongardDatasetInfo":
        with open(f"{path}/dataset.json", "r", encoding="utf-8") as f:
            json_str = f.read()
            dataset_info = BongardDatasetInfo.model_validate_json(json_str)

            if adjust_path_prefix:
                for problem in dataset_info.problems:
                    for i in range(len(problem.left_images)):
                        problem.left_images[i] = f"{path}/{problem.left_images[i]}"
                    for i in range(len(problem.right_images)):
                        problem.right_images[i] = f"{path}/{problem.right_images[i]}"

                    problem.left_side_image = f"{path}/{problem.left_side_image}"
                    problem.right_side_image = f"{path}/{problem.right_side_image}"
                    problem.whole_image = f"{path}/{problem.whole_image}"

            return dataset_info
