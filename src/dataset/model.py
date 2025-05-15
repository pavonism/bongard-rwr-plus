from typing import List

from pydantic import BaseModel
from src.messenger.content import Content, ImageContent


class BongardImage(BaseModel):
    image_id: int
    path: str

    def content(self) -> Content:
        return ImageContent(self.path)


class BongardProblem(BaseModel):
    id: int
    left_images: List[BongardImage]
    right_images: List[BongardImage]
    left_side_image: str = ""
    right_side_image: str = ""
    whole_image: str = ""


class BongardDatasetInfo(BaseModel):
    problems: List[BongardProblem]

    def subset(self, problem_ids: List[int]) -> "BongardDatasetInfo":
        problems = [problem for problem in self.problems if problem.id in problem_ids]
        return BongardDatasetInfo(problems=problems)

    def collect_images(self) -> List[BongardImage]:
        image_ids = set()
        images = []
        for problem in self.problems:
            for img in problem.left_images + problem.right_images:
                if img.image_id not in image_ids:
                    images.append(img)
                    image_ids.add(img.image_id)
        return images

    @staticmethod
    def from_directory(
        path: str, adjust_path_prefix: bool = True
    ) -> "BongardDatasetInfo":
        with open(f"{path}/dataset.json", "r", encoding="utf-8") as f:
            json_str = f.read()
            dataset_info = BongardDatasetInfo.model_validate_json(json_str)

            if adjust_path_prefix:
                for problem in dataset_info.problems:
                    for image in problem.left_images + problem.right_images:
                        image.path = f"{path}/{image.path}"
                    problem.left_side_image = f"{path}/{problem.left_side_image}"
                    problem.right_side_image = f"{path}/{problem.right_side_image}"
                    problem.whole_image = f"{path}/{problem.whole_image}"

            return dataset_info

    def to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json_str = self.model_dump_json(indent=4)
            f.write(json_str)


class BaseExperiment(BaseModel):
    @classmethod
    def from_file(cls, path: str) -> "BaseExperiment":
        with open(path, "r", encoding="utf-8") as f:
            json_str = f.read()
            return cls.model_validate_json(json_str)

    def to_file(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as f:
            json_str = self.model_dump_json(indent=4)
            return f.write(json_str)

    def merge(self, other: "BaseExperiment") -> bool:
        pass


class ExperimentResult(BaseModel):
    experiment: BaseExperiment
    experiment_file: str
