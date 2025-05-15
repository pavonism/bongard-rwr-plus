from collections import Counter
from enum import Enum
from typing import Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import classification_report

from src.dataset.model import BaseExperiment, BongardImage, BongardProblem
from src.messenger.content import Content, TextContent


class ImageDescription(BongardImage):
    description: str

    def content(self) -> Content:
        return TextContent(self.description)


class Side(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class DescriptionToSideResponse(BaseModel):
    concept: str
    explanation: str
    answer: Side


class DescriptionToSideRequest(BaseModel):
    left_descriptions: List[str]
    right_descriptions: List[str]
    test_description: str


class BongardProblemDescriptions(BaseModel):
    left_descriptions: List[ImageDescription]
    right_descriptions: List[ImageDescription]

    def create_request(self, side: Side) -> DescriptionToSideRequest:
        return DescriptionToSideRequest(
            left_descriptions=[x.description for x in self.left_descriptions[:-1]],
            right_descriptions=[x.description for x in self.right_descriptions[:-1]],
            test_description=self._get_last_description(side).description,
        )

    def _get_last_description(self, side: Side) -> Optional[ImageDescription]:
        if side == Side.LEFT:
            return self.left_descriptions[-1]
        elif side == Side.RIGHT:
            return self.right_descriptions[-1]


class DescriptionToSideExperimentInstance(BaseModel):
    problem_id: int
    descriptions: BongardProblemDescriptions
    response: Optional[DescriptionToSideResponse]
    expected_answer: Optional[Side]

    def evaluate(self) -> bool:
        return self.response.answer == self.expected_answer


class BongardDescriptionsDictionary(BaseModel):
    descriptor_name: str
    descriptions: Dict[int, ImageDescription] = {}

    def has_image_description(self, image: BongardImage) -> bool:
        return (
            image.image_id in self.descriptions
            and self.descriptions[image.image_id].description != ""
        )

    def add_description(self, description: ImageDescription):
        if not self.has_image_description(description):
            self.descriptions[description.image_id] = description
        else:
            print(f"Duplicate description: {description.image_id}. Skipping.")

    def has_all_problem_descriptions(self, problem: BongardProblem) -> bool:
        for img in problem.left_images + problem.right_images:
            if img.image_id not in self.descriptions:
                return False
        return True

    def get_descriptions_for_problem(
        self, problem: BongardProblem
    ) -> BongardProblemDescriptions:
        left_descriptions = [
            self.descriptions[x.image_id]
            for x in problem.left_images
            if x.image_id in self.descriptions
        ]
        right_descriptions = [
            self.descriptions[x.image_id]
            for x in problem.right_images
            if x.image_id in self.descriptions
        ]

        return BongardProblemDescriptions(
            left_descriptions=left_descriptions,
            right_descriptions=right_descriptions,
        )

    def flatten(self) -> "BongardDatasetDescriptions":
        return BongardDatasetDescriptions(
            descriptor_name=self.descriptor_name,
            img_descriptions=[desc for desc in self.descriptions.values()],
        )


class BongardDatasetDescriptions(BaseExperiment):
    descriptor_name: str
    img_descriptions: List[ImageDescription] = []

    def to_dict(self) -> BongardDescriptionsDictionary:
        descs = {desc.image_id: desc for desc in self.img_descriptions}
        return BongardDescriptionsDictionary(
            descriptor_name=self.descriptor_name, descriptions=descs
        )


class DescriptionToSideExperiment(BaseExperiment):
    descriptor_name: str
    solver_name: str
    instances: List[DescriptionToSideExperimentInstance] = []

    def has_problem_solution(self, problem: BongardProblem, side: Side) -> bool:
        for instance in self.instances:
            if instance.problem_id == problem.id and instance.response:
                return True
        return False

    def add_solution(self, instance: DescriptionToSideExperimentInstance):
        self.instances.append(instance)

    @staticmethod
    def from_file(path: str) -> "DescriptionToSideExperiment":
        with open(path, "r", encoding="utf-8") as f:
            json_str = f.read()
            return DescriptionToSideExperiment.model_validate_json(json_str)

    def to_file(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as f:
            json_str = self.model_dump_json(indent=4)
            return f.write(json_str)

    def summarize(self) -> str:
        y_true, y_pred = [], []
        for instance in self.instances:
            y_true.append(instance.expected_answer.value)
            y_pred.append(instance.response.answer.value)
        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        df_report = pd.DataFrame(report_dict).transpose()

        pred_support = Counter(y_pred)
        df_report["predicted_support"] = df_report.index.map(
            lambda label: (
                pred_support[Side[label]] if label in {"LEFT", "RIGHT"} else 0
            )
        )

        summary = f"{self.metrics()}\n"
        summary += df_report.to_string(float_format="%.2f")

        return summary.strip()

    def metrics(self) -> Dict[str, Union[int, float]]:
        num_correct = 0
        for instance in self.instances:
            num_correct += int(instance.evaluate())
        n = len(self.instances)
        return {
            "num_correct_answers": num_correct,
            "num_all_answers": n,
            "acc": round(num_correct / n * 100, 2),
        }

    def merge(self, other) -> bool:
        if not isinstance(other, DescriptionToSideExperiment):
            print(
                f"WARN: Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )
            return False

        self.instances.extend(other.instances)
        return True
