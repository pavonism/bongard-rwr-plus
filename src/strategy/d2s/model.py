from collections import Counter
from enum import Enum
import random
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


class Answer(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class ExpectedAnswer(BaseModel):
    first: Answer
    second: Answer

    @staticmethod
    def random():
        answers = [Answer.LEFT, Answer.RIGHT]
        random.shuffle(answers)

        return ExpectedAnswer(
            first=answers[0],
            second=answers[1],
        )


class DescriptionToSideResponse(BaseModel):
    explanation: str
    answer: Answer


class DescriptionsToSidesRequest(BaseModel):
    left_descriptions: List[str]
    right_descriptions: List[str]
    first: str
    second: str
    whole_image: str


class DescriptionsToSidesResponse(BaseModel):
    concept: str = ""
    first: DescriptionToSideResponse
    second: DescriptionToSideResponse


class BongardProblemDescriptions(BaseModel):
    left_descriptions: List[ImageDescription]
    right_descriptions: List[ImageDescription]

    def create_request(
        self,
        problem: BongardProblem,
        expected_answer: ExpectedAnswer,
    ) -> DescriptionsToSidesRequest:
        return DescriptionsToSidesRequest(
            left_descriptions=[x.description for x in self.left_descriptions[:-1]],
            right_descriptions=[x.description for x in self.right_descriptions[:-1]],
            first=self._get_last_description(expected_answer.first).description,
            second=self._get_last_description(expected_answer.second).description,
            whole_image=problem.whole_image,
        )

    def _get_last_description(self, side: Answer) -> Optional[ImageDescription]:
        if side == Answer.LEFT:
            return self.left_descriptions[-1]
        elif side == Answer.RIGHT:
            return self.right_descriptions[-1]


class DescriptionsToSidesExperimentInstance(BaseModel):
    problem_id: int
    descriptions: BongardProblemDescriptions
    response: Optional[DescriptionsToSidesResponse]
    expected_answer: Optional[ExpectedAnswer]

    def evaluate(self) -> bool:
        return (
            self.response.first.answer == self.expected_answer.first
            and self.response.second.answer == self.expected_answer.second
        )


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


class DescriptionsToSidesExperiment(BaseExperiment):
    descriptor_name: str
    solver_name: str
    instances: List[DescriptionsToSidesExperimentInstance] = []

    def has_problem_solution(self, problem: BongardProblem) -> bool:
        for instance in self.instances:
            if instance.problem_id == problem.id and instance.response:
                return True
        return False

    def add_solution(self, instance: DescriptionsToSidesExperimentInstance):
        self.instances.append(instance)

    @staticmethod
    def from_file(path: str) -> "DescriptionsToSidesExperiment":
        with open(path, "r", encoding="utf-8") as f:
            json_str = f.read()
            return DescriptionsToSidesExperiment.model_validate_json(json_str)

    def to_file(self, path: str) -> int:
        with open(path, "w", encoding="utf-8") as f:
            json_str = self.model_dump_json(indent=4)
            return f.write(json_str)

    def summarize(self) -> str:
        num_correct = 0
        answers_to_label = {
            (Answer.LEFT, Answer.LEFT): 0,
            (Answer.LEFT, Answer.RIGHT): 1,
            (Answer.RIGHT, Answer.LEFT): 2,
            (Answer.RIGHT, Answer.RIGHT): 3,
        }

        y_true, y_pred = [], []
        for instance in self.instances:
            num_correct += int(instance.evaluate())
            y_true.append(
                answers_to_label[
                    (instance.expected_answer.first, instance.expected_answer.second)
                ]
            )
            y_pred.append(
                answers_to_label[
                    (instance.response.first.answer, instance.response.second.answer)
                ]
            )

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(4)),
            target_names=["LEFT-LEFT", "LEFT-RIGHT", "RIGHT-LEFT", "RIGHT-RIGHT"],
            output_dict=True,
            zero_division=0,
        )

        df_report = pd.DataFrame(report_dict).transpose()

        pred_support = Counter(y_pred)
        df_report["predicted_support"] = df_report.index.map(
            lambda label: pred_support.get(
                {
                    "LEFT-LEFT": 0,
                    "LEFT-RIGHT": 1,
                    "RIGHT-LEFT": 2,
                    "RIGHT-RIGHT": 3,
                }.get(label, -1),
                0,
            )
            if isinstance(label, str)
            else 0
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
        if not isinstance(other, DescriptionsToSidesExperiment):
            print(
                f"WARN: Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )
            return False

        self.instances.extend(other.instances)
        return True
