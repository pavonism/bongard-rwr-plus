from collections import Counter
from enum import Enum
from typing import Dict, List, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import classification_report

from src.dataset.model import BaseExperiment
from src.messenger.content import Content, ImageContent, TextContent


class BongardProblemImages(BaseModel):
    whole_image_path: str
    first_test_image_path: str
    second_test_image_path: str

    def orderless_equals(self, other: "BongardProblemImages") -> bool:
        if (
            self.whole_image_path == other.whole_image_path
            and self.first_test_image_path == other.first_test_image_path
            and self.second_test_image_path == other.second_test_image_path
        ):
            return True
        if (
            self.whole_image_path == other.whole_image_path
            and self.first_test_image_path == other.second_test_image_path
            and self.second_test_image_path == other.first_test_image_path
        ):
            return True
        return False

    def contents(self) -> List[Content]:
        return [
            TextContent("Bongard problem"),
            ImageContent(self.whole_image_path),
            TextContent("First test image"),
            ImageContent(self.first_test_image_path),
            TextContent("Second test image"),
            ImageContent(self.second_test_image_path),
        ]


class Answer(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class ExpectedAnswer(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "not": {
                        "properties": {
                            "first": {"enum": ["LEFT"]},
                            "second": {"enum": ["LEFT"]},
                        },
                        "required": ["first", "second"],
                    }
                },
                {
                    "not": {
                        "properties": {
                            "first": {"enum": ["RIGHT"]},
                            "second": {"enum": ["RIGHT"]},
                        },
                        "required": ["first", "second"],
                    }
                },
            ]
        }
    )

    first: Answer
    second: Answer


class ImageToSideResponse(BaseModel):
    explanation: str
    answer: Answer


class ImagesToSidesResponse(BaseModel):
    concept: str
    first: ImageToSideResponse
    second: ImageToSideResponse


class ImagesToSidesExperimentInstance(BaseModel):
    problem_id: int
    images: BongardProblemImages
    response: ImagesToSidesResponse
    expected_answer: ExpectedAnswer

    def evaluate(self) -> bool:
        return (
            self.response.first.answer == self.expected_answer.first
            and self.response.second.answer == self.expected_answer.second
        )


class ImagesToSidesExperiment(BaseExperiment):
    solver_name: str
    instances: List[ImagesToSidesExperimentInstance] = []

    def has_solution(self, images: BongardProblemImages) -> bool:
        for instance in self.instances:
            if instance.images.orderless_equals(images):
                return True
        return False

    def add_solution(self, instance: ImagesToSidesExperimentInstance):
        self.instances.append(instance)

    def summarize(self) -> str:
        answers_to_label = {
            (Answer.LEFT, Answer.LEFT): 0,
            (Answer.LEFT, Answer.RIGHT): 1,
            (Answer.RIGHT, Answer.LEFT): 2,
            (Answer.RIGHT, Answer.RIGHT): 3,
        }

        y_true, y_pred = [], []
        for instance in self.instances:
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
        if not isinstance(other, ImagesToSidesExperiment):
            print(
                f"WARN: Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )
            return False

        self.instances.extend(other.instances)
        return True
