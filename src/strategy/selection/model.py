from typing import List, Dict, Union

from pydantic import BaseModel
from sklearn.metrics import classification_report

from src.dataset.model import BaseExperiment
from src.messenger.content import Content, ImageContent, TextContent


class SelectionInput(BaseModel):
    image_path: str

    def contents(self) -> List[Content]:
        return [
            TextContent("Bongard problem"),
            ImageContent(self.image_path),
        ]


class Concept(BaseModel):
    left: str
    right: str
    label: int


class SelectionRequest(BaseModel):
    concepts: List[Concept]


class SelectionResponse(BaseModel):
    explanation: str
    label: int


class SelectionExperimentInstance(BaseModel):
    problem_id: int
    input: SelectionInput
    request: SelectionRequest
    response: SelectionResponse
    expected_label: int

    def evaluate(self) -> bool:
        return self.expected_label == self.response.label


class SelectionExperiment(BaseExperiment):
    solver_name: str
    instances: List[SelectionExperimentInstance] = []

    def has_solution(self, image_path: str) -> bool:
        for instance in self.instances:
            if instance.input.image_path == image_path:
                return True
        return False

    def add_solution(self, instance: SelectionExperimentInstance):
        self.instances.append(instance)

    def summarize(self) -> str:
        num_correct = 0
        y_true, y_pred = [], []
        for instance in self.instances:
            num_correct += int(instance.evaluate())
            y_true.append(instance.expected_label)
            y_pred.append(instance.response.label)
        summary = f"{self.metrics()}\n"
        summary += classification_report(
            y_true, y_pred, labels=list(range(min(y_true), max(y_true) + 1))
        )
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
        if not isinstance(other, SelectionExperiment):
            print(
                f"WARN: Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )
            return False

        self.instances.extend(other.instances)
        return True
