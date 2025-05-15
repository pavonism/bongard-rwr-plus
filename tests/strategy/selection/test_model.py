import tempfile

from src.strategy.selection.model import (
    Concept,
    SelectionExperiment,
    SelectionExperimentInstance,
    SelectionInput,
    SelectionRequest,
    SelectionResponse,
)


def make_instance(
    image_path: str = "image_path", expected_label: int = 1
) -> SelectionExperimentInstance:
    return SelectionExperimentInstance(
        problem_id=1,
        input=SelectionInput(
            image_path=image_path,
        ),
        request=SelectionRequest(
            concepts=[
                Concept(
                    left="l1",
                    right="r1",
                    label=1,
                ),
                Concept(
                    left="l2",
                    right="r2",
                    label=2,
                ),
            ],
        ),
        response=SelectionResponse(
            explanation="explanation",
            label=1,
        ),
        expected_label=expected_label,
    )


def test_has_solution():
    experiment = SelectionExperiment(solver_name="dummy")
    experiment.add_solution(make_instance("first"))
    assert experiment.has_solution("first")
    assert not experiment.has_solution("other")


def test_to_file_and_from_file():
    source = SelectionExperiment(solver_name="dummy")
    source.add_solution(make_instance("first"))
    source.add_solution(make_instance("second"))
    with tempfile.NamedTemporaryFile() as f:
        source.to_file(f.name)
        f.seek(0)
        target = SelectionExperiment.from_file(f.name)
    print(target)
    assert target.has_solution("first")
    assert target.has_solution("second")
    assert not target.has_solution("third")


def test_evaluate():
    assert make_instance("path", 1).evaluate() == True
    assert make_instance("path", 2).evaluate() == False


def test_summarize():
    experiment = SelectionExperiment(solver_name="dummy")
    experiment.add_solution(make_instance("first", 1))
    experiment.add_solution(make_instance("first", 2))
    expected = """
{'num_correct_answers': 1, 'num_all_answers': 2, 'acc': 50.0}
              precision    recall  f1-score   support

           1       0.50      1.00      0.67         1
           2       0.00      0.00      0.00         1

    accuracy                           0.50         2
   macro avg       0.25      0.50      0.33         2
weighted avg       0.25      0.50      0.33         2
""".strip()
    assert experiment.summarize() == expected
