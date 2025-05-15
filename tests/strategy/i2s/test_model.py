import tempfile
from typing import Tuple

from src.strategy.i2s.model import (
    Answer,
    BongardProblemImages,
    ExpectedAnswer,
    ImageToSideResponse,
    ImagesToSidesExperiment,
    ImagesToSidesExperimentInstance,
    ImagesToSidesResponse,
)


def make_images(
    whole: str = "whole", first: str = "first", second: str = "second"
) -> BongardProblemImages:
    return BongardProblemImages(
        whole_image_path=whole,
        first_test_image_path=first,
        second_test_image_path=second,
    )


def make_instance(
    whole: str = "whole",
    first: str = "first",
    second: str = "second",
    actual_answer: Tuple[Answer, Answer] = (Answer.LEFT, Answer.RIGHT),
    expected_answer: Tuple[Answer, Answer] = (Answer.LEFT, Answer.RIGHT),
) -> ImagesToSidesExperimentInstance:
    return ImagesToSidesExperimentInstance(
        problem_id=1,
        images=make_images(whole, first, second),
        response=ImagesToSidesResponse(
            concept="concept",
            first=ImageToSideResponse(
                explanation="explanation",
                answer=actual_answer[0],
            ),
            second=ImageToSideResponse(
                explanation="explanation",
                answer=actual_answer[1],
            ),
        ),
        expected_answer=ExpectedAnswer(
            first=expected_answer[0],
            second=expected_answer[1],
        ),
    )


def test_has_solution():
    experiment = ImagesToSidesExperiment(solver_name="dummy")
    experiment.add_solution(make_instance("w", "1", "2"))
    assert experiment.has_solution(make_images("w", "1", "2"))
    assert experiment.has_solution(make_images("w", "2", "1"))
    assert not experiment.has_solution(make_images("other"))


def test_to_file_and_from_file():
    source = ImagesToSidesExperiment(solver_name="dummy")
    source.add_solution(make_instance("first"))
    source.add_solution(make_instance("second"))
    with tempfile.NamedTemporaryFile() as f:
        source.to_file(f.name)
        f.seek(0)
        target = ImagesToSidesExperiment.from_file(f.name)
    print(target)
    assert target.has_solution(make_images("first"))
    assert target.has_solution(make_images("second"))
    assert not target.has_solution(make_images("third"))


def test_evaluate():
    correct = make_instance(
        actual_answer=(Answer.LEFT, Answer.RIGHT),
        expected_answer=(Answer.LEFT, Answer.RIGHT),
    )
    assert correct.evaluate() == True
    incorrect = make_instance(
        actual_answer=(Answer.RIGHT, Answer.LEFT),
        expected_answer=(Answer.LEFT, Answer.RIGHT),
    )
    assert incorrect.evaluate() == False


def test_summarize():
    experiment = ImagesToSidesExperiment(solver_name="dummy")
    experiment.add_solution(make_instance("first"))
    experiment.add_solution(make_instance("second"))
    expected = """
{'num_correct_answers': 2, 'num_all_answers': 2, 'acc': 100.0}
              precision  recall  f1-score  support  predicted_support
LEFT-LEFT          0.00    0.00      0.00     0.00                  0
LEFT-RIGHT         1.00    1.00      1.00     2.00                  2
RIGHT-LEFT         0.00    0.00      0.00     0.00                  0
RIGHT-RIGHT        0.00    0.00      0.00     0.00                  0
accuracy           1.00    1.00      1.00     1.00                  0
macro avg          0.25    0.25      0.25     2.00                  0
weighted avg       1.00    1.00      1.00     2.00                  0
""".strip()
    assert experiment.summarize() == expected
