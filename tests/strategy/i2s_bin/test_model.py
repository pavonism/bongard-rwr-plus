import tempfile

from src.strategy.i2s_bin.model import (
    Answer,
    BongardProblemImages,
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
    actual_answer: Answer = Answer.LEFT_RIGHT,
    expected_answer: Answer = Answer.LEFT_RIGHT,
) -> ImagesToSidesExperimentInstance:
    return ImagesToSidesExperimentInstance(
        problem_id=1,
        images=make_images(whole, first, second),
        response=ImagesToSidesResponse(
            concept="concept",
            explanation="explanation",
            answer=actual_answer,
        ),
        expected_answer=expected_answer,
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
    with tempfile.NamedTemporaryFile(delete=False) as f:
        source.to_file(f.name)
        f.seek(0)
        target = ImagesToSidesExperiment.from_file(f.name)
    print(target)
    assert target.has_solution(make_images("first"))
    assert target.has_solution(make_images("second"))
    assert not target.has_solution(make_images("third"))


def test_evaluate():
    correct = make_instance(
        actual_answer=Answer.LEFT_RIGHT,
        expected_answer=Answer.LEFT_RIGHT,
    )
    assert correct.evaluate() == True
    incorrect = make_instance(
        actual_answer=Answer.RIGHT_LEFT,
        expected_answer=Answer.LEFT_RIGHT,
    )
    assert incorrect.evaluate() == False


def test_summarize():
    experiment = ImagesToSidesExperiment(solver_name="dummy")
    experiment.add_solution(
        make_instance(
            "first",
            actual_answer=Answer.LEFT_RIGHT,
            expected_answer=Answer.LEFT_RIGHT,
        )
    )
    experiment.add_solution(
        make_instance(
            "second",
            actual_answer=Answer.RIGHT_LEFT,
            expected_answer=Answer.LEFT_RIGHT,
        )
    )
    expected = """
{'num_correct_answers': 1, 'num_all_answers': 2, 'acc': 50.0}
              precision  recall  f1-score  support  predicted_support
LEFT_RIGHT         1.00    0.50      0.67     2.00                  1
RIGHT_LEFT         0.00    0.00      0.00     0.00                  1
accuracy           0.50    0.50      0.50     0.50                  0
macro avg          0.50    0.25      0.33     2.00                  0
weighted avg       1.00    0.50      0.67     2.00                  0
""".strip()
    assert experiment.summarize() == expected
