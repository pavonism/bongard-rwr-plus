import tempfile

from src.strategy.i1s.model import (
    Answer,
    BongardProblemImages,
    ImageToSideExperiment,
    ImageToSideExperimentInstance,
    ImageToSideResponse,
)


def make_images(whole: str = "whole", test: str = "test") -> BongardProblemImages:
    return BongardProblemImages(whole_image_path=whole, test_image_path=test)


def make_instance(
    whole: str = "whole",
    test: str = "test",
    actual_answer: Answer = Answer.LEFT,
    expected_answer: Answer = Answer.LEFT,
) -> ImageToSideExperimentInstance:
    return ImageToSideExperimentInstance(
        problem_id=1,
        images=make_images(whole, test),
        response=ImageToSideResponse(
            concept="concept",
            explanation="explanation",
            answer=actual_answer,
        ),
        expected_answer=expected_answer,
    )


def test_has_solution():
    experiment = ImageToSideExperiment(solver_name="dummy")
    experiment.add_solution(make_instance("w", "1"))
    assert experiment.has_solution(make_images("w", "1"))
    assert not experiment.has_solution(make_images("other"))


def test_to_file_and_from_file():
    source = ImageToSideExperiment(solver_name="dummy")
    source.add_solution(make_instance("first"))
    source.add_solution(make_instance("second"))
    with tempfile.NamedTemporaryFile() as f:
        source.to_file(f.name)
        f.seek(0)
        target = ImageToSideExperiment.from_file(f.name)
    print(target)
    assert target.has_solution(make_images("first"))
    assert target.has_solution(make_images("second"))
    assert not target.has_solution(make_images("third"))


def test_evaluate():
    correct = make_instance(actual_answer=Answer.LEFT, expected_answer=Answer.LEFT)
    assert correct.evaluate() == True
    incorrect = make_instance(actual_answer=Answer.RIGHT, expected_answer=Answer.LEFT)
    assert incorrect.evaluate() == False


def test_summarize():
    experiment = ImageToSideExperiment(solver_name="dummy")
    experiment.add_solution(make_instance("first", actual_answer=Answer.LEFT, expected_answer=Answer.LEFT))
    experiment.add_solution(make_instance("second", actual_answer=Answer.RIGHT, expected_answer=Answer.LEFT))
    experiment.add_solution(make_instance("third", actual_answer=Answer.RIGHT, expected_answer=Answer.RIGHT))
    expected = """
{'num_correct_answers': 2, 'num_all_answers': 3, 'acc': 66.67}
              precision  recall  f1-score  support  predicted_support
LEFT               1.00    0.50      0.67     2.00                  1
RIGHT              0.50    1.00      0.67     1.00                  2
accuracy           0.67    0.67      0.67     0.67                  0
macro avg          0.75    0.75      0.67     3.00                  0
weighted avg       0.83    0.67      0.67     3.00                  0
""".strip()
    assert experiment.summarize() == expected
