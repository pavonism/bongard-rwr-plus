import tempfile
import pytest

from src.messenger.content import ImageContent, TextContent
from src.dataset.model import BongardDatasetInfo, BongardImage, BongardProblem
from src.strategy.d2s.model import (
    Answer,
    BongardDatasetDescriptions,
    BongardDescriptionsDictionary,
    BongardProblemDescriptions,
    ExpectedAnswer,
    DescriptionToSideResponse,
    DescriptionsToSidesExperiment,
    DescriptionsToSidesExperimentInstance,
    DescriptionsToSidesResponse,
    ImageDescription,
)


def make_dataset_info() -> BongardDatasetInfo:
    return BongardDatasetInfo(
        problems=[
            BongardProblem(
                id=1,
                left_images=[
                    BongardImage(image_id=0, path="left1.png"),
                    BongardImage(image_id=1, path="left2.png"),
                ],
                right_images=[
                    BongardImage(image_id=2, path="right1.png"),
                    BongardImage(image_id=3, path="right2.png"),
                ],
            ),
            BongardProblem(
                id=2,
                left_images=[
                    BongardImage(image_id=4, path="left1.png"),
                    BongardImage(image_id=5, path="left2.png"),
                ],
                right_images=[
                    BongardImage(image_id=6, path="right1.png"),
                    BongardImage(image_id=7, path="right2.png"),
                ],
            ),
        ]
    )


def make_problem_descriptions():
    return BongardProblemDescriptions(
        left_descriptions=[
            ImageDescription(
                image_id=0,
                path="left1.png",
                description="A left image",
            ),
            ImageDescription(
                image_id=1,
                path="left2.png",
                description="Another left image",
            ),
        ],
        right_descriptions=[
            ImageDescription(
                image_id=2,
                path="right1.png",
                description="A right image",
            ),
            ImageDescription(
                image_id=3,
                path="right2.png",
                description="Another right image",
            ),
        ],
    )


def make_instance(problem_id: int = 1):
    return DescriptionsToSidesExperimentInstance(
        problem_id=problem_id,
        descriptions=make_problem_descriptions(),
        response=DescriptionsToSidesResponse(
            concept="concept",
            first=DescriptionToSideResponse(
                explanation="explanation",
                answer=Answer.LEFT,
            ),
            second=DescriptionToSideResponse(
                explanation="explanation",
                answer=Answer.RIGHT,
            ),
        ),
        expected_answer=ExpectedAnswer(
            first=Answer.LEFT,
            second=Answer.RIGHT,
        ),
    )


def make_dataset_descriptions() -> BongardDatasetDescriptions:
    return BongardDatasetDescriptions(
        descriptor_name="Author",
        img_descriptions=[
            ImageDescription(
                image_id=0,
                path="left1.png",
                description="A left image",
            ),
            ImageDescription(
                image_id=1,
                path="left2.png",
                description="Another left image",
            ),
            ImageDescription(
                image_id=2,
                path="right1.png",
                description="A right image",
            ),
            ImageDescription(
                image_id=3,
                path="right2.png",
                description="Another right image",
            ),
        ],
    )


def make_problem(problem_id: int = 1) -> BongardProblem:
    return BongardProblem(
        id=problem_id,
        left_images=[
            BongardImage(image_id=0, path="left1.png"),
            BongardImage(image_id=1, path="left2.png"),
        ],
        right_images=[
            BongardImage(image_id=2, path="right1.png"),
            BongardImage(image_id=3, path="right2.png"),
        ],
    )


def test_subset():
    dataset_info = make_dataset_info()
    subset = dataset_info.subset([1])
    assert len(subset.problems) == 1
    assert subset.problems[0].id == 1


def test_collect_images():
    dataset_info = make_dataset_info()
    images = dataset_info.collect_images()
    assert len(images) == 8
    assert all(isinstance(img, BongardImage) for img in images)


def test_bongard_image_content():
    image = BongardImage(image_id=0, path="left1.png")
    assert isinstance(image.content(), ImageContent)


def test_image_description_content():
    image = ImageDescription(image_id="0", path="left1.png", description="A left image")
    assert isinstance(image.content(), TextContent)


def test_random_expected_answer():
    answer = ExpectedAnswer.random()
    assert answer.first in [Answer.LEFT, Answer.RIGHT]
    assert answer.second in [Answer.LEFT, Answer.RIGHT]
    assert answer.first != answer.second


@pytest.mark.parametrize(
    "expected_answer",
    [
        ExpectedAnswer(first=Answer.LEFT, second=Answer.RIGHT),
        ExpectedAnswer(first=Answer.RIGHT, second=Answer.LEFT),
    ],
)
def test_create_request(expected_answer: ExpectedAnswer):
    problem = make_problem()
    descriptions = make_problem_descriptions()
    request = descriptions.create_request(problem, expected_answer)

    first_expected_images = (
        descriptions.left_descriptions
        if expected_answer.first == Answer.LEFT
        else descriptions.right_descriptions
    )
    second_expected_images = (
        descriptions.right_descriptions
        if expected_answer.first == Answer.LEFT
        else descriptions.left_descriptions
    )

    assert request.first == first_expected_images[-1].description
    assert request.second == second_expected_images[-1].description
    assert request.left_descriptions == [
        desc.description for desc in descriptions.left_descriptions[:-1]
    ]
    assert request.right_descriptions == [
        desc.description for desc in descriptions.right_descriptions[:-1]
    ]


def test_evaluate():
    instance = make_instance()
    assert instance.evaluate() is True

    instance.response.first.answer = Answer.RIGHT
    assert instance.evaluate() is False


def test_to_dict():
    descriptions = make_dataset_descriptions().to_dict()
    assert isinstance(descriptions, BongardDescriptionsDictionary)
    assert len(descriptions.descriptions) == 4
    assert descriptions.descriptor_name == "Author"


def test_has_image_description():
    descriptions = BongardDescriptionsDictionary(descriptor_name="Author")
    image = BongardImage(image_id="0", path="left1.png")
    assert not descriptions.has_image_description(image)

    description = ImageDescription(
        image_id=0,
        path="left1.png",
        description="A left image",
    )
    descriptions.add_description(description)
    assert descriptions.has_image_description(image)


def test_add_description():
    descriptions = BongardDescriptionsDictionary(descriptor_name="Author")
    description = ImageDescription(
        image_id=0,
        path="left1.png",
        description="A left image",
    )
    descriptions.add_description(description)
    assert len(descriptions.descriptions) == 1
    assert descriptions.descriptions[0] == description


def test_has_all_problem_descriptions():
    descriptions = BongardDescriptionsDictionary(descriptor_name="Author")
    problem = make_problem()
    assert not descriptions.has_all_problem_descriptions(problem)

    for img in problem.left_images + problem.right_images:
        description = ImageDescription(
            image_id=img.image_id,
            path=img.path,
            description=f"A {img.path} image",
        )
        descriptions.add_description(description)

    assert descriptions.has_all_problem_descriptions(problem)


def test_get_descriptions_for_problem():
    descriptions = BongardDescriptionsDictionary(descriptor_name="Author")
    problem = make_problem()
    for img in problem.left_images + problem.right_images:
        description = ImageDescription(
            image_id=img.image_id,
            path=img.path,
            description=f"A {img.path} image",
        )
        descriptions.add_description(description)

    result = descriptions.get_descriptions_for_problem(problem)
    assert result is not None
    assert len(result.left_descriptions) == 2
    assert len(result.right_descriptions) == 2


def test_flatten():
    descriptions = make_dataset_descriptions().to_dict()
    flattened = descriptions.flatten()
    assert isinstance(flattened, BongardDatasetDescriptions)
    assert flattened.descriptor_name == descriptions.descriptor_name
    assert len(flattened.img_descriptions) == len(descriptions.descriptions)


def test_add_solution():
    experiment = DescriptionsToSidesExperiment(
        solver_name="dummy", descriptor_name="Author"
    )
    instance = make_instance()
    experiment.add_solution(instance)
    assert experiment.instances[0] == instance


def test_has_problem_solution():
    experiment = DescriptionsToSidesExperiment(
        solver_name="dummy", descriptor_name="Author"
    )
    experiment.add_solution(make_instance())
    assert experiment.has_problem_solution(make_problem())


def test_to_file_and_from_file():
    source = DescriptionsToSidesExperiment(
        solver_name="dummy", descriptor_name="Author"
    )
    source.add_solution(make_instance(problem_id=1))
    source.add_solution(make_instance(problem_id=2))
    with tempfile.NamedTemporaryFile(delete=False) as f:
        source.to_file(f.name)
        f.seek(0)
        target = DescriptionsToSidesExperiment.from_file(f.name)
    print(target)
    assert target.has_problem_solution(make_problem(1))
    assert target.has_problem_solution(make_problem(2))
    assert not target.has_problem_solution(make_problem(3))


def test_to_file_and_from_file_descriptions():
    source = make_dataset_descriptions()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        source.to_file(f.name)
        f.seek(0)
        target = BongardDatasetDescriptions.from_file(f.name)
    print(target)
    assert target.descriptor_name == source.descriptor_name
    assert len(target.img_descriptions) == len(source.img_descriptions)
    assert target.img_descriptions[0].image_id == source.img_descriptions[0].image_id
