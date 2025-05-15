from typing import Dict, List
from pydantic import BaseModel

from src.dataset.model import BaseExperiment, BongardImage, BongardProblem
from src.messenger.content import Content, TextContent


class ImageDescription(BongardImage):
    description: str

    def content(self) -> Content:
        return TextContent(self.description)


class DescriptionsToSidesRequest(BaseModel):
    left_descriptions: List[str]
    right_descriptions: List[str]
    first: str
    second: str
    whole_image: str


class BongardProblemDescriptions(BaseModel):
    left_descriptions: List[ImageDescription]
    right_descriptions: List[ImageDescription]


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
