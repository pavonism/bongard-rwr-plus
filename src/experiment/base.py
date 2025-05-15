from typing import List

from src.messenger.vllm_messenger import VllmMessenger
from src.strategy.i1d.method import generate_descriptions
from src.strategy.d1s.method import resolve_description_to_side
from src.strategy.d2s.method import resolve_descriptions_to_sides
from src.strategy.d2s_bin.method import resolve_descriptions_to_sides_binary
from src.strategy.i1s.method import resolve_image_to_side
from src.strategy.i2s.method import resolve_images_to_sides
from src.strategy.i2s_bin.method import resolve_images_to_sides_binary
from src.strategy.selection.selection import resolve_selection


def set_up_messengers(messengers: List[VllmMessenger], log_directory: str):
    for messenger in messengers:
        messenger.set_log_directory(log_directory)


async def run(
    args,
    messengers: List[VllmMessenger],
    problem_ids: List[int],
    reevaluate: bool = False,
):
    dataset = args.dataset
    data_dir = args.data_dir
    labels_file = args.labels_file
    solver = messengers[0].get_name()

    strategies = args.strategies
    prompt_version = args.prompt_version
    dataset_path = f"{data_dir}/{dataset}"

    strategies = set(strategies.split(","))
    print(f"strategies: {strategies}")

    if "all" in strategies or "i1s" in strategies:
        print("Binary classification: ImageToSide")
        experiment_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_i1s_{prompt_version}"
        set_up_messengers(messengers, experiment_dir)
        output_file = f"{experiment_dir}/{solver}_experiment.json"
        await resolve_image_to_side(
            problem_ids=problem_ids,
            dataset_path=dataset_path,
            experiment_file=output_file,
            prompt_version=prompt_version,
            messengers=messengers,
            reevaluate=reevaluate,
        )

    if "all" in strategies or "i2s" in strategies:
        print("Multi label classification: ImagesToSides")
        experiment_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_i2s_{prompt_version}"
        set_up_messengers(messengers, experiment_dir)
        output_file = f"{experiment_dir}/{solver}_experiment.json"
        await resolve_images_to_sides(
            problem_ids=problem_ids,
            dataset_path=dataset_path,
            experiment_file=output_file,
            prompt_version=prompt_version,
            messengers=messengers,
            reevaluate=reevaluate,
        )

    if "all" in strategies or "i2s-bin" in strategies:
        print("Binary classification: ImagesToSides")
        experiment_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_i2s-bin_{prompt_version}"
        set_up_messengers(messengers, experiment_dir)
        output_file = f"{experiment_dir}/{solver}_experiment.json"
        await resolve_images_to_sides_binary(
            problem_ids=problem_ids,
            dataset_path=dataset_path,
            experiment_file=output_file,
            prompt_version=prompt_version,
            messengers=messengers,
            reevaluate=reevaluate,
        )

    if "all" in strategies or "i1d" in strategies:
        print("Generating descriptions")
        experiment_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_i1d_{prompt_version}"
        set_up_messengers(messengers, experiment_dir)
        output_file = f"{experiment_dir}/{solver}_descriptions.json"
        await generate_descriptions(
            problem_ids=problem_ids,
            dataset_path=dataset_path,
            descriptions_file=output_file,
            prompt_version=prompt_version,
            messengers=messengers,
            reevaluate=reevaluate,
        )

    if "all" in strategies or "d1s" in strategies:
        print("Binary classification: DescriptionToSide")
        desc_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_i1d_v1"
        experiment_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_d1s_{prompt_version}"
        set_up_messengers(messengers, experiment_dir)
        descriptor = args.descriptor_name or solver
        output_file = f"{experiment_dir}/{descriptor}/{solver}_experiment.json"
        descriptions_file = f"{desc_dir}/{descriptor}_descriptions.json"
        await resolve_description_to_side(
            problem_ids=problem_ids,
            dataset_path=dataset_path,
            experiment_file=output_file,
            descriptions_file=descriptions_file,
            prompt_version=prompt_version,
            messengers=messengers,
            reevaluate=reevaluate,
        )

    if "all" in strategies or "d2s" in strategies:
        print("Multi label classification: DescriptionsToSides")
        desc_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_i1d_v1"
        experiment_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_d2s_{prompt_version}"
        set_up_messengers(messengers, experiment_dir)
        descriptor = args.descriptor_name or solver
        output_file = f"{experiment_dir}/{descriptor}/{solver}_experiment.json"
        descriptions_file = f"{desc_dir}/{descriptor}_descriptions.json"
        await resolve_descriptions_to_sides(
            problem_ids=problem_ids,
            dataset_path=dataset_path,
            experiment_file=output_file,
            descriptions_file=descriptions_file,
            prompt_version=prompt_version,
            messengers=messengers,
            reevaluate=reevaluate,
        )

    if "all" in strategies or "d2s-bin" in strategies:
        print("Binary classification: DescriptionsToSides")
        desc_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_i1d_v1"
        experiment_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_d2s-bin_{prompt_version}"
        set_up_messengers(messengers, experiment_dir)
        descriptor = args.descriptor_name or solver
        output_file = f"{experiment_dir}/{descriptor}/{solver}_experiment.json"
        descriptions_file = f"{desc_dir}/{descriptor}_descriptions.json"
        await resolve_descriptions_to_sides_binary(
            problem_ids=problem_ids,
            dataset_path=dataset_path,
            experiment_file=output_file,
            descriptions_file=descriptions_file,
            prompt_version=prompt_version,
            messengers=messengers,
            reevaluate=reevaluate,
        )

    if "all" in strategies or "mc-selection" in strategies:
        print("Multiclass classification: Selection")
        for num_choices in args.num_choices:
            experiment_dir = f"{data_dir}/processed/bongard/MM-experiments/{dataset}_mc-selection-{num_choices}_{prompt_version}"
            set_up_messengers(messengers, experiment_dir)
            output_file = f"{experiment_dir}/{solver}_answers.json"
            await resolve_selection(
                problem_ids=problem_ids,
                dataset_path=dataset_path,
                labels_file=labels_file,
                output_file=output_file,
                prompt_version=prompt_version,
                messengers=messengers,
                num_choices=num_choices,
                reevaluate=reevaluate,
            )
