import pandas as pd
import pytest

from src.strategy.selection.selection import get_negative_concept_indices

NUM_CHOICES = 16


@pytest.mark.parametrize(
    "labels_file",
    [
        "/home/mikmal/Projects/llm-avr-benchmarks/data/raw/labels.csv",
        "/home/mikmal/Projects/llm-avr-benchmarks/data/raw/bongard_hoi_mix_labels.csv",
        "/home/mikmal/Projects/llm-avr-benchmarks/data/raw/bongard_open_world_labels.csv",
    ],
)
def test_get_negative_concept_indices(labels_file: str):
    df = pd.read_csv(labels_file)
    for positive_idx in range(100):
        idxs = get_negative_concept_indices(
            positive_idx=positive_idx,
            num_negatives=NUM_CHOICES - 1,
            df=df,
        )
        assert 1 + len(idxs) == NUM_CHOICES
        assert positive_idx not in idxs
