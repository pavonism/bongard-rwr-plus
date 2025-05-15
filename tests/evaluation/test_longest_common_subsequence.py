import pytest

from src.evaluation.longest_common_subsequence import (
    get_longest_common_word_subsequence_length,
    get_longest_common_word_subsequence_length_memoization,
)


@pytest.mark.parametrize(
    "first, second, expected_length",
    [
        ("An acute angle directed inward", "No angle directed inward", 3),
        ("More solid black figures", "More solid black circles", 3),
        ("More solid black figures", "A line and More solid black circles", 3),
        ("A line and More solid black figures", "More solid black circles", 3),
        ("A line with a self-crossing", "More solid black circles", 0),
    ],
)
def test_get_longest_common_word_subsequence_length(
    first: str, second: str, expected_length: int
):
    assert expected_length == get_longest_common_word_subsequence_length(first, second)
    assert expected_length == get_longest_common_word_subsequence_length_memoization(
        first, second
    )
