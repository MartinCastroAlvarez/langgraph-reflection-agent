"""Bundled coding tasks for Reflexion: a prompt the model sees and tests it never sees.

The tests are trusted code written for this package. The model gets the prompt (and the
evaluator's feedback after a failure), never the test source, and its code runs only in
the :mod:`~reflection_agent.sandbox`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CODE_TASKS", "CodeTask", "get_task"]


@dataclass(frozen=True, slots=True)
class CodeTask:
    name: str
    prompt: str
    entry_point: str
    tests: tuple[str, ...]


CODE_TASKS: dict[str, CodeTask] = {
    task.name: task
    for task in (
        CodeTask(
            name="luhn",
            prompt=(
                "Write a Python function luhn_valid(number: str) -> bool that returns True "
                "when number passes the Luhn checksum. Spaces may appear between digits."
            ),
            entry_point="luhn_valid",
            tests=(
                'assert luhn_valid("79927398713") is True',
                'assert luhn_valid("79927398710") is False',
                'assert luhn_valid("4539 3195 0343 6467") is True',
                'assert luhn_valid("8273 1232 7352 0569") is False',
                'assert luhn_valid("0") is False',
                'assert luhn_valid("059a") is False',
            ),
        ),
        CodeTask(
            name="roman",
            prompt=(
                "Write a Python function roman_to_int(numeral: str) -> int that converts a "
                "Roman numeral such as 'MCMXCIV' to an integer."
            ),
            entry_point="roman_to_int",
            tests=(
                'assert roman_to_int("III") == 3',
                'assert roman_to_int("IV") == 4',
                'assert roman_to_int("IX") == 9',
                'assert roman_to_int("LVIII") == 58',
                'assert roman_to_int("MCMXCIV") == 1994',
                'assert roman_to_int("mmxxvi") == 2026',
            ),
        ),
        CodeTask(
            name="rle",
            prompt=(
                "Write Python functions rle_encode(text: str) -> str and rle_decode(code: str) "
                "-> str for run-length encoding, where 'aaabcc' encodes to '3a1b2c'."
            ),
            entry_point="rle_encode",
            tests=(
                'assert rle_encode("aaabcc") == "3a1b2c"',
                'assert rle_encode("") == ""',
                'assert rle_decode("3a1b2c") == "aaabcc"',
                'assert rle_decode("12x") == "x" * 12',
                'assert rle_decode(rle_encode("hello world")) == "hello world"',
            ),
        ),
    )
}


def get_task(name: str) -> CodeTask:
    try:
        return CODE_TASKS[name]
    except KeyError:
        raise ValueError(f"unknown code task {name!r}; one of {sorted(CODE_TASKS)}") from None
