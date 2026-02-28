from typing import List

def build_lis_prompt(seq: List[int]) -> str:
    """Строит однозначный промпт на английском для задачи про длину LIS."""
    return (
        "You are given a sequence of integers.\n\n"
        "Task: compute the length of the Longest Increasing Subsequence (LIS).\n\n"
        "Definitions:\n"
        "- A subsequence is obtained by deleting zero or more elements without changing the order of remaining elements.\n"
        "- An increasing subsequence is STRICTLY increasing: each next element is > the previous one.\n\n"
        "Output requirements:\n"
        "- Return ONLY the LIS length as a single integer.\n"
        "- Do NOT output the subsequence itself or any list/array.\n\n"
        f"Sequence: {seq}\n"
    )
