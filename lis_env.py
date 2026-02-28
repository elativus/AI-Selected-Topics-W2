"""LIS RL environment (Week 2 assignment).

This module contains:
- Data / Verifier / Env interfaces (per assignment PDF)
- LISVerifier: extracts & verifies LIS length answers
- LISEnv: generates tasks with adjustable difficulty

Kept intentionally lightweight so it can be imported from both TRAIN and EVAL notebooks.
"""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from bisect import bisect_left
from typing import Any, Dict, List, Optional, Type

from prompt_templates import build_lis_prompt


class Data:
    """Container for a single task.

    Matches the interface shown in the assignment PDF:
    - question: str
    - answer: str
    - difficulty: int in [1, 10]
    - metadata: dict (optional)
    """

    def __init__(
        self,
        question: str,
        answer: str,
        difficulty: int = 1,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ):
        self.question = question
        self.answer = answer
        self.difficulty = int(difficulty)
        self.metadata = metadata or {}
        self.gpt_response = ""

    def to_json(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response,
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "Data":
        instance = cls(
            question=json_dict.get("question") or json_dict.get("prompt") or "",
            answer=json_dict.get("answer", ""),
            difficulty=json_dict.get("difficulty", 1),
            metadata=json_dict.get("metadata", {}) or {},
        )
        if "gpt_response" in json_dict:
            instance.gpt_response = json_dict["gpt_response"]
        return instance

    @classmethod
    def from_jsonl_file(cls, file_path: str) -> List["Data"]:
        items: List[Data] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(cls.from_json_dict(json.loads(line)))
        return items


class Verifier(ABC):
    """Base class for verifiers (per assignment interface)."""

    def __init__(self):
        pass

    @abstractmethod
    def verify(self, data: Data, test_solution: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_answer(self, test_solution: str) -> Optional[str]:
        raise NotImplementedError


class Env(ABC):
    """Base class for an RL environment (per assignment interface)."""

    def __init__(self, name: str, verifier: Type[Verifier]):
        self.name = name
        # The PDF expects passing a Verifier *class* and instantiating it here.
        self.verifier = verifier()

    @abstractmethod
    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs: Any,
    ) -> List[Data]:
        """Generate a list of tasks."""
        raise NotImplementedError

    def verify(self, data: Data, test_solution: str) -> bool:
        return self.verifier.verify(data, test_solution)

    @abstractmethod
    def extract_answer(self, test_solution: str) -> Optional[str]:
        raise NotImplementedError

    def sample_task(
        self,
        rng: Optional[random.Random] = None,
        difficulty: Optional[int] = 1,
        **kwargs: Any,
    ) -> Data:
        """Convenience helper: sample a single task.

        Not required by the assignment interface, but used in the notebooks.
        If rng is provided, sampling becomes reproducible.
        """
        # If caller didn't provide explicit seed, derive one from rng.
        seed = kwargs.pop("seed", None)
        if seed is None and rng is not None:
            seed = rng.randint(0, 10_000_000)

        items = self.generate(num_of_questions=1, difficulty=difficulty, seed=seed, **kwargs)
        if not items:
            raise RuntimeError("Env.generate() returned empty list")
        return items[0]


def lis_length(seq: List[int]) -> int:
    """Length of LIS in O(n log n) for STRICTLY increasing subsequence."""
    tails: List[int] = []
    for x in seq:
        i = bisect_left(tails, x)  # strict increasing
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


class LISVerifier(Verifier):
    """Verifier for LIS-length tasks."""

    ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
    INT_RE = re.compile(r"-?\d+")
    ONLY_INT_RE = re.compile(r"^\s*-?\d+\s*$")
    # Non-greedy: removes each bracketed chunk like [1, 2, 3] (possibly multiline).
    BRACKETED_RE = re.compile(r"\[[\s\S]*?\]")

    # If the model violates the output format and includes extra numbers (e.g. a subsequence/dp list),
    # we only accept an integer that is explicitly marked as the final answer.
    KEYWORD_PATTERNS = [
        re.compile(r"(?:\bLIS\b\s*)?(?:length|len)\s*(?:is|=|:)\s*(-?\d+)", re.IGNORECASE),
        re.compile(r"\banswer\b\s*(?:is|=|:)\s*(-?\d+)", re.IGNORECASE),
        re.compile(r"\bfinal\b\s*\banswer\b\s*(?:is|=|:)\s*(-?\d+)", re.IGNORECASE),
        # Russian variants (sometimes appear in completions)
        re.compile(r"\bответ\b\s*(?:это|=|:)\s*(-?\d+)", re.IGNORECASE),
        re.compile(r"\bдлина\b\s*(?:это|=|:)\s*(-?\d+)", re.IGNORECASE),
    ]

    @classmethod
    def _extract_int_from_text(cls, text: str) -> Optional[str]:
        """Extract a single, unambiguous integer answer from a text chunk.

        Rules (in order):
        1) If the whole chunk is just an integer => accept.
        2) If there's an explicitly marked answer ("length=", "answer:") => accept that.
        3) Otherwise, strip bracketed lists like [1,2,3] and accept only if exactly one integer remains.

        If multiple integers remain with no explicit marker, return None (avoid false positives).
        """
        if text is None:
            return None

        cand = str(text).strip()
        if not cand:
            return None

        # Strictly formatted: <answer> 7 </answer>
        if cls.ONLY_INT_RE.fullmatch(cand):
            return cand.strip()

        # Pure list/array inside <answer> is considered invalid for this task.
        if cand.startswith("[") and cand.endswith("]"):
            return None

        # Look for explicit markers.
        for pat in cls.KEYWORD_PATTERNS:
            m = pat.search(cand)
            if m:
                return m.group(1)

        # Remove bracketed lists to avoid accidentally taking an element from a printed list.
        no_lists = cls.BRACKETED_RE.sub(" ", cand)

        # If there's a line that is exactly an integer, prefer the *last* such line.
        lines = [ln.strip() for ln in no_lists.splitlines() if ln.strip()]
        for ln in reversed(lines):
            if cls.ONLY_INT_RE.fullmatch(ln):
                return ln.strip()

        ints = cls.INT_RE.findall(no_lists)
        if len(ints) == 1:
            return ints[0]
        return None

    def extract_answer(self, test_solution: str) -> Optional[str]:
        if test_solution is None:
            return None

        # 1) Preferred: <answer>...</answer>
        m = self.ANSWER_TAG_RE.search(test_solution)
        if m:
            # Avoid extracting a random element from a printed list.
            return self._extract_int_from_text(m.group(1))

        # 2) Fallback: try to parse the whole completion.
        return self._extract_int_from_text(test_solution)

    def verify(self, data: Data, test_solution: str) -> bool:
        pred = self.extract_answer(test_solution)
        if pred is None:
            return False
        try:
            return int(pred) == int(data.answer)
        except Exception:
            return False


class LISEnv(Env):
    """Environment that generates LIS-length tasks."""

    def __init__(self, name: str = "lis_length", verifier: Type[Verifier] = LISVerifier):
        super().__init__(name=name, verifier=verifier)

    def difficulty_to_params(self, difficulty: int) -> Dict[str, Any]:
        d = int(max(1, min(10, difficulty)))
        # Sequence length grows with difficulty.
        seq_len = 6 + 2 * d  # 8..26
        value_abs = 5 + 3 * d  # 8..35
        return {
            "seq_len": seq_len,
            "low": -value_abs,
            "high": value_abs,
        }

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs: Any,
    ) -> List[Data]:
        # Support direct hyperparams (seq_len/low/high/seed) OR difficulty mapping.
        seed = kwargs.get("seed", None)
        rng = random.Random(seed) if seed is not None else random

        if any(k in kwargs for k in ["seq_len", "low", "high"]):
            seq_len = int(kwargs.get("seq_len", 10))
            low = int(kwargs.get("low", -10))
            high = int(kwargs.get("high", 10))
            d = int(kwargs.get("difficulty", difficulty if difficulty is not None else 1))
        else:
            d = int(difficulty if difficulty is not None else 1)
            params = self.difficulty_to_params(d)
            seq_len, low, high = params["seq_len"], params["low"], params["high"]

        items: List[Data] = []
        for _ in range(int(num_of_questions)):
            ok = False
            for _attempt in range(int(max_attempts)):
                seq = [rng.randint(low, high) for _ in range(seq_len)]
                ans = lis_length(seq)

                # Drop trivial sequences.
                if 1 < ans < seq_len:
                    question = build_lis_prompt(seq)
                    items.append(
                        Data(
                            question=question,
                            answer=str(ans),
                            difficulty=d,
                            metadata={
                                "seq": seq,
                                "seq_len": seq_len,
                                "low": low,
                                "high": high,
                            },
                        )
                    )
                    ok = True
                    break
            if not ok:
                raise RuntimeError(
                    f"Failed to sample a non-trivial task in {max_attempts} attempts (difficulty={d})."
                )
        return items

    def extract_answer(self, test_solution: str) -> Optional[str]:
        return self.verifier.extract_answer(test_solution)


__all__ = [
    "Data",
    "Verifier",
    "Env",
    "lis_length",
    "LISVerifier",
    "LISEnv",
]
