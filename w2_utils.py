"""Shared utilities for Week 2 (TRAIN + EVAL notebooks).

Kept small and pragmatic:
- SYSTEM_PROMPT and chat prompt builder
- JSONL load/save helpers
- Answer parsing (integer inside <answer> or last integer fallback)
- vLLM evaluation helper (baseline vs trained)
- Paired bar plot helper

The goal is to remove duplication between notebooks without over-modularizing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# IMPORTANT: must match the system prompt used during training.
SYSTEM_PROMPT = """Отвечай в следующем формате:
<think>
...
</think>
<answer>
...
</answer>
"""


def build_chat_prompt(tokenizer: Any, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Builds a chat prompt (system + user) for Instruct models.

    Works with HuggingFace tokenizers that implement `apply_chat_template`.
    Falls back to a minimal plaintext format otherwise.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(user_prompt)},
    ]

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback (should almost never be used for Qwen2.5-Instruct)
    return f"{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n"


def cleanup_cuda() -> None:
    """Best-effort CUDA memory cleanup (useful in Colab/Kaggle)."""
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# -----------------------
# JSONL helpers
# -----------------------
JsonLikePath = Union[str, Path]


def load_jsonl(path: JsonLikePath) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[Dict[str, Any]], path: JsonLikePath) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_user_prompt(row: Dict[str, Any]) -> str:
    """Extracts the user prompt from a dataset row.

    Supports several common schemas:
    - {"prompt": ...}
    - {"question": ...}
    - {"input": ...}
    """
    if "prompt" in row:
        return str(row["prompt"])
    if "question" in row:
        return str(row["question"])
    if "input" in row:
        return str(row["input"])
    raise KeyError(f"Cannot find prompt/question/input field. keys={list(row.keys())}")


# -----------------------
# Answer parsing
# -----------------------
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")

_ONLY_INT_RE = re.compile(r"^\s*-?\d+\s*$")
_BRACKETED_RE = re.compile(r"\[[\s\S]*?\]")

# If the completion contains extra numbers (e.g. prints a subsequence / dp array),
# accept only an integer that is unambiguous.
_KEYWORD_PATTERNS = [
    re.compile(r"(?:\bLIS\b\s*)?(?:length|len)\s*(?:is|=|:)\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"\banswer\b\s*(?:is|=|:)\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"\bfinal\b\s*\banswer\b\s*(?:is|=|:)\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"\bответ\b\s*(?:это|=|:)\s*(-?\d+)", re.IGNORECASE),
    re.compile(r"\bдлина\b\s*(?:это|=|:)\s*(-?\d+)", re.IGNORECASE),
]


def _extract_single_int(text: str) -> Optional[int]:
    """Return a single, unambiguous integer from a text chunk.

    - If the chunk is exactly an integer => OK
    - If it contains an explicitly marked integer ("length=", "answer:") => OK
    - Otherwise, strip bracketed lists like [1,2,3] and accept only if exactly one integer remains.

    If multiple integers remain and there is no explicit marker, returns None.
    """
    if text is None:
        return None
    cand = str(text).strip()
    if not cand:
        return None

    if _ONLY_INT_RE.fullmatch(cand):
        try:
            return int(cand)
        except Exception:
            return None

    # Pure list/array inside <answer> should not be treated as a scalar answer.
    if cand.startswith("[") and cand.endswith("]"):
        return None

    for pat in _KEYWORD_PATTERNS:
        m = pat.search(cand)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None

    no_lists = _BRACKETED_RE.sub(" ", cand)
    lines = [ln.strip() for ln in no_lists.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if _ONLY_INT_RE.fullmatch(ln):
            try:
                return int(ln)
            except Exception:
                return None

    ints = _INT_RE.findall(no_lists)
    if len(ints) == 1:
        try:
            return int(ints[0])
        except Exception:
            return None
    return None


def extract_int(text: Optional[str]) -> Optional[int]:
    """Extracts an integer answer from model completion.

    Priority:
    1) unambiguous integer inside <answer>...</answer>
    2) unambiguous integer in the whole text
    """
    if text is None:
        return None

    m = _ANSWER_TAG_RE.search(text)
    if m:
        v = _extract_single_int(m.group(1))
        if v is not None:
            return v

    return _extract_single_int(text)


# -----------------------
# vLLM evaluation helpers
# -----------------------
def eval_all_testsets_with_llm(
    llm: Any,
    tokenizer: Any,
    testsets: Dict[str, Iterable[Dict[str, Any]]],
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    system_prompt: str = SYSTEM_PROMPT,
    stop: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate all provided testsets with a *single* vLLM engine.

    Args:
        llm: vllm.LLM instance
        tokenizer: HF tokenizer used to apply chat template
        testsets: dict(name -> iterable of rows with at least prompt + answer)
        max_new_tokens: generation length
        temperature: 0.0 => greedy
        system_prompt: must match training
        stop: optional stop strings for vLLM
        verbose: prints per-dataset accuracy if True
    """
    # Lazy import: keeps this module importable without vLLM installed.
    from vllm import SamplingParams

    sampling_kwargs: Dict[str, Any] = dict(
        temperature=float(temperature),
        max_tokens=int(max_new_tokens),
        top_p=1.0,
    )
    if stop:
        sampling_kwargs["stop"] = list(stop)

    sampling = SamplingParams(**sampling_kwargs)

    scores: Dict[str, float] = {}
    for name, ds in testsets.items():
        prompts: List[str] = []
        golds: List[int] = []

        for ex in ds:
            user_prompt = get_user_prompt(ex)
            prompts.append(build_chat_prompt(tokenizer, user_prompt, system_prompt=system_prompt))
            try:
                golds.append(int(ex["answer"]))
            except Exception:
                # keep alignment with prompts list
                golds.append(None)  # type: ignore

        outs = llm.generate(prompts, sampling)

        correct = 0
        total = 0
        for o, gold in zip(outs, golds):
            if gold is None:
                continue
            text = o.outputs[0].text
            pred = extract_int(text)

            with open("log-train.txt", "a", encoding="utf-8") as f:
                f.write(text + "\n")
                f.write("pred="+ str(pred) + "\n")

            correct += int(pred is not None and int(pred) == int(gold))
            total += 1

        acc = correct / max(1, total)
        scores[name] = acc
        if verbose:
            print(f"[{name}] acc={acc:.4f} (n={total})")

    return scores


# -----------------------
# Plotting
# -----------------------
def plot_paired_bars_accuracy(
    baseline_scores: Dict[str, float],
    trained_scores: Dict[str, float],
    *,
    out_path: JsonLikePath,
    title: str = "Accuracy: baseline vs trained",
) -> None:
    """Save a paired bar chart (baseline vs trained) to a PNG file."""
    import matplotlib.pyplot as plt

    names = list(baseline_scores.keys())
    # Keep stable order: if trained has same keys - prefer names from baseline.
    base_vals = [float(baseline_scores[n]) for n in names]
    tr_vals = [float(trained_scores.get(n, 0.0)) for n in names]

    x = list(range(len(names)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="baseline")
    ax.bar([i + width / 2 for i in x], tr_vals, width, label="trained")

    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200)
    plt.close(fig)
