#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publish trained model + generated datasets to Hugging Face Hub.

Этот скрипт рассчитан на артефакты, которые появляются после выполнения
week2_grpo_lis_train_v7_my.ipynb:

- merged модель (по умолчанию: models/qwen2p5_1p5b_grpo_lis_merged)
  либо путь из results/trained_model.json
- фиксированные наборы для оценки: data/test_*.jsonl
- (опционально) dev: data/dev_*.jsonl
- (опционально) предсэмпленный train: data/train_*.jsonl

Пример запуска:

  # 1) Авторизация (один раз)
  huggingface-cli login
  # или:
  export HF_TOKEN=hf_xxx

  # 2) Публикация
  python publish_to_hf.py \
    --model_repo <username>/<repo-model> \
    --dataset_repo <username>/<repo-dataset> \
    --private

Зависимости:
  pip install -U huggingface_hub datasets

Примечания:
- Для ускорения загрузок можно поставить:
    pip install -U "huggingface_hub[hf_transfer]"
  и экспортировать:
    export HF_HUB_ENABLE_HF_TRANSFER=1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ДОЛЖЕН совпадать с train/eval ноутбуками
SYSTEM_PROMPT = (
    "Отвечай в следующем формате:\n"
    "<think>\n"
    "...\n"
    "</think>\n"
    "<answer>\n"
    "...\n"
    "</answer>"
)


@dataclass
class DatasetFileInfo:
    kind: str          # "test" | "dev" | "train"
    name: str          # "easy" | "medium" | "hard" | "single" | ...
    difficulty: Optional[int]
    n: Optional[int]
    seed: Optional[int]
    path: Path


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_extract_spec_from_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """
    Пытаемся распарсить имя, difficulty, n, seed из имени файла вида:
      test_easy_d2_n200_seed1001.jsonl
      dev_medium_d5_n128_seed4002.jsonl
      train_single_d1-10_n2000_seed1234.jsonl  (difficulty здесь неоднозначен)
    """
    # test/dev pattern
    m = re.match(r"^(test|dev)_(?P<name>[a-zA-Z0-9\-]+)_d(?P<d>\d+)_n(?P<n>\d+)_seed(?P<seed>\d+)\.jsonl$", filename)
    if m:
        return m.group("name"), int(m.group("d")), int(m.group("n")), int(m.group("seed"))

    # train pattern (часто dmin-dmax), difficulty оставим None
    m = re.match(r"^train_(?P<name>[a-zA-Z0-9\-]+)_d(?P<dmin>\d+)-(?P<dmax>\d+)_n(?P<n>\d+)_seed(?P<seed>\d+).*\.jsonl$", filename)
    if m:
        return m.group("name"), None, int(m.group("n")), int(m.group("seed"))

    return None, None, None, None


def discover_model_dir(
    model_dir: Optional[str],
    results_dir: str,
    fallback_models_dir: str = "models/qwen2p5_1p5b_grpo_lis_merged",
) -> Path:
    """
    1) Если передан --model_dir -> используем его
    2) Иначе пробуем results/trained_model.json (создаётся train-ноутбуком)
    3) Иначе fallback: models/qwen2p5_1p5b_grpo_lis_merged
    """
    if model_dir:
        p = Path(model_dir).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--model_dir не найден: {p}")
        return p

    trained_model_json = Path(results_dir) / "trained_model.json"
    if trained_model_json.exists():
        info = _read_json(trained_model_json)
        td = info.get("trained_model_dir")
        if td:
            p = Path(td).expanduser().resolve()
            if p.exists():
                return p

    p = Path(fallback_models_dir).expanduser().resolve()
    if p.exists():
        return p

    raise FileNotFoundError(
        "Не удалось найти директорию модели.\n"
        f"Пробовал:\n"
        f"  - --model_dir (не задан)\n"
        f"  - {trained_model_json} (или путь внутри него)\n"
        f"  - {p}\n"
        "Укажите --model_dir явно."
    )


def discover_dataset_files(data_dir: str, include_train: bool) -> List[DatasetFileInfo]:
    """
    Ищем jsonl, которые генерит train-ноутбук:
      data/test_*.jsonl, data/dev_*.jsonl, (опционально) data/train_*.jsonl
    """
    d = Path(data_dir).expanduser().resolve()
    if not d.exists():
        raise FileNotFoundError(f"data_dir не найден: {d}")

    out: List[DatasetFileInfo] = []

    for kind in ("test", "dev"):
        for p in sorted(d.glob(f"{kind}_*.jsonl")):
            name, diff, n, seed = _maybe_extract_spec_from_filename(p.name)
            out.append(DatasetFileInfo(kind=kind, name=name or p.stem, difficulty=diff, n=n, seed=seed, path=p))

    if include_train:
        for p in sorted(d.glob("train_*.jsonl")):
            name, diff, n, seed = _maybe_extract_spec_from_filename(p.name)
            out.append(DatasetFileInfo(kind="train", name=name or p.stem, difficulty=diff, n=n, seed=seed, path=p))

    # sanity: обязательно должны быть test файлы
    if not any(x.kind == "test" for x in out):
        raise FileNotFoundError(
            f"В {d} не найдено ни одного файла test_*.jsonl.\n"
            "Сначала выполните train-ноутбук (часть с фиксированными тестсетами) "
            "или укажите корректный --data_dir."
        )

    return out


def _make_model_card(model_repo: str, base_model: str, dataset_repo: str) -> str:
    return f"""---
language: ru
tags:
- reinforcement-learning
- grpo
- qwen2.5
- lis
license: other
base_model: {base_model}
---

# GRPO LIS agent (Week 2)

Это модель, дообученная с помощью GRPO (RL) на среде **Longest Increasing Subsequence (LIS)**:
по заданной последовательности целых чисел нужно вернуть длину LIS.

## Важно про формат ответа

Системный промпт (должен совпадать с train/eval):

```text
{SYSTEM_PROMPT}
```

## Датасеты для оценки

Фиксированные test/dev наборы, сгенерированные в train-ноутбуке, опубликованы тут:
- `{dataset_repo}`

## Быстрый пример инференса (Transformers)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

repo = "{model_repo}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="auto")

user_prompt = "..."  # вопрос из датасета (одна задача)
messages = [
    {{"role": "system", "content": {SYSTEM_PROMPT!r}}},
    {{"role": "user", "content": user_prompt}},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```
"""


def _make_dataset_card(dataset_repo: str, file_infos: List[DatasetFileInfo]) -> str:
    lines = [
        "---",
        "language: ru",
        "tags:",
        "- reinforcement-learning",
        "- grpo",
        "- lis",
        "license: other",
        "---",
        "",
        "# LIS fixed datasets (Week 2)",
        "",
        "В этом репозитории лежат **фиксированные** датасеты (jsonl), которые генерируются в train-ноутбуке.",
        "Они нужны для воспроизводимого сравнения baseline vs trained.",
        "",
        "## Формат примера",
        "",
        "Каждая строка JSONL имеет вид:",
        "",
        "```json",
        '{"prompt": "...", "answer": "..."}',
        "```",
        "",
        "где `prompt` — это **user prompt** (условие задачи), а `answer` — эталонная длина LIS.",
        "",
        "## System prompt для инференса",
        "",
        "При оценке (см. eval-ноутбук) используется системный промпт:",
        "",
        "```text",
        SYSTEM_PROMPT,
        "```",
        "",
        "## Файлы",
        "",
    ]

    # сгруппируем по kind
    by_kind: Dict[str, List[DatasetFileInfo]] = {}
    for fi in file_infos:
        by_kind.setdefault(fi.kind, []).append(fi)

    for kind in ("test", "dev", "train"):
        if kind not in by_kind:
            continue
        lines.append(f"### {kind}")
        for fi in by_kind[kind]:
            meta = []
            if fi.difficulty is not None:
                meta.append(f"difficulty={fi.difficulty}")
            if fi.n is not None:
                meta.append(f"n={fi.n}")
            if fi.seed is not None:
                meta.append(f"seed={fi.seed}")
            meta_str = (", ".join(meta)) if meta else ""
            lines.append(f"- `{kind}/{fi.path.name}` ({meta_str})")
        lines.append("")

    lines.append("## Как загрузить через 🤗 Datasets")
    lines.append("")
    lines.append("Пример (подставьте свои имена файлов):")
    lines.append("")
    lines.append("```python")
    lines.append("from datasets import load_dataset")
    lines.append(f'repo = "{dataset_repo}"')
    lines.append('data_files = {')
    lines.append('  "test_easy": "test/test_easy_d2_n200_seed1001.jsonl",')
    lines.append('  "test_medium": "test/test_medium_d5_n200_seed2001.jsonl",')
    lines.append('  "test_hard": "test/test_hard_d8_n200_seed3001.jsonl",')
    lines.append('}')
    lines.append('ds = load_dataset(repo, data_files=data_files, split="test_easy")')
    lines.append("print(ds[0])")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _make_dataset_index(file_infos: List[DatasetFileInfo]) -> dict:
    idx: Dict[str, dict] = {
        "schema": {"prompt": "string", "answer": "string"},
        "system_prompt": SYSTEM_PROMPT,
        "files": {"test": {}, "dev": {}, "train": {}},
    }
    for fi in file_infos:
        entry = {
            "path_in_repo": f"{fi.kind}/{fi.path.name}",
            "filename": fi.path.name,
            "difficulty": fi.difficulty,
            "n": fi.n,
            "seed": fi.seed,
        }
        if fi.kind not in idx["files"]:
            idx["files"][fi.kind] = {}
        idx["files"][fi.kind][fi.name] = entry
    return idx


def _require_hf_libs():
    try:
        import huggingface_hub  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Не найден пакет huggingface_hub. Установите:\n"
            "  pip install -U huggingface_hub datasets\n"
        ) from e


def _get_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    for env_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(env_name)
        if v:
            return v
    return None


def upload_model_to_hub(
    model_repo: str,
    model_dir: Path,
    token: str,
    private: bool,
    dataset_repo: str,
    base_model: str,
    dry_run: bool,
):
    _require_hf_libs()
    from huggingface_hub import HfApi

    try:
        from huggingface_hub import upload_folder  # type: ignore
    except Exception:
        upload_folder = None  # type: ignore


    api = HfApi(token=token)

    if dry_run:
        print(f"[DRY RUN] Would create model repo: {model_repo} (private={private})")
    else:
        api.create_repo(repo_id=model_repo, repo_type="model", private=private, exist_ok=True)

    # Upload model folder
    print(f"[MODEL] Uploading folder: {model_dir} -> {model_repo}")
    if dry_run:
        # list a few files for visibility
        files = sorted([p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file()])
        print(f"[DRY RUN] {len(files)} files. First 20:")
        for f in files[:20]:
            print("  -", f)
    else:
        if upload_folder is not None:
            upload_folder(
                repo_id=model_repo,
                folder_path=str(model_dir),
                repo_type="model",
                token=token,
                commit_message="Upload trained merged model",
            )
        else:
            # Fallback: если upload_folder недоступен в вашей версии huggingface_hub,
            # грузим файлы по одному.
            for fp in sorted(model_dir.rglob("*")):
                if not fp.is_file():
                    continue
                rel = fp.relative_to(model_dir).as_posix()
                api.upload_file(
                    path_or_fileobj=str(fp),
                    path_in_repo=rel,
                    repo_id=model_repo,
                    repo_type="model",
                    token=token,
                    commit_message=f"Upload {rel}",
                )

    # Upload/overwrite README.md
    card_text = _make_model_card(
        model_repo=model_repo,
        base_model=base_model,
        dataset_repo=dataset_repo,
    )

    if dry_run:
        print("[DRY RUN] Would upload model README.md")
    else:
        tmp = Path(".") / "_tmp_model_README.md"
        tmp.write_text(card_text, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo="README.md",
            repo_id=model_repo,
            repo_type="model",
            token=token,
            commit_message="Add/Update model card",
        )
        tmp.unlink(missing_ok=True)

    print(f"[MODEL] Done: https://huggingface.co/{model_repo}")


def upload_datasets_to_hub(
    dataset_repo: str,
    file_infos: List[DatasetFileInfo],
    token: str,
    private: bool,
    dry_run: bool,
):
    _require_hf_libs()
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    if dry_run:
        print(f"[DRY RUN] Would create dataset repo: {dataset_repo} (private={private})")
    else:
        api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=private, exist_ok=True)

    # Upload dataset files under kind/filename
    print(f"[DATASET] Uploading {len(file_infos)} files -> {dataset_repo}")
    for fi in file_infos:
        path_in_repo = f"{fi.kind}/{fi.path.name}"
        if dry_run:
            print(f"[DRY RUN] Would upload {fi.path} -> {path_in_repo}")
            continue
        api.upload_file(
            path_or_fileobj=str(fi.path),
            path_in_repo=path_in_repo,
            repo_id=dataset_repo,
            repo_type="dataset",
            token=token,
            commit_message=f"Upload {fi.kind} file {fi.path.name}",
        )

    # Upload dataset index
    ds_index = _make_dataset_index(file_infos)
    if dry_run:
        print("[DRY RUN] Would upload dataset_index.json and README.md")
    else:
        tmp_idx = Path(".") / "_tmp_dataset_index.json"
        tmp_idx.write_text(json.dumps(ds_index, ensure_ascii=False, indent=2), encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(tmp_idx),
            path_in_repo="dataset_index.json",
            repo_id=dataset_repo,
            repo_type="dataset",
            token=token,
            commit_message="Add dataset index",
        )
        tmp_idx.unlink(missing_ok=True)

        # Upload README.md
        card_text = _make_dataset_card(dataset_repo=dataset_repo, file_infos=file_infos)
        tmp = Path(".") / "_tmp_dataset_README.md"
        tmp.write_text(card_text, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo="README.md",
            repo_id=dataset_repo,
            repo_type="dataset",
            token=token,
            commit_message="Add/Update dataset card",
        )
        tmp.unlink(missing_ok=True)

    print(f"[DATASET] Done: https://huggingface.co/datasets/{dataset_repo}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish model + datasets to Hugging Face Hub")
    p.add_argument("--model_repo", type=str, required=False, help="HF repo_id for model, e.g. username/my-model")
    p.add_argument("--dataset_repo", type=str, required=False, help="HF repo_id for dataset, e.g. username/my-datasets")

    p.add_argument("--model_dir", type=str, default=None, help="Path to merged model dir. If omitted, auto-detect.")
    p.add_argument("--data_dir", type=str, default="data", help="Directory with generated jsonl datasets (default: data)")
    p.add_argument("--results_dir", type=str, default="results", help="Directory with trained_model.json (default: results)")

    p.add_argument("--include_train", action="store_true", help="Also upload data/train_*.jsonl")
    p.add_argument("--private", action="store_true", help="Create repos as private")
    p.add_argument("--token", type=str, default=None, help="HF token (or use env HF_TOKEN / huggingface-cli login)")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name for README")
    p.add_argument("--dry_run", action="store_true", help="Do not upload, only print what would be done")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    token = _get_token(args.token)
    if not token and not args.dry_run:
        print(
            "[WARN] HF токен не передан через --token/ENV. "
            "Попробую использовать токен из `huggingface-cli login` (если вы залогинены).",
            file=sys.stderr,
        )

    model_dir = discover_model_dir(args.model_dir, results_dir=args.results_dir)
   

    print("[INFO] Model dir:", model_dir)

    upload_model_to_hub(
        model_repo=args.model_repo,
        model_dir=model_dir,
        token=token,
        private=bool(args.private),
        dataset_repo=args.dataset_repo,
        base_model=args.base_model,
        dry_run=bool(args.dry_run),
    )

    file_infos = discover_dataset_files(args.data_dir, include_train=args.include_train)
    if file_infos:
        print("[INFO] Dataset files:")
        for fi in file_infos:
            print(f"  - {fi.kind:5s} {fi.path.name}")

        # Upload datasets first (чтобы ссылка на датасет была в model card)
        upload_datasets_to_hub(
            dataset_repo=args.dataset_repo,
            file_infos=file_infos,
            token=token,
            private=bool(args.private),
            dry_run=bool(args.dry_run),
        )

    print("[OK] All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
