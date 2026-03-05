# Week 2 — GRPO LIS (Инструкция по запуску)

Ниже — инструкция по воспроизведению решения для ДЗ «Неделя 2 / Избранные темы в ИИ» (обучение GRPO + финальная оценка на фиксированных тестах).

Репозиторий рассчитан на запуск **из корня папки `W2-1/`** (рядом лежат `.ipynb` и `.py` файлы).

---

## 1) Что делает решение

**Задача:** по входной последовательности целых чисел предсказать **длину** Longest Increasing Subsequence (LIS).

Пайплайн состоит из двух частей:

1) **Обучение (TRAIN)**: RL‑обучение базовой LLM методом **GRPO** (Unsloth + TRL) с curriculum по сложности.
2) **Оценка (EVAL)**: инференс через **vLLM** и расчёт accuracy на фиксированных test‑наборах для:
   - baseline модели (без обучения)
   - trained модели (после GRPO)

---

## 2) Структура проекта и важные файлы

Минимально важное (то, что нужно кураторам):

- **`week2_grpo_lis_train_v8_my.ipynb`** — основной ноутбук обучения (GRPO/curriculum)  
- **`week2_grpo_lis_eval_vllm_v6.ipynb`** — основной ноутбук оценки (vLLM: baseline vs trained)

Общие модули:

- `lis_env.py` — “среда” и верификатор (интерфейсы `Data/Env/Verifier` + генерация задач LIS)
- `prompt_templates.py` — генерация user‑prompt на **английском** для задачи LIS (по требованиям ДЗ)
- `w2_utils.py` — общий код для train/eval (SYSTEM_PROMPT, сборка chat‑prompt, jsonl‑утилиты, vLLM‑eval, график)

Скрипты/прочее:

- `publish_to_hf.py` — публикация обученной модели и сгенерированных датасетов на Hugging Face (опционально)
- `requirements-train.txt` — зависимости для окружения обучения
- `requirements-eval.txt` — зависимости для окружения оценки (vLLM)

Папки‑артефакты:

- `data/` — фиксированные датасеты (train/dev/test) в формате `.jsonl`
- `models/` — сохранённая **merged** модель (base + LoRA), готовая для инференса
- `results/` — конфиги, метаданные best‑модели, результаты eval (json + график)

---

## 3) Окружения (env) и требования

Рекомендовано держать **2 отдельных окружения**:

- **TRAIN‑env** (Unsloth + TRL/GRPO)  
- **EVAL‑env** (vLLM + Transformers)  

Причина: vLLM часто предъявляет жёсткие требования к версиям torch/CUDA и удобнее не смешивать с train‑стеком.

### Железо
- Для **обучения** нужен NVIDIA GPU (CUDA). Для Qwen2.5‑1.5B в 4‑битном режиме обычно достаточно 16–24 GB VRAM, но зависит от настроек (`num_generations`, `max_seq_length` и т.д.).
- Для **оценки** (vLLM) также рекомендован GPU.

---

## 4) Установка зависимостей

Ниже пример для Linux/macOS (venv). Аналогично можно использовать conda.

### 4.1 TRAIN‑env
```bash
cd W2-1

python -m venv .venv-train
source .venv-train/bin/activate
pip install -U pip

pip install -r requirements-train.txt
```

### 4.2 EVAL‑env
```bash
cd W2-1

python -m venv .venv-eval
source .venv-eval/bin/activate
pip install -U pip

pip install -r requirements-eval.txt
```

> Примечание: базовая модель (`Qwen/Qwen2.5-1.5B-Instruct`) скачивается с Hugging Face при первом запуске (если не закэширована локально).

---

## 5) Запуск обучения (TRAIN)

1) Активируйте окружение обучения:
```bash
cd W2-1
source .venv-train/bin/activate
```

2) Запустите Jupyter:
```bash
jupyter lab
# или: jupyter notebook
```

3) Откройте и выполните **Run All**:
- `week2_grpo_lis_train_v8_my.ipynb`

### 5.1 Где настраивать параметры
В ноутбуке есть единый объект конфигурации `cfg` (dataclass `ExperimentConfig`).
Ключевые параметры:
- `cfg.base_model` — базовая модель
- `cfg.curriculum_phases` — две фазы curriculum (d1–5 и d6–10), количество шагов и частота dev‑eval
- `cfg.load_in_4bit`, LoRA‑параметры
- флаги стабильности (`disable_*compile*`), чтобы избежать `torch.compile`/Dynamo проблем

### 5.2 Какие артефакты появляются после train
После успешного выполнения:
- **Фиксированные наборы задач**:
  - `data/test_*.jsonl` — *финальные тесты* (easy/medium/hard) для eval‑ноутбука
  - `data/dev_*.jsonl` — dev‑срезы для валидации во время RL
  - (опционально) `data/train_*.jsonl` — предсэмпленный train для каждой фазы
- **Модель**:
  - `models/qwen2p5_1p5b_grpo_lis_merged/` — merged веса (base + LoRA), готовые для inference
  - `results/trained_model.json` — путь к merged‑директории (используется eval‑ноутбуком)
- **Метаданные**:
  - `results/config.json` — конфиг эксперимента (для воспроизводимости)
  - `results/best_model_by_dev.json` — информация о лучшей точке по dev‑метрике
  - `results/phase_checkpoints/` — лучшие LoRA состояния по фазам (если включено сохранение)

### 5.3 Где смотреть метрики обучения
В логах ноутбука печатаются строки вида:
- `[DEV-QUICK @ step ... | phase...] ... dev/avg=...`
- `[DEV-FULL  @ step ... | phase...] ... dev/avg=...`
- `[BEST] dev/avg=... @ step=... (phase=...)`

`step` в этих строках — **абсолютный шаг по всем фазам** (поэтому в фазе2 первый eval может отображаться как `step 220`, если фаза1 была 200 шагов и eval каждые 20 шагов).

---

## 6) Запуск финальной оценки (EVAL)

1) Активируйте окружение оценки:
```bash
cd W2-1
source .venv-eval/bin/activate
```

2) Запустите Jupyter и выполните **Run All**:
- `week2_grpo_lis_eval_vllm_v6.ipynb`

### 6.1 Что делает eval‑ноутбук
- Загружает фиксированные тесты из `data/test_*.jsonl`
- Загружает путь к обученной модели из `results/trained_model.json`
- Если датасетов и модели не будет в локальной папке, ноутбук загрузих их с HF
- Запускает vLLM для:
  - baseline: `Qwen/Qwen2.5-1.5B-Instruct`
  - trained: `models/qwen2p5_1p5b_grpo_lis_merged/` (или другой путь из `trained_model.json`)
- Считает accuracy на easy/medium/hard

### 6.2 Результаты eval (куда сохраняются)
После выполнения появятся:
- `results/baseline_scores.json`
- `results/trained_scores.json`
- `results/paired_bars_accuracy.png` — график baseline vs trained

В ноутбуке также печатается summary с дельтами (`trained - baseline`).

### 6.3 Если не хватает GPU памяти в eval
В eval‑ноутбуке есть параметр `gpu_memory_utilization` (по умолчанию ~0.3).  
Если ловите OOM — уменьшите значение или раскомментируйте блок, который удаляет baseline‑движок перед загрузкой trained.

---

## 7) (Опционально) Публикация на Hugging Face

Скрипт `publish_to_hf.py` публикует:
- merged модель (из `results/trained_model.json` или `models/...`)
- датасеты `data/test_*.jsonl` и `data/dev_*.jsonl` (и опционально `train_*.jsonl`)

Пример:
```bash
cd W2-1
source .venv-eval/bin/activate  # или любое env где есть huggingface_hub

huggingface-cli login
python publish_to_hf.py \
  --model_repo <username>/<repo-model> \
  --dataset_repo <username>/<repo-dataset> \
  --private
```

---

## 8) Контрольный список для проверки 

1) После TRAIN:
- существует `results/trained_model.json`
- существует папка `models/qwen2p5_1p5b_grpo_lis_merged/`
- в `data/` есть `test_easy...jsonl`, `test_medium...jsonl`, `test_hard...jsonl`

2) После EVAL:
- `results/baseline_scores.json` и `results/trained_scores.json` созданы
- есть `results/paired_bars_accuracy.png`
- в консоли напечатан summary с accuracy и delta