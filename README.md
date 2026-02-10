# LLM Profiling for Learner Type Prediction (EDM 2026)

This repository contains code and experiment artifacts for an Educational Data Mining (EDM) 2026 submission/paper on **profiling student programming behavior** and **predicting learner types** from time-ordered code snapshots, engineered behavioral features, and (optionally) embeddings.

It includes:
- **LLM-based profiling**: prompts an LLM to assign *one* learner-type label to each `(user_id, exercise_id)` session.
- **Baseline ML models**:
  - **Non-sequential** aggregated models (tree ensembles over per-session aggregates).
  - **Sequential** models (TensorFlow/Keras BiLSTM over padded snapshot sequences).

## Repository layout

- `main.py`: end-to-end LLM profiling pipeline (data load -> prompt build -> LLM call -> caching/logging -> optional evaluation)
- `config.py`: paths, label set, provider/model configuration, prompt-size controls
- `data_loader.py`: loads/merges required data sources
- `prompt_builder.py`: builds per-session prompts (downsampling + feature blocks + code snapshots)
- `llm_client.py`: LLM provider adapters (OpenAI/DeepSeek/Together/Gemini/Anthropic)
- `evaluator.py`: evaluation against `data/final_labels.csv`
- `non_sequential_prediction_learner_types_*.py`: non-sequential baselines (requires embeddings parquet)
- `sequential_prediction_learner_types_tensorflow*.py`: sequential baselines (requires embeddings parquet)
- `data/README.md`: expected data files
- `output/`: generated predictions + caches (some example outputs are included)
- `logs/`: raw prompts/responses logs (may include student code; handle accordingly)

## Data

Place the following files in `data/` (see `data/README.md`):
- `all_features_df.csv`
- `thinned_snapshots_60s.csv`
- `rationale_from_llm.xlsx`
- `final_labels.csv` (ground truth labels used for evaluation)

Optional (for baseline models):
- `thinned_snapshots_60s_with_embeddings.parquet` (used by the baseline scripts; paths are set at the top of each script)

## Setup

Tested with Python 3.12 (see `.venv/pyvenv.cfg`). A Python 3.10+ environment is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure LLM access

Create a `.env` file in the repository root containing the API key for your chosen provider:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
TOGETHER_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

Then set `LLM_PROVIDER` and `LLM_MODEL` in `config.py`.

## Run: LLM profiling

```powershell
python main.py
```

Notes:
- `main.py` currently uses `limit_sessions=1` as a smoke test. Set `limit_sessions=None` in `main.py` to process all sessions.
- Outputs are written to `output/` and logs to `logs/`.

## Evaluate predictions

If `data/final_labels.csv` is present:

```powershell
python evaluator.py
```

This evaluates all `output/llm_profile_predictions*.csv` files (and applies the default held-out test students defined in `evaluator.py`).

## Baseline ML models (optional)

These scripts expect `thinned_snapshots_60s_with_embeddings.parquet` and `final_labels.csv` to be available at the paths configured near the top of each script (edit paths if you keep them under `data/`).

Non-sequential aggregated baselines:

```powershell
python non_sequential_prediction_learner_types_optimized.py
python non_sequential_prediction_learner_types_original.py
```

Sequential (BiLSTM) baselines:

```powershell
python sequential_prediction_learner_types_tensorflow.py
python sequential_prediction_learner_types_tensorflow_original.py
python sequential_prediction_learner_types_tensorflow_improved_features.py
python sequential_prediction_learner_types_tensorflow_improved_model.py
```

## Included results/artifacts

- `results_llms.txt`, `results_llm_testset_only.txt`
- `results_nonsequential_ml.txt`
- `results_sequential_model_BiLSTM.txt`
- `all_results.txt`
- `results_table_latex.txt` (LaTeX table used for paper write-up)

## Reproducibility notes

- LLM calls are cached by `(session_key, prompt_hash, provider, model)` in `output/llm_profile_cache*.csv`.
- Logs in `logs/` include raw prompts/responses for auditability; they may contain student code snapshots.
- Baseline models use student-level splitting to reduce leakage (see each script for details).

## Citation

If you use this repository, please cite the corresponding EDM 2026 paper (update the fields below):

```bibtex
@inproceedings{edm2026-learner-types,
  title     = {LLM Profiling for Learner Type Prediction},
  author    = {<Authors>},
  booktitle = {Proceedings of the Educational Data Mining (EDM) Conference},
  year      = {2026}
}
```
