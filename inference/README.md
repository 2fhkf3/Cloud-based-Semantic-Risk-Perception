# Inference and evaluation

## Single-image inference

`infer.py` loads the base model once, optionally attaches a LoRA adapter, formats one image and prompt, and prints one generated answer.

## JSONL inference

`infer_jsonl.py` loads the model once and processes records sequentially. Each output preserves the input record and adds `record_index`, `prediction`, `status`, or an error message. This makes malformed records auditable instead of silently dropping them.

## State-conditioned inference

`build_state_inference.py` joins a task corpus with exactly seven state values per `(trial_id, frame_index)`. It writes the state names and the formatted state text into each prompt. The state order is fixed and documented in `docs/methodology.md`.

## Evaluation

`compare_pilot_predictions.py` compares generated answers against reference labels for the pilot experiment. All evaluation scripts must state whether a score measures exact task-label agreement, JSON validity, retrieval/recall, or agreement with a geometric heuristic. Do not call heuristic-label agreement flight-safety validation.
