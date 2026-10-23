# Experiment protocol and paper checklist

Use one record per experiment. The record can be a Markdown file under `artifacts/experiment_records/` (that directory is ignored) or an external versioned experiment tracker.

## A. Data

- Dataset name, release/version, source URL or access identifier.
- Number of raw trials and frames.
- Reasons and counts for excluded trials.
- Coordinate system, unit conversion, camera/frame synchronization method.
- Number of records and unique frames per task.
- Counts by train/validation/test trial, never only by row.
- Pseudo-label generation and human-review status.

## B. Training

- Repository commit and command line.
- Base model revision and tokenizer/processor revision.
- LoRA rank, alpha, target modules, dropout, and frozen modules.
- Image pixel limit, sequence length, batch size, gradient accumulation, optimizer, learning rate, schedule, warmup, epochs, seed.
- Hardware, number of devices, software versions, wall-clock time, and peak memory.
- Checkpoint selection rule fixed before looking at test results.

## C. Inference

- Checkpoint path and hash.
- Prompt template and state schema.
- Generation temperature, sampling setting, max tokens, and decoding implementation.
- Number of attempted records, successful records, malformed records, and generation errors.

## D. Metrics

- JSON validity and parse failure rate.
- Task-specific exact match/F1/IoU or recall definitions.
- Risk-label agreement and direction agreement against the reference policy.
- Confidence intervals or trial-level bootstrap when reporting aggregate metrics.
- Separate results for seen and unseen trials.
- Qualitative failure cases, especially ambiguous visibility and unit-sensitive state inputs.

## E. Claims

Use precise wording: “agreement with geometric reference labels” is acceptable. “Proves flight safety” or “validated collision avoidance” is not supported by the current labels and offline tests.
