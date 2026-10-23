# ODA extraction

## Inputs

The dataset root must contain `trial_overview.csv` and one directory per trial. The overview may contain multiple rows for a trial when two obstacles are present; both rows are retained.

## Command

```bash
python pipeline/oda/extract_multimodal_batch.py \
  --dataset-root /path/to/ODA_full/dataset \
  --output-root /path/to/multimodal_output \
  --workers 32
```

Useful controls are `--trial-ids`, `--limit`, `--sample-every`, `--overwrite`, and `--list-only`.

## Outputs

Each successful trial contains decoded RGB frames and aligned annotations. Each batch creates `batch_logs/run_*/manifest.csv` with `ok`, `skipped`, or `failed` status and a reason. Report the number of input, runnable, skipped, failed, and successful trials in the paper.

## Coordinate and label semantics

The horizontal ODA plane uses raw OptiTrack X/Z coordinates. Yaw is derived from the configured body-forward axis and initial rotation offset. Four-direction distances are geometric raycast labels derived from obstacle and wall geometry; they are not necessarily sensor measurements. Validate units before training state-conditioned tasks.
