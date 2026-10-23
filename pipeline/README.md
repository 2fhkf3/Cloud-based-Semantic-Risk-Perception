# Pipeline

This directory implements the transformation from raw multimodal flight data to the supervised instruction dataset consumed by MS-SWIFT.

## Stage 1: Extraction and alignment

`oda/extract_multimodal_batch.py` is the batch driver. It reads `trial_overview.csv`, filters trials with missing video or obstacle coordinates, launches workers, and writes a manifest. `oda/extract_multimodal_dataset.py` performs one trial's work: video decoding, timestamp alignment, pose interpolation, yaw calculation, obstacle geometry, and per-frame metadata.

The output is an intermediate multimodal directory, not the final training JSONL. Preserve its manifest because it is the evidence for the dataset filtering table in a paper.

## Stage 2: Instruction corpus

The corpus builder converts each valid frame and task definition into a JSONL record containing an image path, a user prompt, an assistant reference answer, trial/frame identifiers, and label quality. The corpus builder is dataset-side code and is documented as an external preprocessing dependency in the main README.

## Stage 3: Validation and split

`prepare_dataset.py` checks every record and writes MS-SWIFT-compatible messages. It keeps all instructions belonging to one image together. Existing trial-level splits are trusted after consistency checks; legacy single-trial data is split into chronological frame blocks. The generated `dataset_summary.json` is the source of sample counts in the paper.

Never hand-edit generated JSONL. Change the source data or command-line options and regenerate it.
