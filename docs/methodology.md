# Method and data specification

## Task definition

The model receives an RGB frame and a task prompt. Depending on the task, the prompt requests scene description, obstacle attributes, object grounding, risk analysis, free-space analysis, or uncertainty awareness. State-conditioned prompts additionally contain a seven-value vector.

## State schema

The training order is fixed:

| Index | Name | Unit |
| ---: | --- | --- |
| 0 | `x` | m |
| 1 | `y` | m |
| 2 | `d_goal` | m |
| 3 | `d_front` | m |
| 4 | `d_left` | m |
| 5 | `d_back` | m |
| 6 | `d_right` | m |

Risk prompts may also include `yaw_rad`; it is a scalar prompt field and is not part of the seven-value vector. Image and state timestamps must refer to the same frame.

## Split policy

The preferred split unit is the trial. This prevents near-duplicate frames from one flight appearing in both training and test sets. For legacy data with no usable split field, `prepare_dataset.py` uses contiguous chronological frame groups and records the method in `dataset_summary.json`.

## Label policy

Obstacle boxes proposed by HSV heuristics are pseudo-labels until human review. Risk and direction references generated from geometry are annotation-policy labels. The paper must distinguish visual-task performance, label agreement, and real-world safety evidence.
