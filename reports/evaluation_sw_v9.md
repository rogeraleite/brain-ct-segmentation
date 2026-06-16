# Evaluation Report - sw_v9

Generated: `2026-06-15T19:28:52`

## Setup

| Field | Value |
|---|---|
| Model | `sw_v9` |
| Description | Sliding-window v9: SAM skull masks at training, largest-CC at inference. |
| Checkpoint | `models/best_model_slidingWindow_v9_dice.pth` |
| Checkpoint epoch | 130 |
| Checkpoint val Dice | 0.2237 |
| Checkpoint val loss | 0.9283 |
| Threshold | 0.30 |
| Stride | 64 |
| Device | `mps` |
| Cases | 82 |

## Aggregate Metrics

| metric | mean | median | min | max |
| --- | --- | --- | --- | --- |
| dice | 0.0362 | 0.0000 | 0.0000 | 0.4332 |
| iou | 0.0207 | 0.0000 | 0.0000 | 0.2765 |
| precision | 0.0219 | 0.0000 | 0.0000 | 0.3051 |
| recall | 0.7799 | 1.0000 | 0.0000 | 1.0000 |
| specificity | 0.9766 | 0.9802 | 0.8705 | 0.9942 |
| abs_volume_error_ml | 1456.4409 | 1243.8925 | 326.9600 | 7098.5200 |

## Worst Dice Cases

| case_id | dice | recall | precision | true_volume_ml | pred_volume_ml | abs_volume_error_ml |
| --- | --- | --- | --- | --- | --- | --- |
| 054.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1268.5500 | 1268.5500 |
| 055.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1774.3600 | 1774.3600 |
| 056.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 845.7100 | 845.7100 |
| 057.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1187.0700 | 1187.0700 |
| 058.nii.gz | 0.0000 | 0.0000 | 0.0000 | 2.2650 | 1424.6250 | 1422.3600 |

## Lowest Recall Cases

| case_id | dice | recall | precision | fn_voxels | true_volume_ml | pred_volume_ml |
| --- | --- | --- | --- | --- | --- | --- |
| 058.nii.gz | 0.0000 | 0.0000 | 0.0000 | 453 | 2.2650 | 1424.6250 |
| 067.nii.gz | 0.0000 | 0.0000 | 0.0000 | 1086 | 5.4300 | 434.2250 |
| 079.nii.gz | 0.0000 | 0.0000 | 0.0000 | 266 | 1.3300 | 479.5450 |
| 082.nii.gz | 0.0002 | 0.0120 | 0.0001 | 3125 | 15.8150 | 1770.0200 |
| 086.nii.gz | 0.0018 | 0.0133 | 0.0010 | 5057 | 25.6250 | 352.5850 |

## Largest Volume Errors

| case_id | dice | recall | precision | true_volume_ml | pred_volume_ml | abs_volume_error_ml |
| --- | --- | --- | --- | --- | --- | --- |
| 050.nii.gz | 0.0132 | 0.8437 | 0.0066 | 56.3250 | 7154.8450 | 7098.5200 |
| 085.nii.gz | 0.0040 | 0.7188 | 0.0020 | 12.0750 | 4285.9300 | 4273.8550 |
| 124.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 3214.8250 | 3214.8250 |
| 123.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 3127.5350 | 3127.5350 |
| 069.nii.gz | 0.0013 | 0.4690 | 0.0006 | 3.8700 | 2894.9100 | 2891.0400 |

## Notes

- Dice and IoU measure mask overlap, but they can hide clinically important false negatives on small lesions.
- Recall/sensitivity is especially important for screening-style stroke workflows because missed lesions are costly.
- Specificity is voxel-level and often very high because background dominates CT volumes; interpret it alongside false-positive voxels and visual review.
- This is an internal validation report on local data, not evidence of clinical readiness.
