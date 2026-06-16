# Evaluation Report - sw_v8

Generated: `2026-06-15T11:46:00`

## Setup

| Field | Value |
|---|---|
| Model | `sw_v8` |
| Description | Sliding-window v8 dice checkpoint, skull_excl 3mm (sem skull_strip — incompatível com dados brain-windowed). |
| Checkpoint | `models/best_model_slidingWindow_v8_dice.pth` |
| Checkpoint epoch | 132 |
| Checkpoint val Dice | 0.3774 |
| Checkpoint val loss | 0.9252 |
| Threshold | 0.30 |
| Stride | 64 |
| Device | `mps` |
| Cases | 82 |

## Aggregate Metrics

| metric | mean | median | min | max |
| --- | --- | --- | --- | --- |
| dice | 0.0033 | 0.0000 | 0.0000 | 0.0733 |
| iou | 0.0017 | 0.0000 | 0.0000 | 0.0381 |
| precision | 0.0018 | 0.0000 | 0.0000 | 0.0412 |
| recall | 0.6198 | 1.0000 | 0.0000 | 1.0000 |
| specificity | 0.8979 | 0.8983 | 0.8539 | 0.9540 |
| abs_volume_error_ml | 6528.3195 | 6546.5250 | 1831.8600 | 9909.8200 |

## Worst Dice Cases

| case_id | dice | recall | precision | true_volume_ml | pred_volume_ml | abs_volume_error_ml |
| --- | --- | --- | --- | --- | --- | --- |
| 054.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 8209.3600 | 8209.3600 |
| 055.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 6115.9900 | 6115.9900 |
| 056.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 5836.4900 | 5836.4900 |
| 057.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 7874.3800 | 7874.3800 |
| 058.nii.gz | 0.0000 | 0.0000 | 0.0000 | 2.2650 | 7166.7150 | 7164.4500 |

## Lowest Recall Cases

| case_id | dice | recall | precision | fn_voxels | true_volume_ml | pred_volume_ml |
| --- | --- | --- | --- | --- | --- | --- |
| 058.nii.gz | 0.0000 | 0.0000 | 0.0000 | 453 | 2.2650 | 7166.7150 |
| 069.nii.gz | 0.0000 | 0.0000 | 0.0000 | 774 | 3.8700 | 9913.6900 |
| 072.nii.gz | 0.0000 | 0.0000 | 0.0000 | 883 | 4.4150 | 7779.9600 |
| 082.nii.gz | 0.0000 | 0.0000 | 0.0000 | 3163 | 15.8150 | 6470.5050 |
| 085.nii.gz | 0.0000 | 0.0000 | 0.0000 | 2415 | 12.0750 | 1843.9350 |

## Largest Volume Errors

| case_id | dice | recall | precision | true_volume_ml | pred_volume_ml | abs_volume_error_ml |
| --- | --- | --- | --- | --- | --- | --- |
| 069.nii.gz | 0.0000 | 0.0000 | 0.0000 | 3.8700 | 9913.6900 | 9909.8200 |
| 101.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 9564.8650 | 9564.8650 |
| 126.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 9147.1800 | 9147.1800 |
| 115.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 9132.1750 | 9132.1750 |
| 054.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 8209.3600 | 8209.3600 |

## Notes

- Dice and IoU measure mask overlap, but they can hide clinically important false negatives on small lesions.
- Recall/sensitivity is especially important for screening-style stroke workflows because missed lesions are costly.
- Specificity is voxel-level and often very high because background dominates CT volumes; interpret it alongside false-positive voxels and visual review.
- This is an internal validation report on local data, not evidence of clinical readiness.
