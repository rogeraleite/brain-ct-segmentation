# Evaluation Report - sw_v10

Generated: `2026-06-16T03:22:38`

## Setup

| Field | Value |
|---|---|
| Model | `sw_v10` |
| Description | Sliding-window v10: SAM skull masks + pos_weight=5 (down from 50). |
| Checkpoint | `models/best_model_slidingWindow_v10.pth` |
| Checkpoint epoch | 75 |
| Checkpoint val Dice | 0.2824 |
| Checkpoint val loss | 0.8066 |
| Threshold | 0.50 |
| Stride | 64 |
| Device | `mps` |
| Cases | 82 |

## Aggregate Metrics

| metric | mean | median | min | max |
| --- | --- | --- | --- | --- |
| dice | 0.0559 | 0.0000 | 0.0000 | 0.5869 |
| iou | 0.0348 | 0.0000 | 0.0000 | 0.4153 |
| precision | 0.0741 | 0.0000 | 0.0000 | 0.8126 |
| recall | 0.6432 | 1.0000 | 0.0000 | 1.0000 |
| specificity | 0.9981 | 0.9989 | 0.9585 | 1.0000 |
| abs_volume_error_ml | 126.2312 | 66.3075 | 0.1900 | 2256.4950 |

## Worst Dice Cases

| case_id | dice | recall | precision | true_volume_ml | pred_volume_ml | abs_volume_error_ml |
| --- | --- | --- | --- | --- | --- | --- |
| 054.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 51.9750 | 51.9750 |
| 055.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 39.1600 | 39.1600 |
| 056.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 107.7250 | 107.7250 |
| 057.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 98.8650 | 98.8650 |
| 059.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 7.8050 | 7.8050 |

## Lowest Recall Cases

| case_id | dice | recall | precision | fn_voxels | true_volume_ml | pred_volume_ml |
| --- | --- | --- | --- | --- | --- | --- |
| 074.nii.gz | 0.0000 | 0.0000 | 0.0000 | 11922 | 59.6100 | 13.0750 |
| 086.nii.gz | 0.0000 | 0.0000 | 0.0000 | 5125 | 25.6250 | 1.8500 |
| 070.nii.gz | 0.0001 | 0.0001 | 0.0013 | 63712 | 318.5850 | 18.9600 |
| 082.nii.gz | 0.0006 | 0.0019 | 0.0004 | 3157 | 15.8150 | 83.0600 |
| 068.nii.gz | 0.0083 | 0.0045 | 0.0524 | 21732 | 109.1550 | 9.4400 |

## Largest Volume Errors

| case_id | dice | recall | precision | true_volume_ml | pred_volume_ml | abs_volume_error_ml |
| --- | --- | --- | --- | --- | --- | --- |
| 050.nii.gz | 0.0308 | 0.6486 | 0.0158 | 56.3250 | 2312.8200 | 2256.4950 |
| 071.nii.gz | 0.2654 | 0.1742 | 0.5573 | 1034.9250 | 323.4650 | 711.4600 |
| 061.nii.gz | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 489.3400 | 489.3400 |
| 085.nii.gz | 0.0105 | 0.1872 | 0.0054 | 12.0750 | 419.3800 | 407.3050 |
| 075.nii.gz | 0.4950 | 0.3791 | 0.7128 | 771.0450 | 410.0950 | 360.9500 |

## Notes

- Dice and IoU measure mask overlap, but they can hide clinically important false negatives on small lesions.
- Recall/sensitivity is especially important for screening-style stroke workflows because missed lesions are costly.
- Specificity is voxel-level and often very high because background dominates CT volumes; interpret it alongside false-positive voxels and visual review.
- This is an internal validation report on local data, not evidence of clinical readiness.
