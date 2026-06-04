# MedSAM2 3D Segmentation - Bidirectional Propagation

This repository demonstrates the MedSAM2 3D segmentation pipeline modified to properly execute both **forward and backward** propagation in volumetric medical images (like MRI/CT scans). 

## Overview
By default, placing a prompt (bounding box) on a key slice in a 3D medical scan using MedSAM2 often propagates only forward. This results in missing segmentations for any slices preceding the key slice.

We modified the core inference loop within the `LLD_MMRI_MedSAM2.ipynb` notebook to run the `propagate_in_video` function bidirectionally:

```python
# 1. Forward Propagation
for f_idx, _, l in predictor.propagate_in_video(state):
    segs[f_idx] = (l[0] > 0.0).cpu().numpy()
                    
# 2. Backward Propagation (Added)
for f_idx, _, l in predictor.propagate_in_video(state, reverse=True):
    segs[f_idx] = (l[0] > 0.0).cpu().numpy()
```

## Included Files
- `LLD_MMRI_MedSAM2.ipynb`: The primary notebook, updated with the bidirectional propagation fix.
- `MR-391135_1_C+A_0000.nii`: The original input MRI scan.
- `MR-391135_1_C+A_0000_k36_mask.nii`: The resulting generated 3D mask (label image).
- `MR-391135_1_C+A_0000_comparison.png`: A visualization comparing the raw image, ground truth, and the MedSAM2 generated mask, including the Dice score.

## Results
With backward propagation enabled, the segmentation successfully spans the entire lesion in 3D, improving the overall Dice score significantly compared to unidirectional propagation.
