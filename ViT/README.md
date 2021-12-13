# ViT Transformer-Explainability

## Reproducing results on ViT

### Section 1. Perturbation Results

Generate the heatmaps:
```
python ViT/generate_heatmaps.py --method transformer_attribution --work-path /path/to/working_directory
```

Now to run the perturbation test run the following command:
```
python ViT/pertubation_eval_from_hdf5.py --method transformer_attribution --work-path /path/to/working_directory
```

### Section 2. Segmentation Results

Example:
```
python ViT/imagenet_seg_eval.py --method transformer_attribution --imagenet-seg-path /path/to/gtsegs_ijcv.mat

```