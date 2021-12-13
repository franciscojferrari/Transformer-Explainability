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

[Link to download dataset](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat).

In the exmaple above we run a segmentation test with our method. Notice you can choose which method you wish to run using the `--method` argument. 
You must provide a path to imagenet segmentation data in `--imagenet-seg-path`.
