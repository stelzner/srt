# SRT: Scene Representation Transformer

This is an independent PyTorch implementation of SRT, as presented in the paper
["Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations"](https://srt-paper.github.io/) by Sajjadi et al., CVPR 2022.

The authors have kindly [reviewed](https://srt-paper.github.io/#code) this code and confirmed that it appears to match their results.
All credit for the model goes to the original authors.

**New:** This repo now also supports the
[improved version of SRT](https://github.com/stelzner/srt/tree/main/runs/msn/isrt)
discussed in Appendix A.4 of the
[OSRT paper](https://osrt-paper.github.io/). It yields higher reconstruction accuracy, uses fewer parameters, and runs faster. An example checkpoint is provided below.


<img src="https://drive.google.com/uc?id=1K1hX4jc50tVc1sCLacPT0p8SjWjkcRgv" alt="NMR Rotation" width="256"/><img src="https://drive.google.com/uc?id=1-yF6QD-v663beyGgevmt8MDuH2FByeuS" alt="MSN Rotation" width="256"/>

## Setup
After cloning the repository and creating a new conda environment, the following steps will get you started:

### Data
The code currently supports the following datasets. Simply download and place (or symlink) them in the data directory.

- The 3D datasets introduced by [ObSuRF](https://stelzner.github.io/obsurf/).
- The NMR multiclass ShapeNet dataset, hosted by [Niemeyer et al.](https://github.com/autonomousvision/differentiable_volumetric_rendering)
    It may be downloaded [here](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip).
    
- SRT's [MultiShapeNet (MSN)](https://srt-paper.github.io/#dataset) dataset, specifically version 2.3.3. It may be downloaded via gsutil:
 ```
 pip install gsutil
 mkdir -p data/msn/multi_shapenet_frames/
 gsutil -m cp -r gs://kubric-public/tfds/multi_shapenet_frames/2.3.3/ data/msn/multi_shapenet_frames/
 ```

### Dependencies
This code requires at least Python 3.9 and [PyTorch 1.11](https://pytorch.org/get-started/locally/). Additional dependencies may be installed via `pip -r requirements.txt`.
Note that Tensorflow is required to load SRT's MultiShapeNet data, though the CPU version suffices.

Rendering videos additionally depends on `ffmpeg>=4.3` being available in your `$PATH`.

## Running Experiments
Each run's config, checkpoints, and visualization are stored in a dedicated directory. Recommended configs can be found under `runs/[dataset]/[model]`.

### Training
To train a model on a single GPU, simply run e.g.:
```
python train.py runs/nmr/srt/config.yaml
```
To train on multiple GPUs on a single machine, launch multiple processes via [Torchrun](https://pytorch.org/docs/stable/elastic/run.html), where $NUM_GPUS is the number of GPUs to use:
```
torchrun --standalone --nnodes 1 --nproc_per_node $NUM_GPUS train.py runs/nmr/srt/config.yaml
```
Checkpoints are automatically stored in and (if available) loaded from the run directory. Visualizations and evaluations are produced periodically.
Check the args of `train.py` for additional options. Importantly, to log training progress, use the `--wandb` flag to enable [Weights & Biases](https://wandb.ai).

### Resource Requirements
The default training configurations require a significant amount of total VRAM.
- SRT on NMR (`nmr/srt/config.yaml`) requires about 130GB, e.g. 4 A100 GPUs with 40GB VRAM, each.
- SRT on MSN (`msn/srt/config.yaml`) requires about 350GB, e.g. 6 A100 GPUS with 80GB VRAM, each.

If you do not have those resources, consider modifying the config files by reducing the batch size (`training: batch_size`),
the number of target points per scene (`data: num_points`), or both. The model has not appeared to be particularly sensitive to either.

### Rendering videos
Videos may be rendered using `render.py`, e.g.
```
python render.py runs/nmr/srt/config.yaml --sceneid 1 --motion rotate_and_closeup --fade
```
Rendered frames and videos are placed in the run directory. Check the args of `render.py` for various camera movements,
and `compile_video.py` for different ways of compiling videos.

## Results
Here you find some checkpoints which partially reproduce the quantitative results of the paper.

|Run | Training Iterations | Test Set PSNR | Download |
|---|---|---|---|
|`nmr/srt` |3 Million | 27.28 |[Link](https://drive.google.com/file/d/1gBSMKBduIgweWsVdUSxK-ugGwb0QFd3D/view?usp=sharing)|
|`msn/srt` |4 Million*| 23.39 |[Link](https://drive.google.com/file/d/1cGxY-g99u63Jj_DcmUIudCfJk5yVsu68/view?usp=sharing)|
|`msn/isrt`|2.8 Million** |24.84|[Link](https://drive.google.com/file/d/12gr3deWgGhwDZrAjwT6XmO7w2XYd2J_B/view?usp=sharing)|

(\*) The SRT MSN run was largely trained with a batch size of 192, due to memory constrains.
(\*\*) Similarly, the ISRT MSN run was trained with a batch size of 48, and 4096 target pixels per
training scene.

### Known Issues
On the NMR dataset, SRT overfits to the 24 camera positions in the training dataset (left). It will generally not produce coherent images when given other cameras at test time (right).

<img src="https://drive.google.com/uc?id=1ps9txhEghQOg1nADskt-3ok7umUQ9-jr" alt="NMR Camera Overfitting" width="512"/>

On the MSN dataset, this is not an issue, as cameras are densely sampled in the training data.
The model even has some ability to produce closeups which use cameras outside of the training distribution.

<img src="https://drive.google.com/uc?id=1buhq_iVmuqn9TQXIowGjqPn-52Z2LG6A" alt="MSN Closeup" width="512"/>

## Citation

```
@article{srt22,
  title={{Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations}},
  author={Mehdi S. M. Sajjadi and Henning Meyer and Etienne Pot and Urs Bergmann and Klaus Greff and Noha Radwan and Suhani Vora and Mario Lucic and Daniel Duckworth and Alexey Dosovitskiy and Jakob Uszkoreit and Thomas Funkhouser and Andrea Tagliasacchi},
  journal={{CVPR}},
  year={2022},
  url={https://srt-paper.github.io/},
}
```

