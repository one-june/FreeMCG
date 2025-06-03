### Getting Started

1. Clone the repository
```
git clone https://github.com/one-june/FreeMCG.git
cd FreeMCG
```

2. Download pretrained unconditional diffusion model checkpoint (256x256_diffusion.pt) from OpenAI's [`guided-diffusion`](https://github.com/openai/guided-diffusion) repository and place it in the folder `diffusion_ckpts/openai-guided-diffusion`

3. Prepare imagenet validation set and input its path as the `--imagenet_dir` argument in `run.sh`.

```
ILSVRC2012/
└── val/
    ├── n01440764/
    │ ├── ILSVRC2012_val_00000001.JPEG
    │ ├── ...
    │ └── (50 images total)
    ├── n01443537/
    │ ├── ILSVRC2012_val_00000051.JPEG
    │ └── ...
    └── ...
```

4. Run FreeMCG example. The results will be saved be the `outputs` directory:
```bash
run.sh
```