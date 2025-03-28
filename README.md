# Plugging in Long-CLIP into LaVie

- Used Long-CLIP instead of LaVie's CLIP which uses only 77 tokens. Now with Long-CLIP, it uses 248 tokens as context for the SD model
- To make this work, install libraries from `requirements.txt`

## Download the following weights:
In `LaVie/pretrained_models`, download pre-trained [LaVie models](https://huggingface.co/YaohuiW/LaVie/tree/main), [Stable Diffusion 1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main), [stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/main) to `./pretrained_models`. You should be able to see the following:
```
├── pretrained_models
│   ├── lavie_base.pt
│   ├── lavie_interpolation.pt
│   ├── lavie_vsr.pt
│   ├── stable-diffusion-v1-4
│   │   ├── ...
└── └── stable-diffusion-x4-upscaler
        ├── ...
```

In `LaVie/base/Long-CLIP/checkpoints`, download the checkpoints of [LongCLIP-B](https://huggingface.co/BeichenZhang/LongCLIP-B) and/or [LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L)
```
├── checkpoints
│   ├── longclip-B.pt
│   ├── ...
```

In `LaVie/base/Long-CLIP/SDXL/longclip-L`, download [https://huggingface.co/BeichenZhang/LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L)
```
├── longclip-L
│   ├── longclip-L.pt
│   ├── ...
```

In `LaVie/base/Long-CLIP/SDXL/openclip-bigG`, download [https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
```
├── openclip-bigG
│   ├── config.json
│   ├── open_clip_config.json
│   ├── open_clip_model.safetensors
│   ├── open_clip_pytorch_model.bin
│   ├── ...
```


## Sample video from LaVie using Long-CLIP
To generate videos by sampling from the base model of LaVie using Long-CLIP:
- `cd LaVie/base`
- `python pipelines/sample.py --config configs/sample.yaml`




