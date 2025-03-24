import os
import torch
import argparse
import torchvision

from pipeline_videogen import VideoGenPipeline

from download import find_model
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from omegaconf import OmegaConf
from Long_CLIP.model import longclip
# from long_clip_features import get_features
from typing import Union, Tuple, Optional


import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from models import get_models
import imageio

def stack_embeddings(
    text_features: torch.Tensor, 
    pose_features: torch.Tensor,
    dim: int = 1,
    verify_output: bool = True
) -> torch.Tensor:
    """
    Stack different embeddings along a specified dimension.
    
    For two tensors of shape [N, D] each, this function creates a tensor of 
    shape [2*N, D] (if dim=0) or [N, 2*D] (if dim=1) where text and pose 
    embeddings are stacked together.
    
    Args:
        text_features: First input tensor of embeddings, expected shape [N, D]
        pose_features: Second input tensor of embeddings, expected shape [N, D]
        dim: Dimension along which to stack. Default 0 means stack vertically,
             1 means stack horizontally (concatenate features)
        verify_output: If True, perform additional verification on the output
        
    Returns:
        Tensor with embeddings stacked together
        
    Raises:
        ValueError: For invalid inputs or dimensions
    """
    # Input validation
    if not isinstance(text_features, torch.Tensor) or not isinstance(pose_features, torch.Tensor):
        raise ValueError(f"Expected torch.Tensors, got {type(text_features)} and {type(pose_features)}")
    
    if text_features.dim() != 2 or pose_features.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got {text_features.dim()}D and {pose_features.dim()}D tensors")
    
    if text_features.shape != pose_features.shape:
        raise ValueError(f"Input shapes must match: {text_features.shape} vs {pose_features.shape}")
    
    if dim not in [0, 1]:
        raise ValueError(f"Dimension must be 0 or 1, got {dim}")
    
    # Get original shape for verification
    original_shape = text_features.shape
    device = text_features.device
    dtype = text_features.dtype
    
    # Check device and dtype consistency between inputs
    if verify_output and (text_features.device != pose_features.device):
        raise ValueError(f"Device mismatch between inputs: {text_features.device} vs {pose_features.device}")
    
    if verify_output and (text_features.dtype != pose_features.dtype):
        raise ValueError(f"Dtype mismatch between inputs: {text_features.dtype} vs {pose_features.dtype}")
    
    # Perform stacking based on dimension
    if dim == 0:
        # Stack vertically - interleave text and pose embeddings
        # For [t1, t2, ..., tN] and [p1, p2, ..., pN] we get [t1, p1, t2, p2, ..., tN, pN]
        stacked = torch.stack([text_features, pose_features], dim=1).view(-1, original_shape[1])
        expected_shape = (original_shape[0] * 2, original_shape[1])
    else:  # dim == 1
        # Stack horizontally - concatenate text and pose embeddings
        # For embeddings [t1, t2, ..., tN] and [p1, p2, ..., pN] we get [t1;p1, t2;p2, ..., tN;pN]
        # where ; represents concatenation
        stacked = torch.cat([text_features, pose_features], dim=1)
        expected_shape = (original_shape[0], original_shape[1] * 2)
    
    # Verify output if required
    if verify_output:
        # Check shape
        assert stacked.shape == expected_shape, f"Expected shape {expected_shape}, got {stacked.shape}"
        
        # Check device and dtype consistency
        assert stacked.device == device, f"Device mismatch: {device} vs {stacked.device}"
        assert stacked.dtype == dtype, f"Dtype mismatch: {dtype} vs {stacked.dtype}"
        
        # Additional verification based on dimension
        if dim == 0:
            for i in range(original_shape[0]):
                # Verify text and pose embeddings are correctly interleaved
                text_idx, pose_idx = 2*i, 2*i+1
                assert torch.allclose(stacked[text_idx], text_features[i]), \
                    f"Text embedding at {i} doesn't match stacked at {text_idx}"
                assert torch.allclose(stacked[pose_idx], pose_features[i]), \
                    f"Pose embedding at {i} doesn't match stacked at {pose_idx}"
        else:  # dim == 1
            # Each row should be [text_embedding, pose_embedding]
            left_half = stacked[:, :original_shape[1]]
            right_half = stacked[:, original_shape[1]:]
            assert torch.allclose(left_half, text_features), "Text embeddings don't match left half"
            assert torch.allclose(right_half, pose_features), "Pose embeddings don't match right half"
    
    return stacked


def main(args):
	if args.seed is not None:
		torch.manual_seed(args.seed)
	torch.set_grad_enabled(False)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# Replace hardcoded prompts with the ones from config
	# Using the first two prompts from args.text_prompt (or fewer if only one exists)

	model, preprocess = longclip.load("Long_CLIP/checkpoints/longclip-B.pt", device=device)
	sample_prompts = args.text_prompt
	text = longclip.tokenize(sample_prompts).to(device)
	print("Getting features from Long_CLIP...")
	with torch.no_grad():
		text_features = model.encode_text(text)
		text_features = stack_embeddings(text_features, text_features, dim=1) # pose_features if pose_features is not None else text_features
	print("text_features shape: ", text_features.shape)
	print("Text Features generated from Long_Clip") if len(text_features)!=0 else print("ERROR NO FEATURES FROM LONG_CLIP")
	print("Done.")
    
	sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
	unet = get_models(args, sd_path).to(device, dtype=torch.float16)
	state_dict = find_model(args.ckpt_path)
	unet.load_state_dict(state_dict)
	
	vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
	tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
	text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device) # huge

	# set eval mode
	unet.eval()
	vae.eval()
	text_encoder_one.eval()
	
	if args.sample_method == 'ddim':
		scheduler = DDIMScheduler.from_pretrained(sd_path, 
											   subfolder="scheduler",
											   beta_start=args.beta_start, 
											   beta_end=args.beta_end, 
											   beta_schedule=args.beta_schedule)
	elif args.sample_method == 'eulerdiscrete':
		scheduler = EulerDiscreteScheduler.from_pretrained(sd_path,
											   subfolder="scheduler",
											   beta_start=args.beta_start,
											   beta_end=args.beta_end,
											   beta_schedule=args.beta_schedule)
	elif args.sample_method == 'ddpm':
		scheduler = DDPMScheduler.from_pretrained(sd_path,
											  subfolder="scheduler",
											  beta_start=args.beta_start,
											  beta_end=args.beta_end,
											  beta_schedule=args.beta_schedule)
	else:
		raise NotImplementedError

	videogen_pipeline = VideoGenPipeline(vae=vae, 
								 text_encoder=text_encoder_one, 
								 tokenizer=tokenizer_one, 
								 scheduler=scheduler, 
								 unet=unet).to(device)
	videogen_pipeline.enable_xformers_memory_efficient_attention()

	if not os.path.exists(args.output_folder):
		os.makedirs(args.output_folder)

	video_grids = []
	for i, prompt in enumerate(args.text_prompt):
		print('Processing the ({}) prompt'.format(prompt))
		prompt_text_feature = text_features[i]
		videos = videogen_pipeline(prompt, 
								video_length=args.video_length, 
								height=args.image_size[0], 
								width=args.image_size[1], 
								num_inference_steps=args.num_sampling_steps,
								guidance_scale=args.guidance_scale,
								clip_text_feature = prompt_text_feature).video
		imageio.mimwrite(args.output_folder + prompt.replace(' ', '_') + '.mp4', videos[0], fps=8, quality=9) # highest quality is 10, lowest is 0
	
	print('save path {}'.format(args.output_folder))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="")
	args = parser.parse_args()

	main(OmegaConf.load(args.config))