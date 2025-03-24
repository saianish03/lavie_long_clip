import os
from huggingface_hub import snapshot_download, hf_hub_download
import shutil

# Create the base directory
os.makedirs("/scratch/sreeramagiri.s/lavie_long_clip/LaVie/pretrained_models", exist_ok=True)


# Function to download model with progress tracking
def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Successfully downloaded {repo_id} to {local_dir}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")

# Download Stable Diffusion v1.4
sd_path = "/scratch/sreeramagiri.s/lavie_long_clip/LaVie/pretrained_models/stable-diffusion-v1-4"
download_model("CompVis/stable-diffusion-v1-4", sd_path)

# Download stable-diffusion-x4-upscaler
upscaler_path = "/scratch/sreeramagiri.s/lavie_long_clip/LaVie/pretrained_models/stable-diffusion-x4-upscaler"
download_model("stabilityai/stable-diffusion-x4-upscaler", upscaler_path)