import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Dictionary of files to download with their target paths
files_to_download = {
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/feature_extractor/preprocessor_config.json?download=true": "pretrained_model/stable-diffusion-v1-5/feature_extractor/preprocessor_config.json",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model_index.json?download=true": "pretrained_model/stable-diffusion-v1-5/model_index.json",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json?download=true": "pretrained_model/stable-diffusion-v1-5/unet/config.json",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin?download=true": "pretrained_model/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin",
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-inference.yaml?download=true": "pretrained_model/stable-diffusion-v1-5/v1-inference.yaml",
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json?download=true": "pretrained_model/sd-vae-ft-mse/config.json",
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin?download=true": "pretrained_model/sd-vae-ft-mse/diffusion_pytorch_model.bin",
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors?download=true": "pretrained_model/sd-vae-ft-mse/diffusion_pytorch_model.safetensors",
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json?download=true": "pretrained_model/wav2vec2-base-960h/config.json",
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/feature_extractor_config.json?download=true": "pretrained_model/wav2vec2-base-960h/feature_extractor_config.json",
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/preprocessor_config.json?download=true": "pretrained_model/wav2vec2-base-960h/preprocessor_config.json",
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin?download=true": "pretrained_model/wav2vec2-base-960h/pytorch_model.bin",
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/README.md?download=true": "pretrained_model/wav2vec2-base-960h/README.md",
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/special_tokens_map.json?download=true": "pretrained_model/wav2vec2-base-960h/special_tokens_map.json",
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/tokenizer_config.json?download=true": "pretrained_model/wav2vec2-base-960h/tokenizer_config.json",
    "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json?download=true": "pretrained_model/wav2vec2-base-960h/vocab.json",
    "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/config.json?download=true": "pretrained_model/image_encoder/config.json",
    "https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin?download=true": "pretrained_model/image_encoder/pytorch_model.bin",
    "https://huggingface.co/ZJYang/AniPortrait/resolve/main/audio2mesh.pt?download=true": "pretrained_model/audio2mesh.pt",
    "https://huggingface.co/ZJYang/AniPortrait/resolve/main/audio2pose.pt?download=true": "pretrained_model/audio2pose.pt",
    "https://huggingface.co/ZJYang/AniPortrait/resolve/main/denoising_unet.pth?download=true": "pretrained_model/denoising_unet.pth",
    "https://huggingface.co/ZJYang/AniPortrait/resolve/main/film_net_fp16.pt?download=true": "pretrained_model/film_net_fp16.pt",
    "https://huggingface.co/ZJYang/AniPortrait/resolve/main/motion_module.pth?download=true": "pretrained_model/motion_module.pth",
    "https://huggingface.co/ZJYang/AniPortrait/resolve/main/pose_guider.pth?download=true": "pretrained_model/pose_guider.pth",
    "https://huggingface.co/ZJYang/AniPortrait/resolve/main/reference_unet.pth?download=true": "pretrained_model/reference_unet.pth"
}

# Function to download a file using wget
def download_file(url, dest_path):
    full_path = os.path.join(script_dir, dest_path)
    if not os.path.exists(full_path):
        print(f"Downloading {url} to {full_path}")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        os.system(f"wget -O {full_path} {url}")
    else:
        print(f"{full_path} already exists. Skipping download.")

# Download all files
for url, dest_path in files_to_download.items():
    download_file(url, dest_path)

print("Download completed.")
