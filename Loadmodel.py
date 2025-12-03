import os
import torch

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler

from CoDA_SDXLBasePipeline import CoDA_SDXL
from CoDA_SDXLRefinerPipeline import CoDA_Refiner

model_type = torch.float16

def load_sdxl_and_refiner(args, VAE16_ONLY=False, VAEFIX=False ):

    local_model_path = args.local_model_path
    os.makedirs(local_model_path, exist_ok=True)
    sdxl_base_path = os.path.join(local_model_path, "sdxl-base")
    sdxl_refiner_path = os.path.join(local_model_path, "sdxl-refiner")

    if VAEFIX:
        sdxl_vae_path = os.path.join(sdxl_base_path, "vaefixfp16")
        download_path = "madebyollin/sdxl-vae-fp16-fix"
    else:
        sdxl_vae_path = os.path.join(sdxl_base_path, "vae")
        download_path = "stabilityai/sdxl-vae"

    if VAE16_ONLY:
        print(f"Loading SDXL VAE16 fix: {VAEFIX} Model...")
        if os.path.exists(sdxl_vae_path) and os.listdir(sdxl_vae_path):
            vae = AutoencoderKL.from_pretrained(sdxl_vae_path, torch_dtype=model_type)
        else:
            print(f"Downloading SDXL VAE fix: {VAEFIX} model from Hugging Face...")
            vae = AutoencoderKL.from_pretrained(download_path, torch_dtype=model_type)
            vae.save_pretrained(sdxl_vae_path)
            print(f"SDXL VAE16 fix: {VAEFIX} model saved to: {sdxl_vae_path}")
        return vae.eval()

    unet_path = os.path.join(sdxl_base_path, "unet")
    if os.path.exists(sdxl_base_path) and os.path.isdir(unet_path):
        print(f"Loading SDXL base model from local path: {sdxl_base_path}")
        base_pipeline = CoDA_SDXL.from_pretrained(
            sdxl_base_path,
            torch_dtype=model_type,
            use_safetensors=True,
        )
        if VAEFIX:
            print(f"Loading VAE fix from: {sdxl_vae_path}")
            vae = AutoencoderKL.from_pretrained(sdxl_vae_path, torch_dtype=model_type)
            base_pipeline.vae = vae
    else:
        print("Downloading SDXL base model from Hugging Face...")
        base_pipeline = CoDA_SDXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=model_type,
            use_safetensors=True,
            variant="fp16"
        )
        base_pipeline.save_pretrained(sdxl_base_path, safe_serialization=True)
        print(f"SDXL base model saved to: {sdxl_base_path}")

        if VAEFIX:
            if not (os.path.exists(sdxl_vae_path) and os.listdir(sdxl_vae_path)):
                print(f"Downloading SDXL VAE fix: {VAEFIX} model from Hugging Face...")
                temp_vae = AutoencoderKL.from_pretrained(download_path, torch_dtype=model_type)
                temp_vae.save_pretrained(sdxl_vae_path)
                print(f"SDXL VAE16 fix: {VAEFIX} model saved to: {sdxl_vae_path}")
                vae = temp_vae
            else:
                 print(f"Loading SDXL VAE16 fix: {VAEFIX} Model...")
                 vae = AutoencoderKL.from_pretrained(sdxl_vae_path, torch_dtype=model_type)
            base_pipeline.vae = vae

    base_pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        sdxl_base_path,
        subfolder="scheduler",
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )

    if os.path.exists(sdxl_refiner_path) and os.listdir(sdxl_refiner_path):
        print(f"Loading SDXL refiner from local path: {sdxl_refiner_path}")
        refiner = CoDA_Refiner.from_pretrained(
            sdxl_refiner_path,
            text_encoder_2=base_pipeline.text_encoder_2,
            vae=base_pipeline.vae,
            torch_dtype=model_type,
            use_safetensors=True,
        )
    else:
        print("Downloading SDXL refiner from Hugging Face...")
        refiner = CoDA_Refiner.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base_pipeline.text_encoder_2,
            vae=base_pipeline.vae,
            torch_dtype=model_type,
            use_safetensors=True,
            variant="fp16"
        )
        refiner.save_pretrained(sdxl_refiner_path, safe_serialization=True)
        print(f"SDXL refiner saved to: {sdxl_refiner_path}")

    print(f"Scheduler class name: {base_pipeline.scheduler.__class__.__name__}")

    base_pipeline.unet.eval()
    base_pipeline.vae.eval()
    base_pipeline.text_encoder.eval()
    base_pipeline.text_encoder_2.eval()
    refiner.unet.eval()
    refiner.vae.eval()
    refiner.text_encoder_2.eval()

    return base_pipeline, refiner