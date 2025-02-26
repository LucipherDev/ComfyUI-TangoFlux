import os
import logging
import json
import random
import torch
import torchaudio
import re
import subprocess
import shutil

from diffusers import AutoencoderOobleck, FluxTransformer2DModel

from comfy.utils import load_torch_file, ProgressBar
import folder_paths

from .tangoflux.model import TangoFlux, teacache_forward

log = logging.getLogger("TangoFlux")

TANGOFLUX_DIR = os.path.join(folder_paths.models_dir, "tangoflux")
if "tangoflux" not in folder_paths.folder_names_and_paths:
    current_paths = [TANGOFLUX_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["tangoflux"]
folder_paths.folder_names_and_paths["tangoflux"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)
TEXT_ENCODER_DIR = os.path.join(folder_paths.models_dir, "text_encoders")


class TangoFluxLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_teacache": ("BOOLEAN", {"default": False}),
                "rel_l1_thresh": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("TANGOFLUX_MODEL", "TANGOFLUX_VAE")
    RETURN_NAMES = ("model", "vae")
    OUTPUT_TOOLTIPS = ("TangoFlux Model", "TangoFlux Vae")

    CATEGORY = "TangoFlux"
    FUNCTION = "load_tangoflux"
    DESCRIPTION = "Load TangoFlux model"

    def __init__(self):
        self.model = None
        self.vae = None
        self.enable_teacache = False
        self.rel_l1_thresh = 0.25
        self.original_forward = FluxTransformer2DModel.forward

    def load_tangoflux(
        self,
        enable_teacache=False,
        rel_l1_thresh=0.25,
        tangoflux_path=TANGOFLUX_DIR,
        text_encoder_path=TEXT_ENCODER_DIR,
        device="cuda",
    ):
        if self.model is None or self.enable_teacache != enable_teacache:

            pbar = ProgressBar(4)

            log.info("Loading config")

            with open(os.path.join(tangoflux_path, "config.json"), "r") as f:
                config = json.load(f)

            pbar.update(1)

            text_encoder = re.sub(
                r'[<>:"/\\|?*]',
                "-",
                config.get("text_encoder_name", "google/flan-t5-large"),
            )
            text_encoder_path = os.path.join(text_encoder_path, text_encoder)

            log.info("Loading TangoFlux models")
            
            del self.model
            self.model = None
            
            torch.cuda.empty_cache()

            model_weights = load_torch_file(
                os.path.join(tangoflux_path, "tangoflux.safetensors"),
                device=torch.device(device),
            )

            pbar.update(1)

            if enable_teacache:
                log.info("Enabling TeaCache")
                FluxTransformer2DModel.forward = teacache_forward
            else:
                if self.enable_teacache:
                    log.info("Disabling TeaCache")
                FluxTransformer2DModel.forward = self.original_forward

            model = TangoFlux(config, text_encoder_path)

            model.load_state_dict(model_weights, strict=False)
            model.to(device)

            if enable_teacache:
                model.transformer.__class__.enable_teacache = True
                model.transformer.__class__.cnt = 0
                model.transformer.__class__.rel_l1_thresh = rel_l1_thresh
                model.transformer.__class__.accumulated_rel_l1_distance = 0
                model.transformer.__class__.previous_modulated_input = None
                model.transformer.__class__.previous_residual = None

            pbar.update(1)

            self.model = model
            del model
            
            torch.cuda.empty_cache()
            
            self.enable_teacache = enable_teacache
            self.rel_l1_thresh = rel_l1_thresh

            if self.vae is None:
                log.info("Loading TangoFlux VAE")

                vae_weights = load_torch_file(
                    os.path.join(tangoflux_path, "vae.safetensors")
                )
                self.vae = AutoencoderOobleck()
                self.vae.load_state_dict(vae_weights)
                self.vae.to(device)

            pbar.update(1)

        if self.enable_teacache == True and self.rel_l1_thresh != rel_l1_thresh:
            self.model.transformer.__class__.rel_l1_thresh = rel_l1_thresh

            self.rel_l1_thresh = rel_l1_thresh

        return (self.model, self.vae)


class TangoFluxSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TANGOFLUX_MODEL",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000, "step": 1}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3, "min": 1, "max": 100, "step": 1},
                ),
                "duration": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "offload_model_to_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("TANGOFLUX_LATENTS",)
    RETURN_NAMES = ("latents",)
    OUTPUT_TOOLTIPS = "TangoFlux Sample"

    CATEGORY = "TangoFlux"
    FUNCTION = "sample"
    DESCRIPTION = "Sampler for TangoFlux"

    def sample(
        self,
        model,
        prompt,
        steps=50,
        guidance_scale=3,
        duration=10,
        seed=0,
        batch_size=1,
        offload_model_to_cpu=False,
        device="cuda",
    ):
        pbar = ProgressBar(steps)

        with torch.no_grad():
            model.to(device)

            try:
                if model.transformer.__class__.enable_teacache:
                    model.transformer.__class__.num_steps = steps
            except:
                pass

            log.info("Generating latents with TangoFlux")

            latents = model(
                prompt,
                duration=duration,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_samples_per_prompt=batch_size,
                callback_on_step_end=lambda: pbar.update(1),
            )
            
            if offload_model_to_cpu:
                log.info("Offloading model to CPU")
                model.to("cpu")

        return ({"latents": latents, "duration": duration},)


class TangoFluxVAEDecodeAndPlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("TANGOFLUX_VAE",),
                "tile_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "latents": ("TANGOFLUX_LATENTS",),
                "filename_prefix": ("STRING", {"default": "TangoFlux"}),
                "format": (
                    ["wav", "mp3", "flac", "aac", "wma"],
                    {"default": "wav"},
                ),
                "save_output": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_NODE = True

    CATEGORY = "TangoFlux"
    FUNCTION = "play"
    DESCRIPTION = "Decoder and Player for TangoFlux"

    def decode_tiled(self, vae, latents, tile_size=32):
        results = []
        
        with torch.no_grad():
            for latent in latents:
                torch.cuda.empty_cache()

                latent = latent.unsqueeze(0).transpose(2, 1)

                decoded_tiles = []
                for i in range(0, latent.size(2), tile_size):
                    tile = latent[:, :, i:i + tile_size]
                    
                    decoded_tile = vae.decode(tile).sample.cpu()
                    decoded_tiles.append(decoded_tile)

                    del decoded_tile
                    torch.cuda.empty_cache()

                decoded_latent = torch.cat(decoded_tiles, dim=2)
                results.append(decoded_latent)

            results = torch.cat(results, dim=0)
        return results
    

    def decode(self, vae, latents, tile_size=32):
        results = []

        try:
            with torch.no_grad():
                for latent in latents:
                    torch.cuda.empty_cache()

                    decoded = vae.decode(latent.unsqueeze(0).transpose(2, 1)).sample.cpu()
                    results.append(decoded)

                results = torch.cat(results, dim=0)
            return results

        except RuntimeError as e:
            if "OutOfMemoryError" not in type(e).__name__:
                raise e
            torch.cuda.empty_cache()
            log.warning("OOM encountered. Falling back to tiled decoding.")
            return self.decode_tiled(vae, latents, tile_size)
        
    def load_audio_for_vhs(self, file, sample_rate):
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg_path = get_ffmpeg_exe()
        except:
            pass
        
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            if os.path.isfile("ffmpeg"):
                ffmpeg_path = os.path.abspath("ffmpeg")
            elif os.path.isfile("ffmpeg.exe"):
                ffmpeg_path = os.path.abspath("ffmpeg.exe")
        
        if not ffmpeg_path:
            log.error("No valid ffmpeg found")
            return None
                
        args = [ffmpeg_path, "-i", file]
        
        try:
            res =  subprocess.run(args + ["-f", "f32le", "-"],
                                capture_output=True, check=True)
            audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        except subprocess.CalledProcessError:
            log.error("Couldn't export audio")
            return None
        
        audio = audio.reshape((-1, 2)).transpose(0, 1).unsqueeze(0)
        
        return {"waveform": audio, "sample_rate": sample_rate}

    def play(
        self,
        vae,
        latents,
        tile_size=32,
        filename_prefix="TangoFlux",
        format="wav",
        save_output=True,
        device="cuda",
    ):
        audios = []
        pbar = ProgressBar(len(latents) + 2)

        if save_output:
            output_dir = folder_paths.get_output_directory()
            prefix_append = ""
            type = "output"
        else:
            output_dir = folder_paths.get_temp_directory()
            prefix_append = "_temp_" + "".join(
                random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5)
            )
            type = "temp"

        filename_prefix += prefix_append
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(filename_prefix, output_dir)
        )

        os.makedirs(full_output_folder, exist_ok=True)
        
        pbar.update(1)

        duration = latents["duration"]
        latents = latents["latents"]

        vae.to(device)

        log.info("Decoding Tangoflux latents")

        waves = self.decode(vae, latents, tile_size)
        
        pbar.update(1)

        for wave in waves:
            waveform_end = int(duration * vae.config.sampling_rate)
            wave = wave[:, :waveform_end]

            file = f"{filename}_{counter:05}_.{format}"

            torchaudio.save(
                os.path.join(full_output_folder, file), wave, sample_rate=44100
            )

            counter += 1

            audios.append({"filename": file, "subfolder": subfolder, "type": type})
            
            pbar.update(1)
        
        first_file = os.path.join(full_output_folder, audios[0]["filename"])
        audio_for_vhs = self.load_audio_for_vhs(first_file, 44100)

        return {
            "ui": {"audios": audios},
            "result": (audio_for_vhs,)
        }


NODE_CLASS_MAPPINGS = {
    "TangoFluxLoader": TangoFluxLoader,
    "TangoFluxSampler": TangoFluxSampler,
    "TangoFluxVAEDecodeAndPlay": TangoFluxVAEDecodeAndPlay,
}
