import sys
import os
import logging
import subprocess
import traceback
import json 
import re

log = logging.getLogger("TangoFlux")

download_models = True

try:
    import folder_paths
    TANGOFLUX_DIR = os.path.join(folder_paths.models_dir, "tangoflux")
    TEXT_ENCODER_DIR = os.path.join(folder_paths.models_dir, "text_encoders")
except:
    download_models = False
    log.info("Not called by ComfyUI Manager. Models will not be downloaded")

EXT_PATH = os.path.dirname(os.path.abspath(__file__))
    
try:
    log.info(f"Installing requirements")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", f"{EXT_PATH}/requirements.txt", "--no-warn-script-location"])
    
    if download_models:
        from huggingface_hub import snapshot_download

        try:
            if not os.path.exists(TANGOFLUX_DIR):
                log.info(f"Downloading TangoFlux models to: {TANGOFLUX_DIR}")
                snapshot_download(
                    repo_id="declare-lab/TangoFlux",
                    allow_patterns=["*.json", "*.safetensors"],
                    local_dir=TANGOFLUX_DIR,
                    local_dir_use_symlinks=False,
                )
        except Exception:
            traceback.print_exc()
            log.error(f"Failed to download TangoFlux models")
            
        log.info(f"Loading config")

        with open(os.path.join(TANGOFLUX_DIR, "config.json"), "r") as f:
            config = json.load(f)
            
        try:
            text_encoder = re.sub(r'[<>:"/\\|?*]', '-', config.get("text_encoder_name", "google/flan-t5-large"))
            text_encoder_path = os.path.join(TEXT_ENCODER_DIR, text_encoder)
            
            if not os.path.exists(text_encoder_path):
                log.info(f"Downloading text encoders to: {text_encoder_path}")
                snapshot_download(
                    repo_id=config.get("text_encoder_name", "google/flan-t5-large"),
                    allow_patterns=["*.json", "*.safetensors", "*.model"],
                    local_dir=text_encoder_path,
                    local_dir_use_symlinks=False,
                )
        except Exception:
            traceback.print_exc()
            log.error(f"Failed to download text encoders")
    
    log.info(f"TangoFlux Installation completed")
        
except Exception:
    traceback.print_exc()
    log.error(f"TangoFlux Installation failed")