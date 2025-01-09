# ComfyUI-TangoFlux
ComfyUI Custom Nodes for ["TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching"](https://arxiv.org/abs/2412.21037). These nodes, adapted from [the official implementations](https://github.com/declare-lab/TangoFlux/), generates high-quality 44.1kHz audio up to 30 seconds using just a text promptproduction.

## Installation

1. Navigate to your ComfyUI's custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/LucipherDev/ComfyUI-TangoFlux
```

3. Install requirements:
```bash
cd ComfyUI-TangoFlux
python install.py
```

### Or Install via ComfyUI Manager

## Usage

**If you installed from the ComfyUI-Manager, all the necessary models should be automatically downloaded into the `models/tangoflux` folder.**

**Otherwise they will be automatically downloaded when the LoadAniDoc node is used for the first time.**

**Manual Download:**
- Download TangoFlux from [here](https://huggingface.co/declare-lab/TangoFlux/tree/main) and put everything in `models/tangoflux` (make sure to include the config files)
- Download text encoders from [here](https://huggingface.co/google/flan-t5-large/tree/main) and put everything in `models/tangoflux/text_encoders/google-flan-t5-large`

The nodes can be found in "TangoFlux" category as `TangoFluxDocLoader`, `TangoFluxSampler`, `TangoFluxVAEDecodeAndPlay`.

## Citation

```bibtex
@misc{hung2024tangofluxsuperfastfaithful,
      title={TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization}, 
      author={Chia-Yu Hung and Navonil Majumder and Zhifeng Kong and Ambuj Mehrish and Rafael Valle and Bryan Catanzaro and Soujanya Poria},
      year={2024},
      eprint={2412.21037},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2412.21037}, 
}
```
