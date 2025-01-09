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

#### Check out some demos from [the official demo page](https://tangoflux.github.io/)

## Example Workflow

![example_workflow](https://github.com/user-attachments/assets/afbf7b53-d712-4c9c-a538-53f0dc001f45)

## Usage

**All the necessary models should be automatically downloaded when the TangoFluxLoader node is used for the first time.**

**Models can also be downloaded using the `install.py` script**

**Manual Download:**
- Download TangoFlux from [here](https://huggingface.co/declare-lab/TangoFlux/tree/main) and put everything in `models/tangoflux` (make sure to include the config files)
- Download text encoders from [here](https://huggingface.co/google/flan-t5-large/tree/main) and put everything in `models/text_encoders/google-flan-t5-large` (make sure to include the config files)

The nodes can be found in "TangoFlux" category as `TangoFluxLoader`, `TangoFluxSampler`, `TangoFluxVAEDecodeAndPlay`.

![teacache_options](https://github.com/user-attachments/assets/29e676d9-902b-4ea2-9f72-18d3607996e8)

> [TeaCache](https://github.com/LiewFeng/TeaCache) can speedup TangoFlux 2x without much audio quality degradation, in a training-free manner.
>
>
> ## 📈 Inference Latency Comparisons on a Single A800
> 
> 
> |      TangoFlux      |        TeaCache (0.25)       |    TeaCache (0.4)    |
> |:-------------------:|:----------------------------:|:--------------------:|
> |      ~4.08 s        |        ~2.42 s                |     ~1.95 s         |

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
```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
```
