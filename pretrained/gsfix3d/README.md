---
language:
- en
license: openrail++
library_name: diffusers
tags:
- 3D Gaussian Splatting
- computer vision
pinned: true
---

<h1 align="center">GSFixer Full Model Card (Replica-room1)</h1>

<p align="center">
<a title="Github" href="https://github.com/GSFix3D/GSFix3D" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
    <img src="https://img.shields.io/github/stars/GSFix3D/GSFix3D?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="Github">
</a>
<a title="Website" href="https://gsfix3d.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
    <img src="https://img.shields.io/badge/%E2%99%A5%20Project%20-Website-blue" alt="Website">
</a>
<a title="arXiv" href="https://arxiv.org/pdf/2508.14717" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
    <img src="https://img.shields.io/badge/%F0%9F%93%84%20Read%20-Paper-AF3436" alt="arXiv">
</a>
<a title="License" href="https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
    <img src="https://img.shields.io/badge/License-OpenRAIL++-929292" alt="License">
</a>
</p>

This model card is for the `gsfixer-full-replica-room1` model, which is designed for novel view repair of an image rendered from a 3DGS model with additional information gain from the mesh. The model is fine-tuned from the `gsfixer-full` [model](https://huggingface.co/goldoak1421/gsfixer-full) following [GSFix3D](https://github.com/GSFix3D/GSFix3D) protocol. 


## Model Details
- **Developed by:** Jiaxin Wei, Stefan Leutenegger, Simon Schaefer.
- **License:** [CreativeML Open RAIL++-M License](https://github.com/GSFix3D/GSFix3D/blob/main/LICENSES/LICENSE-MODEL.txt).
- **Model Description:** This model can be used to generate a repaired image of two input rendered images from the mesh and 3DGS, respectively. 
  - **Resolution**: Even though any resolution can be processed, the model inherits the base diffusion model's effective resolution of roughly **768** pixels. 
    This means that for optimal predictions, any larger input image should be resized to make the longer side 768 pixels before feeding it into the model.
  - **Steps and scheduler**: This model was designed for usage with the **DDIM** scheduler and between **1 and 50** denoising steps.
- **Cite as:**
    ```bibtex
    @article{wei2026gsfix3d,
    title={GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting}, 
    author={Wei, Jiaxin and Leutenegger, Stefan and Schaefer, Simon},
    booktitle={2026 International Conference on 3D Vision (3DV)},
    year={2026},
    organization={IEEE}
    }
    ```