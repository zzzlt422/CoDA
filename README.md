**[ICLR 2026]** This repository contains the official implementation of the paper: **"CoDA: From Text-to-Image Diffusion Models to Training-Free Dataset Distillation"**.

## 🔥 News
- **[2026.01.26]** 🎉 We are thrilled to announce that **CoDA** has been accepted to **ICLR 2026**!
- **[2025.12.04]** CoDA is released on [arXiv](https://arxiv.org/abs/2512.03844).

## 📖 Introduction

CoDA is a novel dataset distillation framework leveraging an off-the-shelf text-to-image model (SDXL). Instead of relying on diffusion models pre-trained on the target dataset (e.g., utilizing an ImageNet-trained DiT to distill ImageNet), we introduce "Distribution Discovery" and "Distribution Alignment" to bridge the distribution gap between general generative priors and specific domains. This achieves SOTA performance without the prohibitive cost of pre-training, establishing CoDA as a truly universal solution capable of performing dataset distillation tasks on any arbitrary dataset.

## 🛠️ Requirements

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## 🚀 Usage

Please make sure to navigate to the project root directory first:

```
cd CoDA
scripts/CoDA.sh
```
