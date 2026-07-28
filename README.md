# Diffusion-101: A Step-by-Step Guide to Training Diffusion Models

A comprehensive, beginner-friendly guide to understanding and training diffusion models from scratch. This repository provides both theoretical explanations and practical implementations with interactive Jupyter notebooks, multiple sampling algorithms (DDIM, Heun, DPM-Solver), and flexible model configurations.

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Guide Structure](#guide-structure)
4. [Quick Start](#quick-start)
5. [Step-by-Step Tutorial](#step-by-step-tutorial)
6. [Project Structure](#project-structure)
7. [Features](#features)
8. [Resources](#resources)

## Introduction

Diffusion models have revolutionized generative AI, powering state-of-the-art image generation models like DALL-E 2, Stable Diffusion, and Midjourney. This guide will walk you through:

- Understanding the mathematical foundations of diffusion models
- Setting up your development environment
- Preparing and preprocessing datasets (with automatic CIFAR-10 download)
- Implementing a diffusion model from scratch with U-Net architecture
- Training your model with ε-prediction
- Generating images with multiple sampling algorithms (DDIM, Heun, DPM-Solver with Karras sigmas)
- Interactive experimentation with different model configurations

## Prerequisites

- Python 3.8+
- Basic understanding of deep learning (neural networks, backpropagation)
- Familiarity with PyTorch or TensorFlow
- GPU recommended (but CPU training is possible for small models)

## Guide Structure

This guide is organized into clear, sequential steps:

- **Step 1**: [Understanding Diffusion Models](./guides/01-understanding-diffusion-models.md) - Theory and intuition
- **Step 2**: [Environment Setup](./guides/02-environment-setup.md) - Installing dependencies
- **Step 3**: [Data Preparation](./guides/03-data-preparation.md) - Dataset handling and preprocessing
- **Step 4**: [Model Architecture](./guides/04-model-architecture.md) - Building the U-Net backbone
- **Step 5**: [Training Process](./guides/05-training-process.md) - Implementing the training loop
- **Step 6**: [Sampling & Inference](./guides/06-sampling-inference.md) - Generating images
- **Step 7**: [Advanced Techniques](./guides/07-advanced-techniques.md) - Optimizations and improvements

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Diffusion-101
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the interactive notebooks**
   ```bash
   jupyter notebook notebooks/
   ```
   
   Recommended order:
   - Start with `01-introduction.ipynb` to understand diffusion models
   - Use `02-data-exploration.ipynb` to explore datasets (auto-downloads CIFAR-10)
   - Train models in `03-training-visualization.ipynb` (multiple model configurations)
   - Experiment with samplers in `04-pytorch-samplers.ipynb` (DDIM, Heun, DPM-Solver)

4. **Or start training via script**
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

## Step-by-Step Tutorial

### Interactive Notebooks (Recommended for Beginners)

The notebooks provide hands-on learning with automatic dataset handling:

1. **[01-introduction.ipynb](./notebooks/01-introduction.ipynb)** - Visualize forward/reverse diffusion processes
2. **[02-data-exploration.ipynb](./notebooks/02-data-exploration.ipynb)** - Explore datasets (auto-downloads CIFAR-10 if needed)
3. **[03-training-visualization.ipynb](./notebooks/03-training-visualization.ipynb)** - Train models with multiple configurations (small/medium/large/custom)
4. **[04-pytorch-samplers.ipynb](./notebooks/04-pytorch-samplers.ipynb)** - Interactive sampler playground with DDIM, Heun, and DPM-Solver

### Detailed Guides

Follow the markdown guides for deeper understanding:

1. Start with [Understanding Diffusion Models](./guides/01-understanding-diffusion-models.md) to build intuition
2. Set up your environment with [Environment Setup](./guides/02-environment-setup.md)
3. Prepare your data following [Data Preparation](./guides/03-data-preparation.md)
4. Build the model using [Model Architecture](./guides/04-model-architecture.md)
5. Train your model with [Training Process](./guides/05-training-process.md)
6. Generate images using [Sampling & Inference](./guides/06-sampling-inference.md)
7. Explore [Advanced Techniques](./guides/07-advanced-techniques.md) for improvements

## Project Structure

```
Diffusion-101/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── guides/                   # Step-by-step guide documents
│   ├── 01-understanding-diffusion-models.md
│   ├── 02-environment-setup.md
│   ├── 03-data-preparation.md
│   ├── 04-model-architecture.md
│   ├── 05-training-process.md
│   ├── 06-sampling-inference.md
│   └── 07-advanced-techniques.md
├── notebooks/                # Jupyter notebooks for interactive learning
│   ├── 01-introduction.ipynb          # Introduction to diffusion concepts
│   ├── 02-data-exploration.ipynb      # Dataset exploration (auto-downloads CIFAR-10)
│   ├── 03-training-visualization.ipynb # Training and visualization (with built-in training)
│   └── 04-pytorch-samplers.ipynb      # Interactive sampler playground (DDIM, Heun, DPM-Solver)
├── src/                      # Source code
│   ├── models/               # Model architectures
│   │   ├── __init__.py
│   │   ├── unet.py          # U-Net backbone
│   │   └── diffusion.py     # Diffusion process
│   ├── data/                 # Data utilities
│   │   ├── __init__.py
│   │   └── dataset.py       # Dataset classes
│   ├── training/             # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py        # Training loop
│   └── utils/                # Helper functions
│       ├── __init__.py
│       ├── scheduler.py      # Noise scheduling
│       └── visualization.py  # Plotting utilities
├── scripts/                  # Executable scripts
│   ├── train.py             # Training script
│   ├── sample.py            # Inference script
│   └── preprocess_data.py   # Data preprocessing
├── configs/                  # Configuration files
│   └── default.yaml         # Default training config
└── outputs/                  # Generated outputs (created at runtime)
    ├── checkpoints/         # Model checkpoints
    ├── samples/             # Generated images
    └── logs/                # Training logs
```

## Features

- 🎓 **Beginner-friendly**: Step-by-step guides with clear explanations
- 🔬 **Interactive notebooks**: Hands-on learning with Jupyter notebooks
- 🚀 **Multiple samplers**: Compare DDIM, Heun, and DPM-Solver with Karras sigmas
- 🎨 **Flexible training**: Choose from small, medium, large, or custom model configurations
- 📊 **Auto-dataset handling**: Automatically downloads and preprocesses CIFAR-10
- 🖼️ **Visualization tools**: Built-in visualization for training progress and generated samples
- ⚙️ **Production-ready**: Clean code structure with configurable training scripts

## Resources

### Papers
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) - Nichol & Dhariwal, 2021
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) - Song et al., 2020
- [DPM-Solver: Fast Solver for Diffusion Probabilistic Models](https://arxiv.org/abs/2206.00927) - Lu et al., 2022

### Tutorials & Blogs
- [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Lilian Weng
- [An Introduction to Diffusion Models](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/) - AssemblyAI

### Datasets
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) - Face images
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - Small natural images
- [LSUN](https://www.yf.io/p/lsun) - Large-scale scene understanding

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This guide is inspired by the excellent work in the diffusion modeling community and aims to make these concepts more accessible to learners.

