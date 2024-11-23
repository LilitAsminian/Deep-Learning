# Deep Learning VAE and CVAE Project

This repository contains the implementation of a Variational Autoencoder (VAE) and Conditional Variational Autoencoder (CVAE) for image generation.

## Project Overview
- **Objective**: To train VAE and CVAE models to generate images from complex datasets.
- **Models**:
  - Variational Autoencoder (VAE)
  - Conditional Variational Autoencoder (CVAE)
- **Dataset**:
  - Art Images

## Features
- **Encoder** and **Decoder** architecture
- **Reparameterization trick** for latent space sampling
- Loss function combining:
  - Binary Cross-Entropy (reconstruction loss)
  - KL Divergence (latent space regularization)

## Folder Structure
.
├── dataset/                # Folder for datasets
│   ├── Art_Images/         # Main dataset folder
│   │   ├── training_set/   # Training images
│   │   └── validation_set/ # Validation images
├── models/                 # Folder containing model implementations
│   ├── cvae.py             # Conditional Variational Autoencoder implementation
│   └── vae.py              # Variational Autoencoder implementation
├── vae/                    # env
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── vae_cvae.ipynb          # Jupyter Notebook for model training and visualization

