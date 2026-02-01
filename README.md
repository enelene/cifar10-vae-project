# CIFAR-10 Generative Models: From VAE to VQ-VAE

This repository contains a comprehensive comparative study of generative models trained on the CIFAR-10 dataset. The project explores the evolution of Variational Autoencoders (VAEs) from simple convolutional baselines to advanced discrete latent variable models, implemented entirely from scratch in PyTorch.

**Constraint:** All models are **unconditional**. Class labels were explicitly discarded during training to force the models to learn the marginal distribution $p(x)$ purely from pixel data.

**Wandb Report:** https://wandb.ai/egabe21-free-university-of-tbilisi-/cifar10-vae-final/reports/Untitled-Report--VmlldzoxNTgwODk4Mw

## 🚀 Key Features
* **From-Scratch Implementations:** No high-level wrappers. All architectures, loss functions (ELBO, Perceptual Loss), and sampling loops are implemented in pure PyTorch.
* **Advanced Architectures:**
    * **ResNet-VAE:** Deep residual encoder/decoder to solve vanishing gradients.
    * **DFC-VAE:** Deep Feature Consistent VAE using VGG-19 Perceptual Loss for sharp texture generation.
    * **VQ-VAE (EMA):** Vector Quantized VAE with Exponential Moving Average codebook updates.
* **Scientific Experiments:** Rigorous analysis of the $\beta$-VAE trade-off (Blurry vs. Sharp) and Latent Space Interpolation.
* **MLOps Integration:** Full experiment tracking and artifact logging using **Weights & Biases (WandB)**.
* **Quantitative Evaluation:** Fréchet Inception Distance (FID) calculation.

## 📂 Repository Structure

The codebase is organized as follows:

```text
├── src/
│   ├── model.py              # Baseline Convolutional VAE implementation
│   ├── model_resnet.py       # ResNet-18 style VAE (Residual Blocks)
│   ├── model_vqvae.py        # VQ-VAE with EMA Vector Quantizer
│   ├── dataset.py            # CIFAR-10 loading & normalization
│   ├── utils.py              # Loss functions (KL, Perceptual/VGG) & FID Calculator
│   └── train.py              # Modular training engine
├── main_notebook.ipynb       # Main experiment runner (Jupyter Notebook)
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

# Best Experiment: The DFC-ResNet

highest-performing model combined the structural depth of ResNet with the texture-awareness of Deep Feature Consistency (Perceptual Loss).Run ID: ResNet_DFC_Run-beta=0.1-epoch=50Configuration:Architecture: ResNet-VAE (Deep Residual Networks)Loss: $\beta=0.1$ (Prioritizing sharpness over latent smoothness) + $\alpha=0.5$ (VGG Perceptual Loss)Epochs: 50Result: This configuration solved the "MSE Blur" problem standard VAEs face on CIFAR-10. By forcing the model to match VGG-19 features (edges, textures) rather than just pixels, and relaxing the KL constraint ($\beta=0.1$), we achieved significantly sharper reconstructions with visible animal features (ears, legs) and distinct backgrounds.

Methods Implemented

1. ResNet-VAE (Structure)
To address the capacity limits of simple ConvNets, we implemented a ResNet-VAE using Residual Blocks with skip connections. This allows gradients to flow through deep networks, preserving spatial structure that is usually lost in standard bottlenecks.

2. DFC-VAE (Texture)
Standard VAEs use Mean Squared Error (MSE), which causes blurriness by "averaging" possible outputs. We implemented Deep Feature Consistency:

Method: Compare activation maps from a frozen VGG-19 network.

Effect: The model penalizes missing features (e.g., "missing dog ear") rather than just missing pixel colors.

3. VQ-VAE (Discrete Latents)
We implemented a Vector Quantized VAE using a discrete codebook.

Method: Snaps continuous latent vectors to the nearest "code" in a dictionary.

EMA: Implemented Exponential Moving Average updates to prevent "Dead Codes" (Codebook Collapse).


# Usage
pip install -r requirements.txt

# References
VAE: Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

Beta-VAE: Higgins, I., et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.

DFC-VAE: Hou, X., et al. (2017). Deep Feature Consistent Variational Autoencoder.

VQ-VAE: Razavi, A., et al. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2.
