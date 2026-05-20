# Convolutional Autoencoders for Image Restoration: A Study on Latent Spaces and Image Reconstruction using CIFAR-10

This repository contains the experimental development, implementation, and rigorous analysis of Deep Convolutional Autoencoder architectures evaluated on the **CIFAR-10** dataset. The primary objective is to study the capability of compression algorithms and unsupervised representation learning across multiple image-to-image mapping operations.

This project empirically validates how bottleneck networks (**Encoders and Decoders**), leveraging modern dimensional reductions through strided convolutions and internal multi-channel projections, can perform complex continuous matrix reconstructions for severe pixel deterioration variants.

---

## 🏛️ Theoretical Framework & Introduction

An **Autoencoder** is a neural network design paradigm trained in an unsupervised manner to copy its input to its output under specific algorithmic constraints. It consists of two complementary operational parts:

1. **Encoder ($\mathcal{E}$):** Maps the high-dimensional input image $x \in \mathbb{R}^{32\times32\times3}$ to a low-dimensional code or bottleneck space $\mathbf{z} \in \mathbb{R}^{8\times8\times d}$ where only relevant topological characteristics are preserved.
2. **Decoder ($\mathcal{D}$):** Takes the latent vectors from the bottleneck $\mathbf{z}$ and dynamically upsamples the shapes to project a pixel space optimization estimate $\hat{x} \in \mathbb{R}^{32\times32\times3}$.

Mathematically, the goal of this architectural blueprint is to minimize a distance constraint between distributions, governed in this environment by the **Mean Squared Error (MSE)** loss function:

$$\mathcal{L}_{MSE}(x, \hat{x}) = \frac{1}{N} \sum_{i=1}^{N} ||x_i - \mathcal{D}(\mathcal{E}(\tilde{x}_i))||^2_2$$

Where $\tilde{x}$ denotes the input signal (which might contain synthetic noise, gray value masks, or missing geometric regions) and $x$ holds the original clean reference target.

### 🔄 Image Restoration Paradigms Evaluated

| Feature / Task | Denoising (Gaussian / S&P) | Colorization | Inpainting |
| :--- | :--- | :--- | :--- |
| **Input State ($\tilde{x}$)** | Image corrupted with continuous standard deviation or boolean bit flips. | Single-channel luminosity representation (Grayscale). | Multi-channel matrix with localized geometric pixel deletions. |
| **Target State ($x$)** | Clean, normalized source RGB state. | Full 3-channel standard color distribution. | Unmasked complete native continuous original frame. |
| **Operational Goal** | Stochastic filtering & noise isolation. | Artificial color distribution mapping. | Context-aware topological interpolation. |

---

## 🧱 Architectural Topology of Individual Blocks

To ensure deep feature extraction, stability, and clean continuous reconstruction, the system is designed around a symmetric grid. The standard backbone utilizes a bottleneck compression reduction layer down to an $8 \times 8$ spatial dimension with $128$ dimensions ($latent\_dim$ adjustable).

### 🛠️ Deep Convolutional Autoencoder Backbone Architecture

```text
                 [INPUT: 32x32x3]
                        │
                        ▼
       ┌─────────────────────────────────┐
       │             ENCODER             │
       ├─────────────────────────────────┤
       │  Conv2d (3->32) + BN + ReLU     │  -> Spatial: 32x32
       │  Conv2d (32->64, stride=2)      │  -> Spatial: 16x16 (Downsampling)
       │  Conv2d (64->128, stride=2)     │  -> Spatial: 8x8   (Downsampling)
       │  Conv2d (128->64) + BN + ReLU   │  -> Latent Dim Bottleneck
       └────────────────┬────────────────┘
                        │
                        ▼
           [BOTTLE-NECK / LATENT SPACE]
                        │
                        ▼
       ┌─────────────────────────────────┐
       │             DECODER             │
       ├─────────────────────────────────┤
       │  Conv2d (64->128) + BN + ReLU   │  -> Processing Latent Code
       │  ConvTrans2d (128->64, s=2)     │  -> Spatial: 16x16 (Upsampling)
       │  ConvTrans2d (64->32, s=2)      │  -> Spatial: 32x32 (Upsampling)
       │  Conv2d (32->3) + Sigmoid       │  -> Final RGB Estimation Out
       └────────────────┬────────────────┘
                        │
                        ▼
                 [OUTPUT: 32x32x3]
```

---

## 📊 Dataset Specifications (CIFAR-10)

| Attribute | Specification |
| :--- | :--- |
| **Total Volume** | 60,000 continuous color images |
| **Resolution & Format** | $32 \times 32$ pixels, RGB (3 channels) |
| **Classes** | 10 categorical labels (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Experimental Splits** | 40,000 Training / 10,000 Validation / 10,000 Testing (80% / 20% validation split) |

---

## 🧪 Experimental Protocols & Hyperparameters

The convolutional model was systematically trained using standard optimization loops across four severe transformation tasks for exactly **30 epochs** each:

| Hyperparameter / Parameter | Configuration Value |
| :--- | :--- |
| **Optimizer** | Adam (`optim.Adam`) |
| **Initial Learning Rate (LR)** | $1 \times 10^{-3}$ (Static Step Gradient) |
| **Loss Function Metric** | Mean Squared Error Loss (`nn.MSELoss`) |
| **Batch Size** | 256 Samples |
| **Maximum Epoch Limit** | 30 Iterations per Model Experiment |
| **Denoising Gaussian Perturbation** | Normal Distribution ($\sigma = 0.1$) |
| **Denoising Salt & Pepper** | Channel Uniform Boolean Noise (Probability = $0.1$) |
| **Inpainting Occlusion Size** | Square Block Erasure Mask ($8 \times 8$ pixels, values set to 0) |
| **Colorization Configuration** | Conversion using standard luminosity weights: $0.2989R + 0.5870G + 0.1140B$ |

---

## 📊 Evaluation Metrics: PSNR Explanation

To rigorously measure image reconstruction quality, we compute the **Peak Signal-to-Noise Ratio (PSNR)** on the test set alongside the standard validation curves. PSNR is an engineering metric defined via the Mean Squared Error (MSE):

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)$$

Where $MAX_I$ is the maximum possible pixel value of the image (since the inputs are normalized to $[0.0, 1.0]$ via a final `nn.Sigmoid` activation layer, $MAX_I = 1.0$). Higher PSNR values indicate superior reconstruction quality, reflecting that the network has successfully learned to invert the synthetic degradations.

---

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Core Framework:** PyTorch & Torchvision
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib
* **Progress Tracking:** TQDM (integrated smoothly for Jupyter Notebook logging)
* **Hardware Acceleration:** NVIDIA GPU + CUDA Toolkit Execution Environment

---

## 🚀 Execution & Replication Guide

#### 1. Clone the Repository
```bash
git clone [https://github.com/NoFtorio/DL-P4-Autoencoder-CIFAR10-Image-Restoration.git](https://github.com/NoFtorio/DL-P4-Autoencoder-CIFAR10-Image-Restoration.git)
cd DL-P4-Autoencoder-CIFAR10-Image-Restoration
```

#### 2. Install Dependencies
```bash
pip install torch torchvision numpy pandas matplotlib tqdm
```

#### 3. Run the Experiment Notebook
Open the file `Practica4_FelipePerezNoeIsrael_DL.ipynb` using Jupyter Notebook or VS Code, select your GPU runtime (CUDA), and execute all cells sequentially to evaluate training paths and log visual metrics.
```bash
jupyter notebook Practica4_FelipePerezNoeIsrael_DL.ipynb
```