# Adversarial Attacks on Traffic Sign Classification

This project explores the vulnerability of deep learning models to adversarial attacks in the context of traffic sign recognition. It trains convolutional neural networks (VGGNet and LeNet) on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html) dataset, generates adversarial examples using the **Fast Gradient Sign Method (FGSM)**, and evaluates the effectiveness of adversarial training as a defense mechanism.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Attack Method](#attack-method)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Overview

Traffic sign recognition is a safety-critical component of autonomous driving systems. This project demonstrates how small, often imperceptible perturbations (adversarial examples) can cause deep learning classifiers to misclassify traffic signs. The workflow is:

1. **Train** CNN models (VGGNet and LeNet) on the GTSRB dataset (43 traffic sign classes).
2. **Generate** adversarial examples using FGSM at multiple perturbation strengths (epsilon values).
3. **Evaluate** model accuracy on both clean and adversarial test images.
4. **Defend** by retraining models on a mixture of clean and adversarial images (adversarial training) and measuring the improvement in robustness.

## Repository Structure

```
AdversarialAttacks/
├── Generate_Adversarial_Samples.py        # Generate FGSM adversarial examples
├── traffic-signs-image-classification.py  # Train models and evaluate robustness
├── README.md
└── .gitattributes
```

| File | Description |
|------|-------------|
| `traffic-signs-image-classification.py` | Loads and preprocesses the GTSRB dataset, builds and trains VGGNet and LeNet models, evaluates accuracy on clean and adversarial test sets, and performs adversarial training. |
| `Generate_Adversarial_Samples.py` | Loads a pre-trained VGGNet model, generates FGSM adversarial examples at multiple epsilon values, visualizes original vs. adversarial images, and saves the adversarial dataset with metadata to CSV files. |

## Models

### VGGNet (Custom Variant)

A deep CNN inspired by VGGNet with the following architecture:

- **Convolutional blocks**: 32 → 32 → 64 → 64 → 128 → 128 → 256 → 256 filters (3×3 kernels)
- **Pooling**: MaxPooling2D after every two convolutional layers
- **Regularization**: Dropout (0.3–0.5) and BatchNormalization
- **Output**: Dense layer with 43 units (softmax)

### LeNet (Variant)

A lighter model based on the LeNet-5 architecture:

- Two convolutional layers with 5×5 kernels
- MaxPooling after each convolutional layer
- Fully connected layers leading to 43-class softmax output

Both models are trained with the Adam optimizer, categorical cross-entropy loss, and an EarlyStopping callback (patience = 5). Trained models are saved as `.h5` files.

## Attack Method

### Fast Gradient Sign Method (FGSM)

FGSM crafts adversarial examples by adding a small perturbation in the direction of the gradient of the loss with respect to the input image:

```text
x_adv = x + ε · sign(∇ₓ J(θ, x, y))
```

Where:
- **x** is the original input image
- **ε (epsilon)** controls the perturbation magnitude
- **∇ₓ J** is the gradient of the loss function with respect to the input
- **sign(·)** takes the element-wise sign of the gradient

The project evaluates four epsilon values: **0.01**, **0.10**, **0.15**, and **0.20**.

## Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**

- **Classes**: 43 types of traffic signs (speed limits, warnings, prohibitions, etc.)
- **Image size**: Resized to 32×32 pixels (RGB)
- **Split**: 80/20 train/validation from the training set; separate test set provided
- **Format**: Images organized in directories by class, with CSV metadata (`Train.csv`, `Test.csv`)

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) or the [official benchmark site](https://benchmark.ini.rub.de/gtsrb_dataset.html).

## Requirements

- Python 3.7+
- TensorFlow 2.x (with GPU support recommended)
- NumPy
- Pandas
- Pillow (PIL)
- OpenCV (`cv2`)
- scikit-learn
- Matplotlib
- Seaborn
- Foolbox

Install the dependencies with:

```bash
pip install tensorflow numpy pandas pillow opencv-python scikit-learn matplotlib seaborn foolbox
```

## Usage

### 1. Prepare the Dataset

Download the GTSRB dataset and update the data paths in both scripts to point to your local copy. Look for path variables near the top of each file and adjust them accordingly.

### 2. Train the Models

```bash
python traffic-signs-image-classification.py
```

This script will:
- Load and preprocess the training images
- Train both VGGNet and LeNet models
- Save trained models as `vggnet.h5` and `lenet.h5`
- Plot training/validation accuracy and loss curves

### 3. Generate Adversarial Samples

```bash
python Generate_Adversarial_Samples.py
```

This script will:
- Load the pre-trained VGGNet model (`vggnet.h5`)
- Generate FGSM adversarial examples for 1,500 test images
- Save adversarial images and metadata (original label, predicted label, confidence) to CSV files for epsilon values 0.01, 0.10, and 0.15

### 4. Evaluate Robustness

The classification script also includes a section that:
- Loads the generated adversarial samples
- Retrains models on a combination of clean and adversarial images
- Evaluates and compares accuracy on clean vs. adversarial test sets

## Results

The project produces:
- **Training curves**: Accuracy and loss plots for VGGNet and LeNet
- **Adversarial visualizations**: Side-by-side comparisons of original and adversarial images with predicted labels and confidence scores
- **Accuracy metrics**: Classification accuracy on clean and adversarial test sets, before and after adversarial training
- **CSV logs**: For each epsilon value, a CSV file containing filenames, true labels, original predictions, adversarial predictions, and confidence scores

Key observations:
- Higher epsilon values produce stronger attacks but with more visible perturbations
- Adversarial training improves robustness against FGSM attacks while maintaining reasonable accuracy on clean images

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572). ICLR 2015.
- Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). [Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition](https://www.sciencedirect.com/science/article/pii/S0893608012000457). Neural Networks, 32, 323–332.
- [Foolbox: A Python toolbox to benchmark the robustness of machine learning models](https://github.com/bethgelab/foolbox)
