# A Methodological Study of AI Robustness Under Real-World Clinical MRI Quality Constraints: Evidence from a Nigerian Brain MRI Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical_Imaging-green.svg)](https://monai.io/)
[![SimpleITK](https://img.shields.io/badge/SimpleITK-Image_Processing-blue.svg)](https://simpleitk.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)

This repository contains the code, evaluation pipelines, and experimental frameworks for **"A Methodological Study of AI Robustness Under Real-World Clinical MRI Quality Constraints: Evidence from a Nigerian Brain MRI Dataset."** 

This study moves beyond standard disease-classification benchmarks to systematically characterize the robustness, calibration, uncertainty, and explainability of machine learning and deep learning models when subjected to real-world clinical MRI quality constraints.

---

## Table of Contents
- [Overview](#overview)
- [Scientific Motivation](#scientific-motivation)
- [What is New in This Study](#what-is-new-in-this-study)
- [Research Questions](#research-questions)
- [Objectives](#objectives)
- [Dataset Specifications](#dataset-specifications)
- [Experimental Design](#experimental-design)
- [Methodology](#methodology)
  - [Image Quality Assessment (IQA)](#image-quality-assessment-iqa)
  - [Controlled Image Degradation](#controlled-image-degradation)
  - [Model Architectures](#model-architectures)
  - [Robustness & Uncertainty Evaluation](#robustness--uncertainty-evaluation)
  - [Explainability Stability Analysis](#explainability-stability-analysis)
  - [Statistical Analysis](#statistical-analysis)
- [Expected Outcomes](#expected-outcomes)
- [Scientific Contribution](#scientific-contribution)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Image Quality Profiling](#2-image-quality-profiling)
  - [3. Applying Controlled Degradations](#3-applying-controlled-degradations)
  - [4. Model Training](#4-model-training)
  - [5. Robustness & Explainability Evaluation](#5-robustness--explainability-evaluation)
- [Citation](#citation)
- [License](#license)

---

## Overview

Artificial intelligence (AI) models in neuroimaging are predominantly developed using high-quality research datasets acquired under carefully standardized imaging protocols (e.g., ADNI, UK Biobank, Human Connectome Project, and OASIS). However, these datasets do not represent the imaging conditions encountered in routine clinical practice, particularly in low- and middle-income countries (LMICs).

Clinical MRI in resource-constrained environments commonly exhibits:
- **Lower magnetic field strength** (e.g., 1.5T and sub-Tesla scanners)
- **Heterogeneous acquisition protocols** across different scanner manufacturers
- **Variable slice orientations** and slice thicknesses
- **Reduced spatial resolution** to minimize scan acquisition times
- **Clinical artifacts** such as patient motion, blur, and flow artifacts
- **Scanner-dependent intensity variations** and bias fields

These variations create a substantial **domain gap** between research-grade MRI and clinical reality. Rather than focusing solely on maximizing classification accuracy, this study characterizes how different AI architectures behave when trained and evaluated under progressively degraded, realistic imaging conditions. We evaluate **robustness, stability, calibration, uncertainty, and explainability** to determine which modeling paradigms remain reliable when clinical image quality deteriorates.

---

## Scientific Motivation

Medical AI literature overwhelmingly evaluates algorithms using carefully curated research datasets. Clinical deployment, however, occurs in environments where MRI quality is substantially lower. This mismatch contributes to poor generalization, unreliable predictions, and limited adoption of AI systems in routine clinical workflows.

Africa represents one of the largest gaps in this literature. Most published neuroimaging AI studies do not evaluate:
1. Low-field MRI scans.
2. Highly heterogeneous scanner configurations.
3. Resource-limited clinical environments.

This study directly addresses this gap by utilizing a unique, open dataset of clinical brain MRI scans from Nigeria to analyze model resilience under clinical reality.

---

## What is New in This Study

Unlike standard disease-classification benchmarks, this study introduces a controlled methodological investigation into AI robustness under real-world clinical imaging constraints:

1. **Real-World African Clinical MRI Benchmark**: Evaluation is performed on a heterogeneous Nigerian clinical MRI dataset containing multiple structural modalities, variable acquisition protocols, and diverse disease groups.
2. **Robustness-Centric Evaluation**: Metrics such as model calibration (Expected Calibration Error), predictive uncertainty, performance degradation curves, and explainability stability are treated as primary outcomes, rather than secondary to accuracy.
3. **Progressive Image Degradation Framework**: Images are systematically degraded through controlled perturbations (Gaussian blur, Gaussian noise, motion artifacts, reduced spatial resolution, and intensity non-uniformity) to map exact failure modes.
4. **Image Quality–Aware AI Evaluation**: Objective MRI quality metrics are incorporated directly into the model evaluation. Performance is analyzed as a function of image quality, rather than treating quality as a nuisance variable.
5. **Explainability Stability**: We investigate whether model explanations (e.g., Grad-CAM, Integrated Gradients) remain stable and anatomically consistent as image quality decreases.
6. **Clinical Translation**: The study evaluates AI under conditions representative of hospitals in low-resource healthcare systems rather than idealized research labs.

---

## Research Questions

1. **How robust are different AI architectures to reduced MRI image quality?**
2. **Which model family (Classical Radiomics vs. CNNs vs. ViTs vs. Medical Foundation Models) exhibits the greatest resilience under low-field clinical imaging?**
3. **How does predictive uncertainty and calibration change as image quality deteriorates?**
4. **Do explainability maps remain anatomically consistent under degraded imaging, or do they shift focus to artifact patterns?**
5. **Can quality-aware learning (incorporating image quality metrics as metadata) improve robustness?**
6. **Which objective image quality metrics (IQMs) best predict AI prediction failure?**

---

## Objectives

### Primary Objectives
- Evaluate and compare the robustness of classical machine learning (Radiomics-based) and deep learning models.
- Quantify performance degradation under realistic clinical image artifacts.
- Measure calibration error and predictive uncertainty under out-of-distribution quality domains.
- Evaluate the spatial stability of explainability maps across progressive degradation levels.

### Secondary Objectives
- Characterize mathematical relationships between MRI quality metrics and AI performance.
- Identify actionable quality thresholds beyond which diagnostic AI reliability substantially decreases.

---

## Dataset Specifications

Experiments are conducted on the **Nigerian Brain MRI Dataset**:

| Parameter | Specification |
| :--- | :--- |
| **Subjects** | 88 subjects |
| **Total Scans** | 787 structural images |
| **Clinical Labels** | Healthy Controls, Dementia, Parkinson's Disease |
| **Structural Modalities** | T1-weighted (T1w), T2-weighted (T2w), FLAIR |
| **Scanners** | 1.5T and lower-field clinical scanners |
| **Acquisitions** | Axial, Coronal, Sagittal orientations |
| **Heterogeneity** | Variable acquisition protocols, multiple runs for some subjects, heterogeneous image quality, and demographic metadata (age, sex, education, socioeconomic status, geopolitical region). |

---

## Experimental Design

The study runs a parallel evaluation pipeline comparing three distinct modeling paradigms:

```
                      ┌───────────────────────────────────────┐
                      │      Nigerian Clinical MRI Dataset    │
                      └───────────────────┬───────────────────┘
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                 ▼
           ┌───────────────────────────┐     ┌───────────────────────────┐
           │   Radiomics Pipeline      │     │  Deep Learning Pipeline   │
           │ (SimpleITK + PyRadiomics) │     │     (PyTorch & MONAI)     │
           └─────────────┬─────────────┘     └─────────────┬─────────────┘
                         │                                 │
         ┌───────────────┼───────────────┐        ┌────────┼────────┐
         ▼               ▼               ▼        ▼        ▼        ▼
    ┌─────────┐    ┌───────────┐    ┌─────────┐ ┌───┐    ┌───┐    ┌───┐
    │Logistic │    │  Random   │    │ XGBoost │ │2D │    │ViT│    │FM │
    │ Regres. │    │  Forest   │    │         │ │CNN│    │   │    │   │
    └─────────┘    └───────────┘    └─────────┘ └───┘    └───┘    └───┘
```

---

## Methodology

### Image Quality Assessment (IQA)

We extract objective Image Quality Metrics (IQMs) using a pipeline inspired by MRIQC. These metrics quantify spatial and statistical properties of the images:

- **Signal-to-Noise Ratio (SNR)**:
  $$\text{SNR} = \frac{\mu_{\text{foreground}}}{\sigma_{\text{background}}}$$
  Measures the relative strength of the diagnostic signal to the background noise.
- **Contrast-to-Noise Ratio (CNR)**:
  $$\text{CNR} = \frac{|\mu_{\text{gray matter}} - \mu_{\text{white matter}}|}{\sigma_{\text{background}}}$$
  Quantifies the distinctness of different tissue classes.
- **Entropy Focus Criterion (EFC)**:
  $$E = -\sum_{i} p_i \ln(p_i)$$
  Measures the focus of the image; ghosting and motion blur increase the entropy of the voxel intensity distribution.
- **Foreground-Background Energy Ratio (FBER)**:
  $$\text{FBER} = \frac{\text{Mean Energy}_{\text{foreground}}}{\text{Mean Energy}_{\text{background}}}$$
- **Full Width at Half Maximum (FWHM)**: Estimates the effective spatial resolution and blur.
- **Intensity Non-uniformity (INU)**: Measures the spatial variation of the bias field (field inhomogeneity).

Each scan receives an empirical **composite quality score** derived from these metrics.

### Controlled Image Degradation

To characterize failure modes, we systematically apply five levels of controlled degradation (perturbations) using SimpleITK and PyTorch:

```
[Level 0: Original] 
       │
       ├──► [Level 1: Mild Blur] ───────► Gaussian kernel σ = 1.0 mm
       │
       ├──► [Level 2: Moderate Blur] ──► Gaussian kernel σ = 2.0 mm
       │
       ├──► [Level 3: Gaussian Noise] ──► Additive noise, σ_noise = 0.05
       │
       ├──► [Level 4: Reduced Res] ─────► Downsampling slice-select dimension (factor of 2)
       │
       └──► [Level 5: Combined] ────────► Noise + Blur + Sinusoidal Motion Artifacts
```

### Model Architectures

We benchmark three paradigm classes:

1. **Classical Machine Learning**: 
   - **Features**: Radiomics features (shape, first-order statistics, GLCM, GLRLM, GLSZM) extracted using PyRadiomics.
   - **Classifiers**: Logistic Regression (L2-regularized), Random Forest, and Gradient Boosted Trees (XGBoost).
2. **Standard Deep Learning (2D/3D CNNs)**:
   - **ResNet-18** (parameter-efficient baseline)
   - **DenseNet-121** (dense feature-reuse architecture)
   - **EfficientNet-B2** (compound-scaled architecture)
3. **Transformer-based & Medical Foundation Models**:
   - **Vision Transformer (ViT-B/16)** (patch-based self-attention)
   - **MONAI Foundation Model** (ViT backbone pretrained on large-scale clinical neuroimaging via self-supervised learning).

### Robustness & Uncertainty Evaluation

- **Robustness Curves**: Model performance (F1, ROC-AUC) is plotted against degradation levels 0 to 5.
- **Relative Robustness Index (RRI)**:
  $$\text{RRI} = \frac{\text{AUC}_{\text{degraded}}}{\text{AUC}_{\text{baseline}}}$$
- **Calibration Assessment**: 
  - **Expected Calibration Error (ECE)**: Measures the discrepancy between predictive confidence and empirical accuracy.
  - **Brier Score**: Evaluates the mean squared error of forecast probabilities.
- **Uncertainty Estimation**: 
  - **Monte-Carlo Dropout**: Estimating epistemic uncertainty during inference.
  - **Deep Ensembles**: Measuring variance across models trained with different initializations.

### Explainability Stability Analysis

We analyze if models make predictions using relevant anatomy or noise artifacts:
- **Attribution Methods**: Grad-CAM (CNNs), Attention Rollout (ViTs), Integrated Gradients, and Occlusion Sensitivity.
- **Evaluation Metrics**:
  - **Localization Consistency**: Intersection over Union (IoU) of top 10% attribution salience maps with clinical ROIs (e.g., hippocampus for Dementia).
  - **Cosine Similarity & Pearson Correlation**: Spatial correlation of attributions across degradation levels $L_0 \rightarrow L_i$.
  - **Attribution Entropy**: Quantifies whether attributions become overly diffuse under noise.

### Statistical Analysis

To ensure reproducibility and statistical validity:
- **Repeated Cross-Validation**: 5-fold stratified subject-level cross-validation repeated 5 times.
- **Confidence Intervals**: 95% bootstrap confidence intervals computed over 1000 resamples.
- **Hypothesis Testing**:
  - **McNemar’s Test**: Compare paired classification proportions between models.
  - **DeLong’s Test**: Statistical comparison of ROC-AUC curves.
- **Mixed-Effects Regression**: Modeling performance degradation as a function of continuous IQMs, treating subject IDs as random effects.

---

## Expected Outcomes

We hypothesize that:
1. **Radiomics Stability**: Classical radiomics models with tree-based classifiers may exhibit greater initial stability than large-capacity deep learning models on low-quality clinical MRI due to lower parameter complexity.
2. **Deep Learning Degradation**: Deep learning performance will degrade more rapidly under out-of-distribution noise, though quality-aware training (e.g., adding IQA metrics as auxiliary inputs or data augmentation) will mitigate this effect.
3. **Foundation Model Generalization**: Medical foundation models pretrained on large-scale clinical datasets will show significantly higher robustness compared to models trained from scratch or pretrained on ImageNet.
4. **Explainability Drift**: Explainability maps will become unstable and shift focus from diagnostic anatomical ROIs to artifact patterns before classification accuracy drops, highlighting that explainability is a leading indicator of model failure.
5. **Quality Thresholds**: Strong correlations between objective MRI quality metrics (like SNR and EFC) and model error will allow us to define clear minimum quality thresholds for clinical deployment.

---

## Scientific Contribution

This study contributes:
- One of the first methodological evaluations of AI robustness on African clinical neuroimaging data.
- A quality-aware benchmark for structural brain MRI.
- A reproducible framework for robustness testing under realistic clinical image degradation.
- Evidence to inform the safe deployment of medical AI in resource-limited healthcare environments.
- Practical guidelines on how image quality influences diagnostic AI performance.

---

## Repository Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── config/
│   ├── config.yaml              # Global training and evaluation parameters
│   └── degradation.yaml         # Controlled perturbation parameters (blur, noise, resampling)
├── data/
│   ├── raw/                     # Unaltered Nigerian Brain MRI Dataset
│   └── processed/               # Preprocessed, registered, and degraded scans
├── notebooks/
│   ├── 01_eda_and_iqa.ipynb     # Exploratory analysis and baseline quality metrics
│   └── 02_robustness_curves.ipynb# Visualization of results and degradation curves
├── scripts/
│   ├── setup_env.sh             # Dependency installer
│   └── run_pipeline.sh          # Orchestrates the end-to-end pipeline
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── loader.py            # Custom PyTorch Dataset/DataLoader (T1, T2, FLAIR)
    │   └── preprocess.py        # Spacing resampling, orientation normalization, N4 bias correction
    ├── degradation/
    │   ├── __init__.py
    │   ├── perturbations.py     # SimpleITK/PyTorch image degradation transforms
    │   └── pipeline.py          # Script to generate degraded image sets
    ├── models/
    │   ├── __init__.py
    │   ├── baseline.py          # PyRadiomics extractor & ML classifiers (XGBoost, RF, LR)
    │   ├── cnn.py               # Deep Learning architectures (ResNet, DenseNet, EfficientNet)
    │   └── monai_fm.py          # ViT / MONAI foundation model wrapper
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py           # IQMs, ECE, Brier Score, and RRI metrics
    │   └── explain.py           # Captum-based Grad-CAM, IG, and attention attribution
    └── utils/
        ├── __init__.py
        ├── stats.py             # Mixed-effects models, DeLong test, McNemar
        └── viz.py               # Generation of saliency maps, curves, and calibration plots
```

---

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended for Deep Learning models)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/clinical-mri-ai-robustness.git
   cd clinical-mri-ai-robustness
   ```

2. Run the environment setup script (or install manually):
   ```bash
   chmod +x scripts/setup_env.sh
   ./scripts/setup_env.sh
   ```

   *Alternatively, using pip:*
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Preprocessing
Standardize raw scans (reorientation to LAS/RAS, N4 bias field correction, and isotropic voxel resampling):
```bash
python src/data/preprocess.py \
    --data_dir ./data/raw \
    --out_dir ./data/processed \
    --resample_spacing 1.0
```

### 2. Image Quality Profiling
Generate baseline objective quality scores across the dataset:
```bash
python src/evaluation/metrics.py \
    --profile_iqa \
    --data_dir ./data/processed \
    --out_file ./data/processed/iqa_metrics.csv
```

### 3. Applying Controlled Degradations
Generate the 5 progressive levels of degraded dataset variants:
```bash
python src/degradation/pipeline.py \
    --input_dir ./data/processed \
    --output_dir ./data/processed/degraded \
    --config config/degradation.yaml
```

### 4. Model Training
Train ML or DL models on clean baseline data or with quality-aware augmentations:
* **Train XGBoost baseline on Radiomics features:**
  ```bash
  python src/models/baseline.py \
      --train \
      --data_dir ./data/processed \
      --model xgboost \
      --out_dir ./checkpoints/xgboost/
  ```
* **Train DenseNet-121 using PyTorch:**
  ```bash
  python src/models/cnn.py \
      --train \
      --data_dir ./data/processed \
      --model densenet121 \
      --epochs 80 \
      --batch_size 16 \
      --out_dir ./checkpoints/densenet/
  ```

### 5. Robustness & Explainability Evaluation
Evaluate model performance, calibration, and explainability stability across all degradation levels:
```bash
python src/evaluation/metrics.py \
    --evaluate_robustness \
    --model_dir ./checkpoints/ \
    --data_dir ./data/processed/degraded \
    --out_dir ./results/
```
Generate and plot explainability attributions (e.g., Grad-CAM vs. Integrated Gradients):
```bash
python src/evaluation/explain.py \
    --model_path ./checkpoints/densenet/densenet121_best.pth \
    --image_path ./data/processed/degraded/L3/sub-001_T1w.nii.gz \
    --method gradcam \
    --out_dir ./results/explanations/
```

---

## Citation

If you use this repository or dataset in your research, please cite:

```bibtex
@article{clinical_mri_robustness_nigerian2026,
  title={A Methodological Study of AI Robustness Under Real-World Clinical MRI Quality Constraints: Evidence from a Nigerian Brain MRI Dataset},
  author={Author, A. and Author, B. and Author, C.},
  journal={Journal of Medical Image Analysis},
  year={2026},
  volume={XX},
  pages={XX-XX},
  doi={10.1016/j.media.2026.xxxxxx}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# clinical_mri_ai_robustness
