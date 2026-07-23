# Beyond Research-Grade MRI: A Methodological Study of AI Robustness on a Nigerian Clinical Brain MRI Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical_Imaging-green.svg)](https://monai.io/)
[![SimpleITK](https://img.shields.io/badge/SimpleITK-Image_Processing-blue.svg)](https://simpleitk.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)

> **Core Thesis:** Most neuroimaging AI systems are developed and validated on highly curated, research-grade MRI datasets. Their behavior on heterogeneous, lower-quality clinical MRI remains poorly understood, particularly in low-resource healthcare settings. This study systematically characterizes model robustness, calibration, uncertainty, and explainability using a real-world Nigerian clinical MRI dataset.

---

## Conceptual Workflow

The experimental framework of the study is structured as follows:

```mermaid
flowchart TD
    A[Nigerian Brain MRI Dataset <br/> 88 Subjects, 787 Scans <br/> Heterogeneous Sequence Profiles] --> B[Image Quality Assessment <br/> Objective IQMs: SNR, CNR, EFC, FWHM]
    B --> C[Controlled Progressive Degradation <br/> Blur, Noise, Resampling, Motion]
    C --> D{Modeling Pipelines <br/> Subject-Level Classification}
    D --> E[Radiomics Pipeline <br/> PyRadiomics + Random Forest]
    D --> F[Deep Learning Baseline <br/> ResNet-18]
    D --> G[Deep Learning Specialized <br/> DenseNet-121]
    E & F & G --> H[Multi-Task & Multi-Sequence Robustness Evaluation]
    H --> I[Robustness Analysis <br/> Performance - Calibration - Uncertainty - Explainability]
```

---

## Table of Contents
- [Research Questions](#research-questions)
- [Objectives](#objectives)
- [Dataset Specifications](#dataset-specifications)
  - [Modality Availability Statistics](#modality-availability-statistics)
- [Experimental Design](#experimental-design)
- [Methodology Overview](#methodology-overview)
- [Expected Outcomes](#expected-outcomes)
- [Limitations and Scope](#limitations-and-scope)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Research Questions

1. **How robust are AI models across three clinically relevant neurological groups under real-world MRI quality constraints?**
2. **Which disease pair (Control vs. Dementia, Control vs. Parkinson's, or Dementia vs. Parkinson's) is most difficult to separate under progressive quality degradation?**
3. **Does image quality affect Parkinson's detection differently than dementia detection?**
4. **Which structural MRI sequence (T1, T2, or FLAIR) contributes most to classification robustness?**
5. **How does incomplete modality coverage (missing sequences) affect model robustness and generalization in multi-sequence configurations?**

---

## Objectives

### Primary Objectives
- Benchmark the robustness of classical machine learning (Radiomics-based) and deep learning models across multiple clinical tasks.
- Quantify performance degradation curves under controlled clinical image artifacts.
- Measure calibration error (Expected Calibration Error) and predictive uncertainty under out-of-distribution quality domains.
- Evaluate the stability of explainability maps (localization consistency) under progressive degradation.

### Secondary Objectives
- Characterize the mathematical correlation between objective MRI quality metrics and classification reliability.
- Evaluate the impact of missing sequences (modality dropout) on multi-modal classification robustness.
- Identify actionable quality thresholds beyond which diagnostic AI reliability substantially decreases.

---

## Dataset Specifications

Experiments are conducted on the **Nigerian Brain MRI Dataset**, reflecting a highly heterogeneous clinical environment:

- **Subjects**: 88 subjects (Control: 33 [37.5%], Dementia: 33 [37.5%], Parkinson's Disease: 22 [25.0%]).
- **Total Scans**: 787 structural images (T1w, T2w, FLAIR) acquired using 1.5T and sub-Tesla clinical scanners.
- **Heterogeneity**: Subjects do not have complete coverage across all sequences. Many subjects have only one or two modalities, with variable orientations (Axial, Coronal, Sagittal) and multiple runs.

### Modality Availability Statistics

To establish statistical feasibility for multi-sequence combinations, the sequence availability across the 88 subjects is outlined below:

| Sequence / Combination | Subjects Available | Percentage |
| :--- | :--- | :--- |
| **T1w** | 88 | 100.0% |
| **T2w** | 82 | 93.2% |
| **FLAIR** | 44 | 50.0% |
| **T1w + T2w** | 82 | 93.2% |
| **T1w + FLAIR** | 44 | 50.0% |
| **T2w + FLAIR** | 44 | 50.0% |
| **T1w + T2w + FLAIR** | 44 | 50.0% |

*Note: Exact available subject counts are verified during ingestion validation. The high frequency of missing FLAIR sequences makes evaluating missing-modality resilience highly relevant.*

---

## Experimental Design

The study is designed around two primary axes: multi-task classification and missing-modality robustness.

### 1. Multi-Task Classification Setup
Models are trained and evaluated across four distinct tasks:
* **Task 1: Control vs. Dementia** — Separation of normal aging from cognitive decline.
* **Task 2: Control vs. Parkinson's** — Sensitivity to subcortical Parkinsonian structural alterations.
* **Task 3: Dementia vs. Parkinson's** — Differential diagnostics (the most clinically significant binary task).
* **Task 4 (Primary): Three-Class Classification** — Simultaneous categorization of Control vs. Dementia vs. Parkinson's.

### 2. Sequence Availability and Missingness Analysis
Since clinical protocols do not always enforce acquisition of all sequences, we structure experiments to accommodate missing modalities:
* **Single-Modality Benchmarks**: Separate models trained on T1-only, T2-only, and FLAIR-only subsets.
* **Available-Modality Baseline**: Subject-level predictions generated using whatever sequence combinations are present.
* **Paired-Subset Analysis**: Evaluations restricted to subjects who share identical modality pairs (e.g., evaluating only on the 44 subjects with complete T1+T2+FLAIR).
* **Missing-Modality Robustness**: Test how model performance degrades when one sequence is deliberately omitted (modality dropout test).

### 3. Subject-Level Unit of Analysis
Because subjects have variable scan counts, we designate **Subject-level classification** as the core evaluation unit:
- Predictions or features are aggregated per subject rather than treating each image independently.
- This prevents data leakage (same subject's images in both training and test splits) and aligns with clinical utility.

---

## Methodology Overview

### 1. Image Quality Assessment (IQA)
We extract objective Image Quality Metrics (IQMs) to characterize spatial and statistical property changes (MRIQC protocol; Esteban et al., 2017):
- **Signal-to-Noise Ratio (SNR)**: Strength of diagnostic signal relative to background noise.
- **Contrast-to-Noise Ratio (CNR)**: Distinctness of tissue classes (gray matter vs. white matter).
- **Entropy Focus Criterion (EFC)**: Focus index; ghosting and motion blur increase voxel intensity entropy (Atkinson et al., 1997).
- **Full Width at Half Maximum (FWHM)**: Estimates effective spatial resolution and blur.

### 2. Controlled Progressive Image Degradation
Inspired by the corruption benchmarking philosophy introduced by **Hendrycks & Dietterich (2019)**, we systematically apply four progressive levels of clinical degradation ($L0$–$L3$) grounded in MRI acquisition physics:
* **Gaussian Blur**: Computational approximation of reduced effective spatial resolution resulting from hardware limits or reconstruction smoothing.
* **Rician Noise**: Signal-dependent noise added to complex channels before magnitude extraction ($|I + N(0, \sigma^2)|$), physically reflecting MRI magnitude signal physics (Gudbjartsson & Patz, 1995).
* **Anisotropic Spatial Resolution Reduction**: Out-of-plane voxel downsampling ($z$-spacing: 2.0mm – 5.0mm) and re-interpolation to simulate thick-slice clinical acquisition constraints (Dufumier et al., 2021; Wood et al., 2022).
* **Representative Motion Artifacts**: k-space phase perturbations ($S'(\mathbf{k}) = S(\mathbf{k}) \cdot e^{i \Delta \phi}$) generating representative motion ghosting (Shaw et al., 2020; Küstner et al., 2018).
* **Composite Degradation ($L3$ Composite)**: Simultaneous combination of severe blur, Rician noise, slice thickness downsampling, and motion to evaluate performance under severely compromised clinical scan conditions.

### 3. Model Architectures
We compare representative paradigm classes:
- **Classical ML**: Radiomics features (shape, first-order statistics, GLCM, GLRLM, GLSZM) extracted using PyRadiomics under IBSI standards (Zwanenburg et al., 2020), classified using Random Forest and XGBoost.
- **Standard Deep Learning (CNNs)**: ResNet-18 (parameter-efficient baseline) and DenseNet-121 (dense feature-reuse architecture).

### 4. Robustness & Uncertainty Metrics
- **Relative Robustness Index (RRI)**: Summary metric quantifying performance retention relative to baseline clean performance ($RRI = \frac{1}{K} \sum_{i=1}^{K} \frac{F1_{\text{degraded}, i}}{F1_{\text{baseline}}}$).
- **Area Under the Robustness Curve (AURC)**: Scalar integral of F1 performance retention plotted across degradation severities ($L0 \to L3$).
- **Calibration Assessment**: Expected Calibration Error (ECE; Guo et al., 2017) and Brier Score.
- **Uncertainty Estimation**: Monte-Carlo (MC) Dropout (Gal & Ghahramani, 2016) and Deep Ensembles.
- **Explainability Stability**: Grad-CAM (Selvaraju et al., 2017) localization consistency (IoU with anatomical ROIs) and spatial correlation across degradation levels (Adebayo et al., 2018).

---

## Benchmark Progress & Findings (Phase 3A Radiomics)

### 1. Leak-Free Clean Baseline ($L0$)
Evaluated via 5-Fold Stratified Group Cross-Validation (`StratifiedGroupKFold`) with in-fold feature scaling, variance thresholding, $K=15$ feature selection, and Platt scaling probability calibration:
- **Clean Baseline Macro F1**: **$0.5518 \pm 0.0531$**
- **Clean Balanced Accuracy**: **$0.5846$**
- **Calibrated Expected Calibration Error (ECE)**: **$0.2300$**

### 2. Radiomics Robustness Decay & Relative Robustness Index ($RRI$)
| Corruption Type | Severity Tier | Macro F1 | Relative Robustness Index ($RRI$) | Impact Summary |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Baseline** | **$L0$** | **0.5518** | **1.0000** | Clean reference baseline |
| **Gaussian Blur** | **$L1$** | **0.4019** | **0.7283** | Mild decay (-27.2%) |
| **Gaussian Blur** | **$L2$** | **0.2394** | **0.4338** | Steep decay (-56.6%) |
| **Gaussian Blur** | **$L3$** | **0.2139** | **0.3876** | Floor performance (-61.2%) |
| **Motion Ringing** | **$L1$** | **0.1898** | **0.3439** | Catastrophic drop (-65.6%) |
| **Motion Ringing** | **$L2$** | **0.1898** | **0.3439** | Floor performance saturation |
| **Motion Ringing** | **$L3$** | **0.1898** | **0.3439** | Floor performance saturation |
| **Rician Noise** | **$L1$** | **0.4005** | **0.7257** | Moderate decay (-27.4%) |

*(Note: Noise L2/L3 & Resolution L1-L3 runs actively progressing)*

---

**current Stage and what to do next

Viewed papper.md:126-135
Viewed cnn.py:1-166

Here is the exact setup and commands for you to continue your work for both **Phase 3A** and **Phase 3B**:

---

### 1. 🔄 Phase 3A: Resume Radiomics Degradation Evaluation (starting from `noise_L2`)

Because the script automatically caches completed feature extraction CSVs (`radiomics_blur_L1.csv` ... `radiomics_noise_L1.csv`), re-running this command will **instantly skip all previously finished tiers** and resume right at **`Evaluating Radiomics model on noise (L2)`**:

```bash
cd /Volumes/MyHDD/ABDN_projects/clinical_mri_ai_robustness && ./venv/bin/python scripts/evaluate_radiomics_degradation.py
```

---

### 2. 🚀 Phase 3B: Launch Deep Learning Benchmark (DenseNet-121 & ResNet-18)

To train the 3D Deep Learning architectures ([src/models/cnn.py](file:///Volumes/MyHDD/ABDN_projects/clinical_mri_ai_robustness/src/models/cnn.py)) using PyTorch and MONAI, run the commands below:

#### A. Train 3D DenseNet-121 Benchmark:
```bash
cd /Volumes/MyHDD/ABDN_projects/clinical_mri_ai_robustness && ./venv/bin/python src/models/cnn.py --model densenet121 --task task4 --epochs 50 --batch_size 4 --out_dir ./checkpoints/densenet
```

#### B. Train 3D ResNet-18 Benchmark:
```bash
cd /Volumes/MyHDD/ABDN_projects/clinical_mri_ai_robustness && ./venv/bin/python src/models/cnn.py --model resnet18 --task task4 --epochs 50 --batch_size 4 --out_dir ./checkpoints/resnet
```

---

### 📋 Checklist & State Saved
- [x] All completed empirical results ($L0$ clean, `blur` $L1$–$L3$, `motion` $L1$–$L3$, `noise` $L1$) are saved in [papper.md](file:///Volumes/MyHDD/ABDN_projects/clinical_mri_ai_robustness/papper.md) and [README.md](file:///Volumes/MyHDD/ABDN_projects/clinical_mri_ai_robustness/README.md).
- [x] `src/models/baseline.py` has the emergency mask threshold fallback to prevent PyRadiomics failures under motion/noise.
- [x] `src/models/cnn.py` is configured with subject-level stratified group splitting, MONAI 3D backbones, early stopping checkpointing, and AdamW optimization.


## Expected Outcomes

We hypothesize that:
1. **Radiomics Stability**: Classical radiomics models with tree-based classifiers will exhibit higher baseline robustness under mild perturbations (due to lower parameter capacity) but will scale poorly on the three-class task and differential diagnostics compared to CNNs.
2. **Modality Synergy**: Multimodal combinations will show higher relative robustness (RRI) compared to single-modality pipelines under equivalent degradation levels.
3. **Sequence Sensitivity**: Image quality degradation will affect Parkinson's classification (reliant on specific subcortical structures) more severely than dementia classification (reliant on wider cortical atrophy patterns).
4. **Explainability Drift**: Salience maps will lose anatomical focus and drift towards artifact patterns before a statistical drop in classification accuracy is observed, acting as a leading indicator of failure.

---

## Limitations and Scope

Before training or deploying the codebase, note the following parameters of the study's scope:
- **Structural MRI Only**: Analysis is limited to structural sequences (T1w, T2w, and FLAIR) and does not cover functional MRI (fMRI) or diffusion-weighted imaging (DWI).
- **Clinical MRI Domain**: Designed around low-field (1.5T and below) scanner profiles exhibiting real-world clinical quality variability.
- **Methodological Characterization**: The project's goal is to characterize model failure modes, calibration, and explainability under noise, rather than building a state-of-the-art diagnostic classifier for clinical deployment.
- **Single-Center Cohort**: The dataset represents a single-center cohort from Nigeria; caution should be exercised when generalising findings to other regional healthcare scanners.

---

## Repository Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── config/
│   ├── config.yaml              # Global training and evaluation parameters
│   └── degradation.yaml         # Controlled perturbation parameters
├── data/
│   ├── raw/                     # Unaltered Nigerian Brain MRI Dataset
│   └── processed/               # Preprocessed and degraded scans
├── notebooks/
│   ├── 01_eda_and_iqa.ipynb     # Exploratory analysis and quality metrics
│   └── 02_robustness_curves.ipynb# Result visualization and degradation curves
├── scripts/
│   ├── setup_env.sh             # Dependency installer
│   └── run_pipeline.sh          # Orchestrates the end-to-end pipeline
└── src/
    ├── __init__.py
    ├── data/
    │   ├── loader.py            # Subject-level PyTorch Dataset (T1, T2, FLAIR handling)
    │   └── preprocess.py        # Resampling, reorientation, and N4 bias correction
    ├── degradation/
    │   ├── perturbations.py     # Image degradation transforms (SimpleITK)
    │   └── pipeline.py          # Batch generator of degraded scans
    ├── models/
    │   ├── baseline.py          # Radiomics extractor & ML classifiers (Random Forest, LR)
    │   └── cnn.py               # Deep Learning architectures (ResNet-18, DenseNet-121)
    ├── evaluation/
    │   ├── metrics.py           # IQMs, ECE, Brier Score, and RRI metrics
    │   └── explain.py           # Grad-CAM and Integrated Gradients stability
    └── utils/
        ├── stats.py             # Mixed-effects models, DeLong test, McNemar
        └── viz.py               # Curve generation and saliency heatmaps
```

---

## Installation

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/clinical-mri-ai-robustness.git
   cd clinical-mri-ai-robustness
   ```

2. Setup virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Data Preprocessing
Standardize raw scans (reorientation, N4 bias correction, and resampling):
```bash
python src/data/preprocess.py \
    --data_dir ./data/raw \
    --out_dir ./data/processed \
    --resample_spacing 1.0
```

### 2. Image Quality Profiling
Generate baseline IQA metrics:
```bash
python src/evaluation/metrics.py \
    --profile_iqa \
    --data_dir ./data/processed \
    --out_file ./data/processed/iqa_metrics.csv
```

### 3. Applying Controlled Degradations
Apply blur, noise, resolution reduction, and motion artifacts:
```bash
python src/degradation/pipeline.py \
    --input_dir ./data/processed \
    --output_dir ./data/processed/degraded \
    --config config/degradation.yaml
```

### 4. Model Training
* **Train Random Forest baseline on Radiomics features:**
  ```bash
  python src/models/baseline.py \
      --train \
      --data_dir ./data/processed \
      --model randomforest \
      --out_dir ./checkpoints/rf/
  ```
* **Train DenseNet-121 on available sequences:**
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
Evaluate across all degradation levels:
```bash
python src/evaluation/metrics.py \
    --evaluate_robustness \
    --model_dir ./checkpoints/ \
    --data_dir ./data/processed/degraded \
    --out_dir ./results/
```

Generate attribution maps to assess explainability stability:
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
  title={Beyond Research-Grade MRI: A Methodological Study of AI Robustness on a Nigerian Clinical Brain MRI Dataset},
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
