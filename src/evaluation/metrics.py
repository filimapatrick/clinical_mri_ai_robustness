import os
import sys
sys.path.insert(0, os.path.abspath("."))

import argparse
import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

def calculate_snr(image_np, foreground_threshold=0.1):
    """Calculate Signal-to-Noise Ratio (SNR) in dB."""
    fg_mask = image_np > foreground_threshold
    bg_mask = ~fg_mask
    if not np.any(fg_mask) or not np.any(bg_mask):
        return 0.0
    mean_signal = np.mean(image_np[fg_mask])
    std_noise = np.std(image_np[bg_mask])
    if std_noise < 1e-6:
        std_noise = 1e-6
    return float(20 * np.log10(mean_signal / std_noise))

def calculate_cnr(image_np):
    """Calculate Contrast-to-Noise Ratio (CNR) between upper and lower intensity quantiles."""
    fg = image_np[image_np > 0]
    if len(fg) == 0:
        return 0.0
    q75 = np.percentile(fg, 75)
    q25 = np.percentile(fg, 25)
    bg_std = np.std(image_np[image_np <= 0])
    if bg_std < 1e-6:
        bg_std = 1e-6
    return float((q75 - q25) / bg_std)

def calculate_efc(image_np):
    """Calculate Entropy Focus Criterion (EFC) for image blur/ghosting assessment."""
    pos_val = np.abs(image_np)
    max_val = np.sqrt(np.sum(pos_val**2))
    if max_val < 1e-6:
        return 0.0
    norm_val = pos_val / max_val
    # Voxel entropy
    ent = -np.sum(norm_val * np.log(norm_val + 1e-12))
    n_voxels = np.prod(image_np.shape)
    max_ent = np.log(1.0 / np.sqrt(n_voxels))
    return float(ent / max_ent)

def calculate_expected_calibration_error(y_true, probs, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    probs: (N, C) predicted class probabilities or (N,) for binary confidence.
    """
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        confidences = probs
        accuracies = y_true.astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)

def calculate_brier_score(y_true, probs, num_classes=3):
    """Calculate Multi-Class Brier Score."""
    if num_classes == 2 and probs.ndim == 1:
        return float(brier_score_loss(y_true, probs))
    
    # One-hot encode y_true
    y_onehot = np.zeros_like(probs)
    for i, label in enumerate(y_true):
        y_onehot[i, label] = 1.0
    
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

def profile_dataset_iqa(processed_dir, output_csv):
    """Profile objective Image Quality Metrics (IQMs) across preprocessed dataset."""
    records = []
    for root, dirs, files in os.walk(processed_dir):
        for f in files:
            if f.endswith(".nii.gz") and not f.startswith("._"):
                full_path = os.path.join(root, f)
                try:
                    img = sitk.ReadImage(full_path, sitk.sitkFloat32)
                    arr = sitk.GetArrayFromImage(img)
                    snr = calculate_snr(arr)
                    cnr = calculate_cnr(arr)
                    efc = calculate_efc(arr)

                    sub_id = os.path.basename(os.path.dirname(full_path))
                    records.append({
                        "file": f,
                        "subject_id": sub_id,
                        "snr_db": snr,
                        "cnr": cnr,
                        "efc": efc
                    })
                except Exception as e:
                    print(f"Error profiling {f}: {e}")

    df_iqa = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_iqa.to_csv(output_csv, index=False)
    print(f"✅ IQA profiling complete. Results saved to {output_csv}")
    return df_iqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_iqa", action="store_true", help="Profile objective IQMs for dataset.")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--out_file", type=str, default="./results/iqa_metrics.csv")
    args = parser.parse_args()

    if args.profile_iqa:
        profile_dataset_iqa(args.data_dir, args.out_file)
