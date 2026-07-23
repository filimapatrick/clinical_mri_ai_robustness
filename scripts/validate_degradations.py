import os
import sys
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

def compute_snr(img_np):
    fg = img_np[img_np > 0]
    bg = img_np[img_np == 0]
    if len(fg) == 0 or len(bg) == 0:
        bg_std = np.std(img_np[img_np < np.percentile(img_np, 10)]) + 1e-5
        fg_mean = np.mean(fg) if len(fg) > 0 else 1.0
        return float(fg_mean / bg_std)
    bg_std = np.std(bg) + 1e-5
    return float(np.mean(fg) / bg_std)

def compute_efc(img_np):
    # Entropy Focus Criterion
    pos_img = np.abs(img_np)
    max_val = np.max(pos_img) + 1e-5
    norm_img = pos_img / max_val
    b_max = np.sqrt(np.sum(norm_img ** 2)) + 1e-5
    p = (norm_img / b_max) + 1e-12
    p = p[p > 1e-10]
    efc = -np.sum(p * np.log(p))
    return float(efc)

def main():
    degraded_base = "./data/processed/degraded"
    processed_base = "./data/processed"
    
    if not os.path.exists(degraded_base):
        print("Degraded directory not found!")
        return

    records = []
    
    # Process L0 (preprocessed clean scans)
    print("Extracting L0 (Baseline) IQMs...")
    l0_files = [f for f in glob.glob(os.path.join(processed_base, "**", "*.nii.gz"), recursive=True) if "degraded" not in f]
    for f in tqdm(l0_files[:5], desc="L0 Validation"):
        try:
            img = sitk.ReadImage(f, sitk.sitkFloat32)
            arr = sitk.GetArrayFromImage(img)
            records.append({
                "degradation": "baseline",
                "level": "L0",
                "snr": compute_snr(arr),
                "efc": compute_efc(arr)
            })
        except Exception:
            pass

    # Process degraded folders
    folders = [d for d in os.listdir(degraded_base) if os.path.isdir(os.path.join(degraded_base, d))]
    for folder in sorted(folders):
        parts = folder.split("_")
        deg_type = parts[0]
        level = parts[1] if len(parts) > 1 else "L1"
        
        files = glob.glob(os.path.join(degraded_base, folder, "**", "*.nii.gz"), recursive=True)
        for f in files[:5]:
            try:
                img = sitk.ReadImage(f, sitk.sitkFloat32)
                arr = sitk.GetArrayFromImage(img)
                records.append({
                    "degradation": deg_type,
                    "level": level,
                    "snr": compute_snr(arr),
                    "efc": compute_efc(arr)
                })
            except Exception:
                pass

    df = pd.DataFrame(records)
    summary = df.groupby(["degradation", "level"]).agg({
        "snr": ["mean", "std"],
        "efc": ["mean", "std"]
    }).reset_index()
    
    print("\n--- DEGRADATION VALIDATION SUMMARY ---")
    print(summary.to_string())
    
    out_csv = "./data/degradation_validation_summary.csv"
    summary.to_csv(out_csv)
    print(f"\nSaved validation summary to {out_csv}")

if __name__ == "__main__":
    main()
