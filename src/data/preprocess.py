import os
import sys
sys.path.insert(0, os.path.abspath("."))

import argparse
import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess clinical brain MRI scans (N4 bias correction, RAS reorientation, isotropic resampling, intensity normalization).")
    parser.add_argument("--catalog", type=str, default="./data/dataset_catalog.csv", help="Path to dataset catalog CSV.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--out_dir", type=str, default="./data/processed", help="Directory to save preprocessed scans.")
    parser.add_argument("--subject", type=str, default="", help="Optional single subject ID to preprocess (e.g. sub-01).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes.")
    return parser.parse_args()

def reorient_to_ras(image):
    """Reorient SimpleITK Image to RAS orientation."""
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAS")
    return orienter.Execute(image)

def apply_n4_bias_field_correction(image):
    """Apply N4 Bias Field Correction using SimpleITK (supports 3D volumes and 2D single-slice scans)."""
    input_image = sitk.Cast(image, sitk.sitkFloat32)
    size = input_image.GetSize()

    # Check if 2D single-slice scan (e.g. depth == 1)
    if min(size) == 1:
        try:
            # Slice along 3rd axis
            slice_2d = input_image[:, :, 0]
            mask_2d = sitk.OtsuThreshold(slice_2d, 0, 1, 200)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([50, 50, 30])
            corrected_2d = corrector.Execute(slice_2d, mask_2d)
            corrected = sitk.JoinSeries(corrected_2d)
            corrected.SetSpacing(input_image.GetSpacing())
            corrected.SetOrigin(input_image.GetOrigin())
            corrected.SetDirection(input_image.GetDirection())
            return corrected
        except Exception:
            return input_image

    mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    try:
        corrected = corrector.Execute(input_image, mask_image)
    except Exception:
        try:
            corrected = corrector.Execute(input_image)
        except Exception:
            corrected = input_image
    return corrected

def resample_image(image, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear):
    """Resample image to target isotropic spacing (in mm)."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Compute new image size based on target spacing
    target_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(target_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())

    return resample.Execute(image)

def normalize_intensity(image_np):
    """Z-score normalize intensity based on foreground (non-zero) voxels."""
    mask = image_np > 0
    if not np.any(mask):
        return image_np
    mean_val = np.mean(image_np[mask])
    std_val = np.std(image_np[mask])
    if std_val < 1e-6:
        std_val = 1e-6
    normalized = np.zeros_like(image_np, dtype=np.float32)
    normalized[mask] = (image_np[mask] - mean_val) / std_val
    return normalized

def preprocess_single_scan(input_path, output_path, target_spacing=(1.0, 1.0, 1.0), apply_n4=True, normalize=True):
    """Preprocess a single NIfTI volume end-to-end."""
    if not os.path.exists(input_path):
        return False, f"Input file not found: {input_path}"

    try:
        image = sitk.ReadImage(input_path, sitk.sitkFloat32)

        # Sanitize zero or non-positive spacing values in header (prevents ITK zero-spacing errors)
        spacing = list(image.GetSpacing())
        sanitized = False
        for i in range(len(spacing)):
            if spacing[i] <= 1e-4:
                spacing[i] = 1.0
                sanitized = True
        if sanitized:
            image.SetSpacing(tuple(spacing))

        # 1. Reorient to RAS
        image = reorient_to_ras(image)

        # 2. Apply N4 Bias Field Correction
        if apply_n4:
            image = apply_n4_bias_field_correction(image)

        # 3. Resample to Target Spacing
        image = resample_image(image, target_spacing=target_spacing)

        # 4. Normalize Intensity
        image_np = sitk.GetArrayFromImage(image)
        if normalize:
            image_np = normalize_intensity(image_np)

        # Re-create SimpleITK image with spatial metadata preserved
        out_image = sitk.GetImageFromArray(image_np)
        out_image.SetSpacing(image.GetSpacing())
        out_image.SetOrigin(image.GetOrigin())
        out_image.SetDirection(image.GetDirection())

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(out_image, output_path)
        return True, output_path

    except Exception as e:
        return False, f"Failed {input_path}: {str(e)}"

def process_subject(row, out_base_dir, cfg):
    sub_id = row['subject_id']
    sub_out_dir = os.path.join(out_base_dir, sub_id)
    target_spacing = tuple(cfg['preprocessing']['target_spacing'])
    apply_n4 = cfg['preprocessing']['n4_bias_field_correction']
    normalize = cfg['preprocessing']['normalize_intensity']

    results = []

    modalities_mapping = [
        ('primary_t1', f"{sub_id}_T1w.nii.gz"),
        ('primary_t2', f"{sub_id}_T2w.nii.gz"),
        ('primary_flair', f"{sub_id}_FLAIR.nii.gz")
    ]

    for col, out_name in modalities_mapping:
        src_path = str(row.get(col, '')).strip()
        if src_path and os.path.exists(src_path):
            dst_path = os.path.join(sub_out_dir, out_name)
            success, msg = preprocess_single_scan(
                input_path=src_path,
                output_path=dst_path,
                target_spacing=target_spacing,
                apply_n4=apply_n4,
                normalize=normalize
            )
            results.append((sub_id, col, success, msg))

    return results

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(args.catalog):
        raise FileNotFoundError(f"Catalog file not found at {args.catalog}. Please run ingest_brainlife.py first.")

    df_catalog = pd.read_csv(args.catalog)

    if args.subject:
        df_catalog = df_catalog[df_catalog['subject_id'] == args.subject]
        if len(df_catalog) == 0:
            print(f"Subject {args.subject} not found in catalog.")
            return

    print(f"Starting preprocessing for {len(df_catalog)} subjects...")
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_subject, row, args.out_dir, cfg)
                for _, row in df_catalog.iterrows()
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
                all_results.extend(future.result())
    else:
        for _, row in tqdm(df_catalog.iterrows(), total=len(df_catalog), desc="Preprocessing"):
            all_results.extend(process_subject(row, args.out_dir, cfg))

    successes = [r for r in all_results if r[2]]
    failures = [r for r in all_results if not r[2]]

    print(f"\n✨ Preprocessing Complete!")
    print(f"   Successfully processed: {len(successes)} volumes.")
    if failures:
        print(f"   Failed processing: {len(failures)} volumes.")
        for f in failures:
            print(f"     - {f[0]} ({f[1]}): {f[3]}")

if __name__ == "__main__":
    main()
