import os
import sys
sys.path.insert(0, os.path.abspath("."))

import argparse
import yaml
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.degradation.perturbations import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_resolution_reduction,
    apply_motion_artifacts
)

def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate controlled clinical degradation dataset variants.")
    parser.add_argument("--input_dir", type=str, default="./data/processed", help="Input preprocessed scans directory.")
    parser.add_argument("--output_dir", type=str, default="./data/processed/degraded", help="Output directory for degraded volumes.")
    parser.add_argument("--config", type=str, default="config/degradation.yaml", help="Degradation configuration YAML.")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel worker count.")
    return parser.parse_args()

def process_degraded_volume(file_path, rel_path, out_base_dir, deg_type, level_name, params):
    try:
        image = sitk.ReadImage(file_path, sitk.sitkFloat32)

        if deg_type == "blur":
            degraded = apply_gaussian_blur(image, sigma_mm=params)
        elif deg_type == "noise":
            degraded = apply_gaussian_noise(image, noise_level=params)
        elif deg_type == "resolution":
            degraded = apply_resolution_reduction(image, target_spacing=tuple(params))
        elif deg_type == "motion":
            degraded = apply_motion_artifacts(image, frequency=params['frequency'], amplitude=params['amplitude'])
        else:
            return False, f"Unknown degradation type {deg_type}"

        out_dir = os.path.join(out_base_dir, f"{deg_type}_{level_name}", os.path.dirname(rel_path))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(rel_path))
        
        sitk.WriteImage(degraded, out_path)
        return True, out_path

    except Exception as e:
        return False, f"Error processing {file_path} for {deg_type}_{level_name}: {str(e)}"

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        deg_cfg = yaml.safe_load(f)['degradations']

    all_scans = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith(".nii.gz") and not f.startswith("._") and "degraded" not in root:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, args.input_dir)
                all_scans.append((full_path, rel_path))

    print(f"Found {len(all_scans)} preprocessed scans in {args.input_dir}.")

    tasks = []
    for deg_type, deg_info in deg_cfg.items():
        levels = deg_info.get('levels', {})
        for level_name, params in levels.items():
            if level_name == "L0": # Skip baseline zero-degradation
                continue
            for full_path, rel_path in all_scans:
                tasks.append((full_path, rel_path, args.output_dir, deg_type, level_name, params))

    print(f"Generating {len(tasks)} degraded volumes across degradation levels...")

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_degraded_volume, *t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Degradation Pipeline"):
                future.result()
    else:
        for t in tqdm(tasks, desc="Degradation Pipeline"):
            process_degraded_volume(*t)

    print("✨ Degradation Pipeline Complete!")

if __name__ == "__main__":
    main()
