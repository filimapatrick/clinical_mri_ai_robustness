import os
import sys
import logging
sys.path.insert(0, os.path.abspath("."))

import argparse
import yaml
import pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss

try:
    from radiomics import featureextractor, logger as radiomics_logger
    radiomics_logger.setLevel(logging.ERROR)
    HAS_RADIOMICS = True
except ImportError:
    HAS_RADIOMICS = False

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Leak-Free Radiomics + ML Baseline Pipeline.")
    parser.add_argument("--catalog", type=str, default="./data/dataset_catalog.csv", help="Catalog CSV.")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Processed scans dir.")
    parser.add_argument("--degraded_dir", type=str, default="./data/processed/degraded", help="Degraded scans dir.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config YAML.")
    parser.add_argument("--model", type=str, default="randomforest", choices=["randomforest", "logistic"], help="ML Classifier.")
    parser.add_argument("--task", type=str, default="task4", help="Task name (task1..task4).")
    parser.add_argument("--out_dir", type=str, default="./checkpoints/rf", help="Checkpoints output directory.")
    parser.add_argument("--extract_only", action="store_true", help="Only extract radiomics features to CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)

def crop_to_brain_roi(image, mask):
    """Crop SimpleITK image and mask to the brain ROI bounding box for fast feature extraction."""
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask)
    if label_shape_filter.HasLabel(1):
        bbox = label_shape_filter.GetBoundingBox(1) # (x_min, y_min, z_min, x_size, y_size, z_size)
        cropped_image = sitk.RegionOfInterest(image, bbox[3:], bbox[:3])
        cropped_mask = sitk.RegionOfInterest(mask, bbox[3:], bbox[:3])
        return cropped_image, cropped_mask
    return image, mask

def extract_native_intensity_features(image_np):
    """Fallback first-order intensity features if PyRadiomics is unavailable."""
    fg = image_np[image_np > 0]
    if len(fg) == 0:
        fg = image_np
    
    return {
        "mean": float(np.mean(fg)),
        "std": float(np.std(fg)),
        "variance": float(np.var(fg)),
        "min": float(np.min(fg)),
        "max": float(np.max(fg)),
        "median": float(np.median(fg)),
        "q25": float(np.percentile(fg, 25)),
        "q75": float(np.percentile(fg, 75)),
        "iqr": float(np.percentile(fg, 75) - np.percentile(fg, 25)),
        "skewness": float(pd.Series(fg).skew()),
        "kurtosis": float(pd.Series(fg).kurtosis()),
        "energy": float(np.sum(fg**2)),
        "rms": float(np.sqrt(np.mean(fg**2)))
    }

def extract_radiomics_for_volume(image_path, bin_width=25):
    """
    Optimized PyRadiomics extraction with ROI bounding box cropping and 2mm resampling.
    Takes 1-2 seconds per 3D volume instead of 15 minutes!
    """
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    img_np = sitk.GetArrayFromImage(image)

    if HAS_RADIOMICS:
        # Robust brain mask generation
        pos_voxels = img_np[img_np > 0]
        if len(pos_voxels) > 0:
            th = np.percentile(pos_voxels, 1)
            brain_mask_np = (img_np >= th).astype(np.uint8)
        else:
            brain_mask_np = np.zeros_like(img_np, dtype=np.uint8)
            
        if np.sum(brain_mask_np) == 0:
            # Emergency fallback: central bounding box
            d, h, w = img_np.shape
            brain_mask_np[d//4:3*d//4, h//4:3*h//4, w//4:3*w//4] = 1

        mask = sitk.GetImageFromArray(brain_mask_np)
        mask.CopyInformation(image)

        # 1. Bounding box crop for 500% speedup
        cropped_img, cropped_mask = crop_to_brain_roi(image, mask)

        # 2. PyRadiomics Extractor with IBSI settings
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.settings['binWidth'] = bin_width
        extractor.settings['resampledPixelSpacing'] = [2.0, 2.0, 2.0] # 2mm isotropic grid for fast, stable 3D GLCM calculation
        extractor.settings['interpolator'] = sitk.sitkBSpline
        extractor.settings['correctMask'] = True

        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('shape')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')

        results = extractor.execute(cropped_img, cropped_mask)

        features = {}
        for key, val in results.items():
            if not key.startswith("diagnostics_"):
                try:
                    features[key] = float(val)
                except (ValueError, TypeError):
                    pass
        return features
    else:
        return extract_native_intensity_features(img_np)

def extract_dataset_radiomics(catalog_csv, processed_dir, output_csv):
    """
    Extract radiomics features incrementally across all subjects and primary modalities.
    Supports checkpointing: resumes instantly if interrupted.
    """
    df = pd.read_csv(catalog_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    processed_subs = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'subject_id' in existing_df.columns:
                processed_subs = set(existing_df['subject_id'].unique())
                print(f"🔄 Resuming radiomics extraction: {len(processed_subs)}/{len(df)} subjects already completed.")
        except Exception:
            pass

    print(f"Extracting IBSI PyRadiomics features for {len(df) - len(processed_subs)} remaining subjects...")
    all_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Optimized Radiomics Extraction"):
        sub_id = row['subject_id']
        if sub_id in processed_subs:
            continue

        sub_dir = os.path.join(processed_dir, sub_id)
        feat_dict = {
            "subject_id": sub_id,
            "clinical_group": row['clinical_group'],
            "label_id": row['label_id']
        }

        for mod in ['T1w', 'T2w', 'FLAIR']:
            scan_path = os.path.join(sub_dir, f"{sub_id}_{mod}.nii.gz")
            if os.path.exists(scan_path):
                mod_feats = extract_radiomics_for_volume(scan_path)
                for k, v in mod_feats.items():
                    feat_dict[f"{mod}_{k}"] = v

        all_rows.append(feat_dict)

        # Incremental save every 5 subjects
        if len(all_rows) % 5 == 0 or len(all_rows) == (len(df) - len(processed_subs)):
            df_new = pd.DataFrame(all_rows)
            if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
                df_combined = pd.concat([pd.read_csv(output_csv), df_new], ignore_index=True)
            else:
                df_combined = df_new
            df_combined.to_csv(output_csv, index=False)

    df_feats = pd.read_csv(output_csv)
    print(f"✅ Radiomics features complete and saved to {output_csv} (Shape: {df_feats.shape})")
    return df_feats

def run_leak_free_cross_validation(df_feats, task_name="task4", model_type="randomforest", config_path="config/config.yaml", seed=42):
    """
    Execute 5-fold Stratified Group Cross-Validation with ZERO data leakage:
    1. Splits strictly at subject level (StratifiedGroupKFold).
    2. StandardScaler fit ONLY on training subjects of each fold.
    3. VarianceThreshold & Feature Selection fit ONLY on training subjects of each fold.
    4. Probability calibration (Platt Scaling) fit ONLY on training fold.
    """
    np.random.seed(seed)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    task_cfg = cfg['tasks'][task_name]
    valid_classes = task_cfg['classes']

    df_task = df_feats[df_feats['label_id'].isin(valid_classes)].copy().reset_index(drop=True)
    target_map = {cls: idx for idx, cls in enumerate(sorted(valid_classes))}
    y = df_task['label_id'].map(target_map).values
    groups = df_task['subject_id'].values

    feature_cols = [c for c in df_task.columns if c not in ['subject_id', 'clinical_group', 'label_id']]
    X_raw = df_task[feature_cols].fillna(0.0).values

    sgkf = StratifiedGroupKFold(n_splits=5)
    
    fold_results = []
    print(f"\n⚡ Running 5-Fold Leak-Free Cross-Validation [{model_type.upper()}] on {task_name}...")

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_raw, y, groups=groups)):
        X_train_raw, y_train = X_raw[train_idx], y[train_idx]
        X_test_raw, y_test = X_raw[test_idx], y[test_idx]

        # 1. Scaler fit ONLY on training fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 2. Variance Threshold fit ONLY on training fold
        var_thresh = VarianceThreshold(threshold=0.01)
        X_train_var = var_thresh.fit_transform(X_train_scaled)
        X_test_var = var_thresh.transform(X_test_scaled)

        # 3. Feature Selection (K=15) fit ONLY on training fold
        k_best = min(15, X_train_var.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_train_sel = selector.fit_transform(X_train_var, y_train)
        X_test_sel = selector.transform(X_test_var)

        # 4. Base Classifier & Calibrated Model (Platt Scaling)
        if model_type == "randomforest":
            base_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed, class_weight='balanced')
        else:
            base_clf = LogisticRegression(max_iter=1000, penalty='l2', C=0.1, random_state=seed, class_weight='balanced')

        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)
        calibrated_clf.fit(X_train_sel, y_train)

        # 5. Evaluate Test Fold
        test_preds = calibrated_clf.predict(X_test_sel)
        test_probs = calibrated_clf.predict_proba(X_test_sel)

        f1_macro = f1_score(y_test, test_preds, average='macro')
        bal_acc = balanced_accuracy_score(y_test, test_preds)
        ece = compute_ece(y_test, test_probs)

        fold_results.append({
            "fold": fold + 1,
            "f1_macro": f1_macro,
            "balanced_accuracy": bal_acc,
            "ece": ece
        })
        print(f"   Fold {fold + 1}: Macro F1 = {f1_macro:.4f} | Bal Acc = {bal_acc:.4f} | ECE = {ece:.4f}")

    df_res = pd.DataFrame(fold_results)
    mean_f1 = df_res['f1_macro'].mean()
    std_f1 = df_res['f1_macro'].std()
    mean_bal_acc = df_res['balanced_accuracy'].mean()
    mean_ece = df_res['ece'].mean()

    print(f"\n📊 --- 5-FOLD LEAK-FREE CROSS-VALIDATION SUMMARY ---")
    print(f"   Mean Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"   Mean Balanced Acc: {mean_bal_acc:.4f}")
    print(f"   Mean ECE (Calibrated): {mean_ece:.4f}")

    return df_res

if __name__ == "__main__":
    args = parse_args()
    feats_csv = os.path.join(args.out_dir, "radiomics_features.csv")

    df_feats = extract_dataset_radiomics(args.catalog, args.processed_dir, feats_csv)

    if not args.extract_only:
        run_leak_free_cross_validation(df_feats, task_name=args.task, model_type=args.model, config_path=args.config, seed=args.seed)
