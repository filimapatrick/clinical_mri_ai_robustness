import os
import sys
sys.path.insert(0, os.path.abspath("."))

import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score

from src.models.baseline import extract_radiomics_for_volume

def main():
    catalog_csv = "./data/dataset_catalog.csv"
    processed_dir = "./data/processed"
    degraded_dir = "./data/processed/degraded"
    output_report = "./data/radiomics_degradation_decay.csv"

    df_catalog = pd.read_csv(catalog_csv)
    
    # Load baseline L0 radiomics features
    l0_features_path = "./checkpoints/rf/radiomics_features.csv"
    if not os.path.exists(l0_features_path):
        print("Baseline features CSV not found! Run baseline.py first.")
        return

    df_l0 = pd.read_csv(l0_features_path)
    
    # Target map for Task 4 (Control vs Dementia vs PD)
    target_map = {0: 0, 1: 1, 2: 2}
    y_l0 = df_l0['label_id'].map(target_map).values
    groups_l0 = df_l0['subject_id'].values
    
    feature_cols = [c for c in df_l0.columns if c not in ['subject_id', 'clinical_group', 'label_id']]
    X_l0 = df_l0[feature_cols].fillna(0.0).values

    # Train 5-fold CV baseline models and store trained pipelines per fold
    sgkf = StratifiedGroupKFold(n_splits=5)
    fold_pipelines = []

    print("⚡ Training 5-Fold Baseline Calibrated RF Models on Clean L0 Data...")
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_l0, y_l0, groups=groups_l0)):
        X_train, y_train = X_l0[train_idx], y_l0[train_idx]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        
        var_th = VarianceThreshold(0.01)
        X_tr_var = var_th.fit_transform(X_tr_sc)
        
        k_best = min(15, X_tr_var.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_tr_sel = selector.fit_transform(X_tr_var, y_train)
        
        base_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
        calibrated_clf = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=3)
        calibrated_clf.fit(X_tr_sel, y_train)
        
        fold_pipelines.append({
            "scaler": scaler,
            "var_th": var_th,
            "selector": selector,
            "model": calibrated_clf,
            "test_indices": test_idx
        })

    # Compute baseline L0 performance
    l0_f1s = []
    for fold_info in fold_pipelines:
        test_idx = fold_info['test_indices']
        X_test = X_l0[test_idx]
        y_test = y_l0[test_idx]
        
        X_sc = fold_info['scaler'].transform(X_test)
        X_var = fold_info['var_th'].transform(X_sc)
        X_sel = fold_info['selector'].transform(X_var)
        
        preds = fold_info['model'].predict(X_sel)
        l0_f1s.append(f1_score(y_test, preds, average='macro'))
        
    baseline_f1 = np.mean(l0_f1s)
    print(f"✅ Baseline L0 Clean Macro F1: {baseline_f1:.4f}")

    # Evaluate on degraded folders
    degraded_folders = [d for d in os.listdir(degraded_dir) if os.path.isdir(os.path.join(degraded_dir, d))]
    results = [{
        "degradation": "baseline",
        "level": "L0",
        "macro_f1": baseline_f1,
        "rri": 1.0
    }]

    for folder in sorted(degraded_folders):
        parts = folder.split("_")
        deg_type = parts[0]
        level = parts[1] if len(parts) > 1 else "L1"

        print(f"\n🔍 Evaluating Radiomics model on {deg_type} ({level})...")
        
        # Check if features for this degraded set exist
        deg_feats_csv = os.path.join(degraded_dir, f"radiomics_{folder}.csv")
        if not os.path.exists(deg_feats_csv):
            # Extract features for sample degraded subjects
            deg_rows = []
            files = glob.glob(os.path.join(degraded_dir, folder, "**", "*.nii.gz"), recursive=True)
            
            # Map files back to subject_id
            sub_files = {}
            for f in files:
                fname = os.path.basename(f)
                sub_id = fname.split("_")[0]
                mod = "T1w" if "T1w" in fname else "T2w" if "T2w" in fname else "FLAIR" if "FLAIR" in fname else None
                if mod:
                    if sub_id not in sub_files:
                        sub_files[sub_id] = {}
                    sub_files[sub_id][mod] = f
            
            for _, row in tqdm(df_catalog.iterrows(), total=len(df_catalog), desc=f"Extracting {folder}"):
                sub_id = row['subject_id']
                if sub_id in sub_files:
                    feat_dict = {
                        "subject_id": sub_id,
                        "clinical_group": row['clinical_group'],
                        "label_id": row['label_id']
                    }
                    for mod, scan_path in sub_files[sub_id].items():
                        mod_feats = extract_radiomics_for_volume(scan_path)
                        for k, v in mod_feats.items():
                            feat_dict[f"{mod}_{k}"] = v
                    deg_rows.append(feat_dict)
            
            df_deg = pd.DataFrame(deg_rows)
            df_deg.to_csv(deg_feats_csv, index=False)
        else:
            df_deg = pd.read_csv(deg_feats_csv)

        if len(df_deg) == 0:
            continue

        # Align columns with L0
        df_deg_task = df_deg[df_deg['label_id'].isin(target_map.keys())].copy()
        y_deg = df_deg_task['label_id'].map(target_map).values
        groups_deg = df_deg_task['subject_id'].values

        # Match columns with feature_cols efficiently without fragmentation
        X_deg_raw = df_deg_task.reindex(columns=feature_cols, fill_value=0.0)
        X_deg = X_deg_raw.fillna(0.0).values

        # Evaluate cross-validation folds on degraded data
        deg_f1s = []
        for fold_info in fold_pipelines:
            # Evaluate on fold test subjects
            test_subs = set(df_l0.iloc[fold_info['test_indices']]['subject_id'])
            test_mask = df_deg_task['subject_id'].isin(test_subs)
            
            if not np.any(test_mask):
                continue

            X_test_deg = X_deg[test_mask]
            y_test_deg = y_deg[test_mask]

            X_sc = fold_info['scaler'].transform(X_test_deg)
            X_var = fold_info['var_th'].transform(X_sc)
            X_sel = fold_info['selector'].transform(X_var)

            preds = fold_info['model'].predict(X_sel)
            deg_f1s.append(f1_score(y_test_deg, preds, average='macro'))

        mean_deg_f1 = np.mean(deg_f1s) if len(deg_f1s) > 0 else 0.0
        rri = mean_deg_f1 / (baseline_f1 + 1e-12)

        results.append({
            "degradation": deg_type,
            "level": level,
            "macro_f1": mean_deg_f1,
            "rri": rri
        })
        print(f"📊 {deg_type} ({level}): Macro F1 = {mean_deg_f1:.4f} | RRI = {rri:.4f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_report, index=False)
    print(f"\n✨ Degradation decay evaluation saved to {output_report}")

if __name__ == "__main__":
    main()
