import os
import yaml
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

class SubjectMRIDataset(Dataset):
    """
    Subject-level PyTorch Dataset for Multimodal Clinical Brain MRI.
    Aggregates images by subject ID to prevent data leakage.
    Supports task-based label filtering and missing modality mask returns.
    """
    def __init__(self, catalog_csv, processed_dir, task_name="task4", modalities=None, config_path="config/config.yaml", transform=None):
        super().__init__()
        self.catalog_csv = catalog_csv
        self.processed_dir = processed_dir
        self.task_name = task_name
        self.transform = transform

        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        if modalities is None:
            self.modalities = self.cfg['modalities']
        else:
            self.modalities = modalities

        # Load catalog
        df = pd.read_csv(catalog_csv)
        
        # Filter according to task classes
        task_cfg = self.cfg['tasks'].get(task_name, self.cfg['tasks']['task4'])
        valid_classes = task_cfg['classes']

        self.df = df[df['label_id'].isin(valid_classes)].reset_index(drop=True)

        # Build task-specific target mapping (e.g., for binary tasks [0, 2] -> map class 2 to target 1)
        self.target_map = {cls: idx for idx, cls in enumerate(sorted(valid_classes))}

    def __len__(self):
        return len(self.df)

    def _load_volume(self, filepath):
        if not os.path.exists(filepath):
            return None
        image = sitk.ReadImage(filepath, sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(image) # (D, H, W)
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sub_id = row['subject_id']
        raw_label = row['label_id']
        target_label = self.target_map[raw_label]

        volumes = []
        modality_mask = []

        sub_dir = os.path.join(self.processed_dir, sub_id)

        for mod in self.modalities:
            file_name = f"{sub_id}_{mod}.nii.gz"
            filepath = os.path.join(sub_dir, file_name)
            arr = self._load_volume(filepath)

            if arr is not None:
                volumes.append(arr)
                modality_mask.append(1.0)
            else:
                # Modality missing: placeholder dummy zeros with standard size
                dummy_shape = tuple(self.cfg['preprocessing']['image_size'])
                volumes.append(np.zeros(dummy_shape, dtype=np.float32))
                modality_mask.append(0.0)

        # Stack modalities into shape (C, D, H, W)
        image_tensor = torch.from_numpy(np.stack(volumes, axis=0)).float()
        mask_tensor = torch.tensor(modality_mask, dtype=torch.float32)
        label_tensor = torch.tensor(target_label, dtype=torch.long)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return {
            "subject_id": sub_id,
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label_tensor,
            "raw_label": torch.tensor(raw_label, dtype=torch.long)
        }

def get_stratified_subject_splits(catalog_csv, config_path="config/config.yaml", seed=42):
    """
    Generate subject-level stratified Train / Val / Test splits.
    Ensures zero subject overlap across splits.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(catalog_csv)
    train_ratio = cfg['split']['train_ratio']
    val_ratio = cfg['split']['val_ratio']
    test_ratio = cfg['split']['test_ratio']

    # First split into train_val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['label_id'],
        random_state=seed
    )

    # Calculate adjusted val ratio for train_val subset
    adj_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adj_val_ratio,
        stratify=train_val_df['label_id'],
        random_state=seed
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=str, default="./data/dataset_catalog.csv")
    parser.add_argument("--processed_dir", type=str, default="./data/processed")
    args = parser.parse_args()

    if os.path.exists(args.catalog):
        splits = get_stratified_subject_splits(args.catalog)
        print("✅ Subject-level Stratified Splits generated successfully:")
        print(f"   Train: {len(splits['train'])} subjects")
        print(f"   Val:   {len(splits['val'])} subjects")
        print(f"   Test:  {len(splits['test'])} subjects")
    else:
        print("Catalog CSV not found. Run ingest_brainlife.py first.")
