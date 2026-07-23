import os
import glob
import pandas as pd
import json
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest Brainlife dataset structure and produce a dataset catalog CSV.")
    parser.add_argument("--data_dir", type=str, default="/Volumes/MyHDD/ABDN_DATA/Data_from_brainlife/proj-6554f423b094062da63aa4c9",
                        help="Root path of the Brainlife project directory.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config yaml.")
    parser.add_argument("--output", type=str, default="./data/dataset_catalog.csv", help="Path to output catalog CSV.")
    return parser.parse_args()

    

def catalog_brainlife_dataset(raw_data_dir, config_path, output_csv):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    participants_file = os.path.join(raw_data_dir, "bids", "participants.tsv")
    if not os.path.exists(participants_file):
        raise FileNotFoundError(f"Participants file not found at {participants_file}")

    participants_df = pd.read_csv(participants_file, sep='\t')
    label_mapping = cfg['labels']['mapping']

    catalog_records = []

    for _, row in participants_df.iterrows():
        sub_id = str(row['participant_id']).strip()
        sex = row.get('sex', 'N/A')
        age = row.get('age', 'N/A')
        clinical_group = str(row.get('clinical_group', '')).strip()
        label_id = label_mapping.get(clinical_group, -1)

        # Match directories belonging to this subject: e.g. sub-01, sub-01.ses-run-1, sub-01.ses-run-2
        pattern = os.path.join(raw_data_dir, f"{sub_id}*")
        matching_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

        t1_files = []
        t2_files = []
        flair_files = []

        for mdir in matching_dirs:
            # Look for dt-neuro-anat-* subdirectories
            for root, dirs, files in os.walk(mdir):
                for f in files:
                    if f.startswith("._"):
                        continue
                    full_path = os.path.join(root, f)
                    lower_f = f.lower()
                    parent_dir = os.path.basename(root).lower()

                    if f == "t1.nii.gz" or ("t1w" in parent_dir and lower_f.endswith(".nii.gz")):
                        t1_files.append(full_path)
                    elif f == "t2.nii.gz" or ("t2w" in parent_dir and lower_f.endswith(".nii.gz")):
                        t2_files.append(full_path)
                    elif f == "flair.nii.gz" or ("flair" in parent_dir and lower_f.endswith(".nii.gz")):
                        flair_files.append(full_path)

        # Remove duplicate paths if any
        t1_files = sorted(list(set(t1_files)))
        t2_files = sorted(list(set(t2_files)))
        flair_files = sorted(list(set(flair_files)))

        record = {
            "subject_id": sub_id,
            "sex": sex,
            "age": age,
            "clinical_group": clinical_group,
            "label_id": label_id,
            "has_t1": len(t1_files) > 0,
            "has_t2": len(t2_files) > 0,
            "has_flair": len(flair_files) > 0,
            "num_t1": len(t1_files),
            "num_t2": len(t2_files),
            "num_flair": len(flair_files),
            "primary_t1": t1_files[0] if t1_files else "",
            "primary_t2": t2_files[0] if t2_files else "",
            "primary_flair": flair_files[0] if flair_files else "",
            "all_t1": ";".join(t1_files),
            "all_t2": ";".join(t2_files),
            "all_flair": ";".join(flair_files),
        }
        catalog_records.append(record)

    df_catalog = pd.DataFrame(catalog_records)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_catalog.to_csv(output_csv, index=False)
    print(f"✅ Dataset catalog saved to {output_csv}")
    print(f"   Total subjects cataloged: {len(df_catalog)}")
    print(f"   Group breakdown:\n{df_catalog['clinical_group'].value_counts().to_string()}")
    print(f"   Modality availability:")
    print(f"     T1w: {df_catalog['has_t1'].sum()} subjects ({df_catalog['has_t1'].mean()*100:.1f}%)")
    print(f"     T2w: {df_catalog['has_t2'].sum()} subjects ({df_catalog['has_t2'].mean()*100:.1f}%)")
    print(f"     FLAIR: {df_catalog['has_flair'].sum()} subjects ({df_catalog['has_flair'].mean()*100:.1f}%)")
    return df_catalog

if __name__ == "__main__":
    args = parse_args()
    catalog_brainlife_dataset(args.data_dir, args.config, args.output)
