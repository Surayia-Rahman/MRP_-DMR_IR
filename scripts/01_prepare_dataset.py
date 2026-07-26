"""
Prepare DMR-IR dataset for the MRP_DMRIR project.

This script:
1. Loads config.
2. Extracts dataset zip if needed.
3. Finds dataset root.
4. Builds full patient index.
5. Saves:
   - full audit index
   - filename-level valid index
   - final modeling cohort index
6. Tests loading one final-cohort 20-frame patient volume.
"""

from pathlib import Path
import sys
import json

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from src.data.load_data import (
    load_config,
    extract_dataset_zip,
    find_dataset_root,
    load_patient_volume_from_files,
    summarize_volume,
)
from src.data.dataset_index import (
    build_patient_index,
    get_filename_valid_sequence_index,
    get_modeling_cohort_index,
    save_patient_index,
    summarize_patient_index,
)


def main():
    config_path = PROJECT_DIR / "configs" / "config.yaml"
    exclusion_path = PROJECT_DIR / "configs" / "cohort_exclusions.yaml"

    config = load_config(config_path)

    dataset_zip = Path(config["paths"]["dataset_zip"])
    extracted_data_dir = Path(config["paths"]["extracted_data_dir"])

    tables_dir = Path(config["paths"]["tables_dir"])
    logs_dir = Path(config["paths"]["outputs_dir"]) / "logs"

    eda_tables_dir = tables_dir / "eda"
    eda_tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("PROJECT_DIR:", PROJECT_DIR)
    print("Dataset ZIP:", dataset_zip)
    print("Extracted data dir:", extracted_data_dir)
    print("Exclusion file:", exclusion_path)

    extract_dataset_zip(dataset_zip, extracted_data_dir, force=False)

    dataset_root = find_dataset_root(extracted_data_dir)

    full_index_df = build_patient_index(dataset_root)
    filename_valid_df = get_filename_valid_sequence_index(full_index_df)
    modeling_index_df = get_modeling_cohort_index(
        full_index_df,
        exclusion_path=exclusion_path
    )

    full_index_path = eda_tables_dir / "dataset_index_full_audit.csv"
    filename_valid_path = eda_tables_dir / "dataset_index_filename_valid.csv"
    modeling_index_path = eda_tables_dir / "dataset_index.csv"

    save_patient_index(full_index_df, full_index_path)
    save_patient_index(filename_valid_df, filename_valid_path)
    save_patient_index(modeling_index_df, modeling_index_path)

    summary = summarize_patient_index(
        full_index_df,
        exclusion_path=exclusion_path
    )

    print("\nDataset index summary:")
    print(json.dumps(summary, indent=2))

    print("\nSaved full audit index:", full_index_path)
    print("Saved filename-valid index:", filename_valid_path)
    print("Saved final modeling cohort index:", modeling_index_path)

    if modeling_index_df.empty:
        raise ValueError("No final modeling cohort sequences found.")

    sample_row = modeling_index_df.iloc[0]
    sample_file_paths = sample_row["file_paths"].split("|")

    print("\nTesting sample patient volume load:")
    print("Patient:", sample_row["patient_id"])
    print("Class:", sample_row["class_name"])
    print("Selected date:", sample_row["selected_date"])
    print("Number of files:", len(sample_file_paths))

    volume = load_patient_volume_from_files(sample_file_paths)
    volume_summary = summarize_volume(volume)

    print("\nSample volume summary:")
    print(json.dumps(volume_summary, indent=2))

    log = {
        "dataset_root": str(dataset_root),
        "full_index_path": str(full_index_path),
        "filename_valid_index_path": str(filename_valid_path),
        "modeling_index_path": str(modeling_index_path),
        "patient_index_summary": summary,
        "sample_patient_id": sample_row["patient_id"],
        "sample_selected_date": sample_row["selected_date"],
        "sample_volume_summary": volume_summary,
    }

    log_path = logs_dir / "dataset_preparation_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print("\nSaved log:", log_path)


if __name__ == "__main__":
    main()