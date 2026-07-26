"""
Dataset indexing utilities for the MRP_DMRIR project.

Builds a patient-level dynamic sequence registry using the DMR-IR filename logic.

For each known-label patient:
1. Keep frontal dynamic files: .1.1.D. or .2.1.D.
2. Group by acquisition date.
3. Select earliest chronological date.
4. Retain only if selected date has exactly 20 frames: 00..19.

The full audit table keeps all patients.
The final modeling cohort may apply a documented exclusion list to reproduce
the validated 248-patient experimental cohort used in the report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from src.data.load_data import CLASS_TO_LABEL, select_earliest_complete_sequence


def load_excluded_patient_ids(exclusion_path: str | Path | None) -> List[str]:
    """
    Load patient IDs to exclude from the final modeling cohort.

    Parameters
    ----------
    exclusion_path:
        Path to configs/cohort_exclusions.yaml.

    Returns
    -------
    list[str]
        Patient IDs to exclude.
    """
    if exclusion_path is None:
        return []

    exclusion_path = Path(exclusion_path)

    if not exclusion_path.exists():
        return []

    with open(exclusion_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    excluded = data.get("excluded_patient_ids", [])

    return [str(x) for x in excluded]


def build_patient_index(dataset_root: str | Path) -> pd.DataFrame:
    """
    Build patient-level index from extracted DMR-IR dataset root.

    This includes both retained and dropped patients.
    Dropped patients are kept in the audit table so the data-cleaning decision is transparent.
    """
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    records = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = dataset_root / class_name

        if not class_dir.exists():
            print(f"Warning: missing class directory: {class_dir}")
            continue

        patient_dirs = sorted([
            p for p in class_dir.iterdir()
            if p.is_dir() and p.name.startswith("T")
        ])

        print(f"{class_name}: found {len(patient_dirs)} patient folders")

        for patient_dir in patient_dirs:
            selected = select_earliest_complete_sequence(patient_dir)

            selected_files = selected.get("selected_files", [])
            selected_files_str = "|".join(str(p) for p in selected_files)

            selected_frame_indices = selected.get("selected_frame_indices", [])
            complete_dynamic_sequence = selected.get("status") == "retained_exact_20"

            records.append({
                "patient_id": patient_dir.name,
                "class_name": class_name,
                "label": label,
                "patient_dir": str(patient_dir),
                "selected_date": selected.get("selected_date"),
                "all_dates": "|".join(selected.get("all_dates", [])),
                "n_dynamic_frames_detected": selected.get("num_selected_files", 0),
                "detected_frame_indices": ",".join(str(i) for i in selected_frame_indices),
                "complete_dynamic_sequence": complete_dynamic_sequence,
                "status": selected.get("status"),
                "file_paths": selected_files_str,
            })

    index_df = pd.DataFrame(records)

    if index_df.empty:
        raise ValueError(f"No patients found under dataset root: {dataset_root}")

    index_df = index_df.sort_values(["class_name", "patient_id"]).reset_index(drop=True)

    return index_df


def get_filename_valid_sequence_index(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return filename-level retained exact 20-frame dynamic sequences.

    This is before applying any manually documented cohort exclusions.
    """
    return index_df[index_df["complete_dynamic_sequence"]].copy().reset_index(drop=True)


def get_modeling_cohort_index(
    index_df: pd.DataFrame,
    exclusion_path: str | Path | None = None
) -> pd.DataFrame:
    """
    Return final modeling cohort.

    This applies:
    1. complete_dynamic_sequence == True
    2. documented patient exclusions, if provided
    """
    valid_df = get_filename_valid_sequence_index(index_df)

    excluded_ids = load_excluded_patient_ids(exclusion_path)

    if excluded_ids:
        valid_df = valid_df[
            ~valid_df["patient_id"].astype(str).isin(excluded_ids)
        ].copy()

    return valid_df.reset_index(drop=True)


def save_patient_index(index_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save patient index CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    index_df.to_csv(output_path, index=False)

    return output_path


def summarize_patient_index(
    index_df: pd.DataFrame,
    exclusion_path: str | Path | None = None
) -> Dict:
    """
    Summarize full patient index and final modeling cohort.
    """
    filename_valid_df = get_filename_valid_sequence_index(index_df)
    modeling_df = get_modeling_cohort_index(index_df, exclusion_path=exclusion_path)

    excluded_ids = load_excluded_patient_ids(exclusion_path)

    summary = {
        "n_patient_folders": int(len(index_df)),
        "all_class_counts": index_df["class_name"].value_counts().to_dict(),

        "filename_level_valid_sequences": int(len(filename_valid_df)),
        "filename_level_valid_class_counts": filename_valid_df["class_name"].value_counts().to_dict(),

        "excluded_patient_ids": excluded_ids,
        "n_excluded_from_modeling": int(len(excluded_ids)),

        "final_modeling_cohort": int(len(modeling_df)),
        "final_modeling_class_counts": modeling_df["class_name"].value_counts().to_dict(),
        "final_modeling_label_counts": modeling_df["label"].value_counts().sort_index().to_dict(),

        "dropped_before_filename_validity": int((~index_df["complete_dynamic_sequence"]).sum()),
        "status_counts": index_df["status"].value_counts(dropna=False).to_dict(),
    }

    return summary