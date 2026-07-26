"""
Extract final statistical and physics-informed features for the DMR-IR cohort.

Outputs:
1. Statistical-only feature table
2. Physics-informed-only feature table
3. Final 144-feature combined table
4. Feature-group summary table
5. Feature extraction log
"""

from pathlib import Path
import sys
import json
from datetime import datetime

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from src.data.load_data import (
    load_config,
    load_patient_volume_from_files,
    summarize_volume,
)

from src.features.statistical_features import compute_statistical_surface_features
from src.features.physics_informed_features import compute_physics_informed_features

from src.features.feature_groups import (
    METADATA_COLS,
    STATISTICAL_FEATURES,
    PHYSICS_INFORMED_FEATURES,
    COMBINED_FEATURES,
    summarize_feature_groups,
    validate_feature_table_columns,
)


def main():
    config_path = PROJECT_DIR / "configs" / "config.yaml"
    config = load_config(config_path)

    tables_dir = Path(config["paths"]["tables_dir"])
    feature_data_dir = Path(config["paths"]["feature_data_dir"])
    logs_dir = Path(config["paths"]["outputs_dir"]) / "logs"

    index_path = tables_dir / "eda" / "dataset_index.csv"

    output_tables_dir = tables_dir / "features"
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    feature_data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not index_path.exists():
        raise FileNotFoundError(f"Final modeling cohort index not found: {index_path}")

    index_df = pd.read_csv(index_path)

    print("Loaded final modeling cohort index:", index_path)
    print("Index shape:", index_df.shape)
    print(index_df["class_name"].value_counts())

    feature_group_summary = summarize_feature_groups()

    print("\nExpected feature group counts:")
    print(json.dumps(feature_group_summary, indent=2))

    statistical_rows = []
    physics_rows = []
    combined_rows = []

    volume_qc_rows = []

    for row_i, row in index_df.iterrows():
        patient_id = str(row["patient_id"])
        class_name = str(row["class_name"])
        label = int(row["label"])
        selected_date = row["selected_date"]
        file_paths = str(row["file_paths"]).split("|")

        print(f"[{row_i + 1}/{len(index_df)}] Extracting features for {patient_id} ({class_name})")

        volume = load_patient_volume_from_files(file_paths)

        volume_qc = summarize_volume(volume)
        volume_qc["patient_id"] = patient_id
        volume_qc["class_name"] = class_name
        volume_qc["label"] = label
        volume_qc_rows.append(volume_qc)

        metadata = {
            "patient_id": patient_id,
            "class_name": class_name,
            "label": label,
            "selected_date": selected_date,
        }

        statistical_features = compute_statistical_surface_features(volume)
        physics_features = compute_physics_informed_features(volume)

        statistical_row = {
            **metadata,
            **statistical_features,
        }

        physics_row = {
            **metadata,
            **physics_features,
        }

        combined_row = {
            **metadata,
            **statistical_features,
            **physics_features,
        }

        statistical_rows.append(statistical_row)
        physics_rows.append(physics_row)
        combined_rows.append(combined_row)

    statistical_df = pd.DataFrame(statistical_rows)
    physics_df = pd.DataFrame(physics_rows)
    combined_df = pd.DataFrame(combined_rows)
    volume_qc_df = pd.DataFrame(volume_qc_rows)

    # Enforce exact column order.
    statistical_df = statistical_df[METADATA_COLS + STATISTICAL_FEATURES]
    physics_df = physics_df[METADATA_COLS + PHYSICS_INFORMED_FEATURES]
    combined_df = combined_df[METADATA_COLS + COMBINED_FEATURES]

    validation = validate_feature_table_columns(combined_df.columns.tolist())

    print("\nFinal combined feature table shape:", combined_df.shape)
    print("Label counts:")
    print(combined_df["label"].value_counts().sort_index())
    print("Class counts:")
    print(combined_df["class_name"].value_counts())

    print("\nValidation:")
    print(json.dumps(validation, indent=2))

    # Strict checks.
    if combined_df.shape[0] != 248:
        raise ValueError(f"Expected 248 patients, got {combined_df.shape[0]}")

    if len(STATISTICAL_FEATURES) != 53:
        raise ValueError(f"Expected 53 statistical features, got {len(STATISTICAL_FEATURES)}")

    if len(PHYSICS_INFORMED_FEATURES) != 91:
        raise ValueError(f"Expected 91 physics-informed features, got {len(PHYSICS_INFORMED_FEATURES)}")

    if len(COMBINED_FEATURES) != 144:
        raise ValueError(f"Expected 144 combined features, got {len(COMBINED_FEATURES)}")

    if combined_df.shape[1] != 148:
        raise ValueError(f"Expected final table shape (248, 148), got {combined_df.shape}")

    if validation["missing_metadata"]:
        raise ValueError(f"Missing metadata columns: {validation['missing_metadata']}")

    if validation["missing_combined"]:
        raise ValueError(f"Missing combined features: {validation['missing_combined']}")

    if validation["unassigned_features"]:
        raise ValueError(f"Unassigned features found: {validation['unassigned_features']}")

    if validation["overlap"]:
        raise ValueError(f"Feature overlap found: {validation['overlap']}")

    # Save outputs.
    statistical_path = feature_data_dir / "statistical_surface_features.csv"
    physics_path = feature_data_dir / "physics_informed_recovery_features.csv"
    combined_path = feature_data_dir / "final_feature_table_144.csv"

    # Compatibility name close to the previous notebook output.
    compatibility_path = feature_data_dir / "statistical_features_v3_with_dynamic_std_and_local_recovery_gpu_reproduced.csv"

    statistical_table_path = output_tables_dir / "statistical_surface_features.csv"
    physics_table_path = output_tables_dir / "physics_informed_recovery_features.csv"
    combined_table_path = output_tables_dir / "final_feature_table_144.csv"
    volume_qc_path = output_tables_dir / "volume_qc_summary.csv"
    feature_group_summary_path = output_tables_dir / "feature_group_summary.csv"

    statistical_df.to_csv(statistical_path, index=False)
    physics_df.to_csv(physics_path, index=False)
    combined_df.to_csv(combined_path, index=False)
    combined_df.to_csv(compatibility_path, index=False)

    statistical_df.to_csv(statistical_table_path, index=False)
    physics_df.to_csv(physics_table_path, index=False)
    combined_df.to_csv(combined_table_path, index=False)
    volume_qc_df.to_csv(volume_qc_path, index=False)

    pd.DataFrame([
        {
            "feature_group": "statistical_surface",
            "n_features": len(STATISTICAL_FEATURES),
        },
        {
            "feature_group": "physics_informed_recovery",
            "n_features": len(PHYSICS_INFORMED_FEATURES),
        },
        {
            "feature_group": "combined_statistical_physics",
            "n_features": len(COMBINED_FEATURES),
        },
        {
            "feature_group": "overlap_between_statistical_and_physics",
            "n_features": len(set(STATISTICAL_FEATURES).intersection(PHYSICS_INFORMED_FEATURES)),
        },
    ]).to_csv(feature_group_summary_path, index=False)

    print("\nSaved feature data files:")
    print(statistical_path)
    print(physics_path)
    print(combined_path)
    print(compatibility_path)

    print("\nSaved feature tables:")
    print(statistical_table_path)
    print(physics_table_path)
    print(combined_table_path)
    print(volume_qc_path)
    print(feature_group_summary_path)

    log = {
        "timestamp": str(datetime.now()),
        "index_path": str(index_path),
        "n_patients": int(combined_df.shape[0]),
        "final_shape": list(combined_df.shape),
        "class_counts": combined_df["class_name"].value_counts().to_dict(),
        "label_counts": combined_df["label"].value_counts().sort_index().to_dict(),
        "feature_group_summary": feature_group_summary,
        "validation": validation,
        "saved_paths": {
            "statistical_path": str(statistical_path),
            "physics_path": str(physics_path),
            "combined_path": str(combined_path),
            "compatibility_path": str(compatibility_path),
            "statistical_table_path": str(statistical_table_path),
            "physics_table_path": str(physics_table_path),
            "combined_table_path": str(combined_table_path),
            "volume_qc_path": str(volume_qc_path),
            "feature_group_summary_path": str(feature_group_summary_path),
        },
    }

    log_path = logs_dir / "feature_extraction_log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print("\nFeature extraction completed.")
    print("Saved log:", log_path)


if __name__ == "__main__":
    main()