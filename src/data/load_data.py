"""
Data loading utilities for the MRP_DMRIR project.

This version uses the real DMR-IR matrix filename structure, e.g.

T0001.1.1.D.2012-10-08.00.txt

The methodology keeps frontal dynamic files:
    .1.1.D. or .2.1.D.

Then groups files by acquisition date and keeps complete 20-frame sequences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import zipfile

import numpy as np
import yaml


CLASS_TO_LABEL = {
    "healthy": 0,
    "sick": 1,
}

DYNAMIC_FRAME_COUNT = 20


def load_config(config_path: str | Path) -> Dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_dataset_zip(
    dataset_zip: str | Path,
    extract_dir: str | Path,
    force: bool = False
) -> Path:
    """Extract dataset ZIP into extract_dir."""
    dataset_zip = Path(dataset_zip)
    extract_dir = Path(extract_dir)

    if not dataset_zip.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {dataset_zip}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    if any(extract_dir.iterdir()) and not force:
        print(f"Extraction skipped. Directory already has files: {extract_dir}")
        return extract_dir

    print(f"Extracting dataset ZIP:\n  {dataset_zip}\ninto:\n  {extract_dir}")

    with zipfile.ZipFile(dataset_zip, "r") as zf:
        zf.extractall(extract_dir)

    print("Extraction completed.")
    return extract_dir


def find_dataset_root(extract_dir: str | Path) -> Path:
    """
    Find folder containing healthy/ and sick/.
    """
    extract_dir = Path(extract_dir)

    if not extract_dir.exists():
        raise FileNotFoundError(f"Extracted directory not found: {extract_dir}")

    candidates = []

    for path in [extract_dir] + [p for p in extract_dir.rglob("*") if p.is_dir()]:
        child_names = {child.name.lower() for child in path.iterdir() if child.is_dir()}
        if "healthy" in child_names and "sick" in child_names:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find dataset root containing healthy/ and sick/ under: {extract_dir}"
        )

    candidates = sorted(candidates, key=lambda p: len(p.parts))
    dataset_root = candidates[0]

    print("Detected dataset root:", dataset_root)
    return dataset_root


def read_thermal_txt(txt_path: str | Path) -> np.ndarray:
    """
    Read one thermal matrix .txt file as a 2D float32 NumPy array.
    """
    txt_path = Path(txt_path)

    if not txt_path.exists():
        raise FileNotFoundError(f"Thermal txt file not found: {txt_path}")

    delimiters = [None, ",", ";", "\t"]

    last_error = None
    for delimiter in delimiters:
        try:
            arr = np.loadtxt(txt_path, delimiter=delimiter)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D matrix but got shape {arr.shape}")
            return arr.astype(np.float32)
        except Exception as e:
            last_error = e

    raise ValueError(f"Could not read thermal txt file: {txt_path}. Last error: {last_error}")


def is_frontal_dynamic_file(file_path: str | Path) -> bool:
    """
    Check whether a file is a frontal dynamic DMR-IR matrix file.

    Accepted patterns:
        .1.1.D.
        .2.1.D.

    These correspond to frontal dynamic acquisitions used in the earlier methodology.
    """
    file_path = Path(file_path)
    name = file_path.name

    return (
        name.lower().endswith(".txt")
        and (".1.1.D." in name or ".2.1.D." in name)
    )


def parse_dmr_dynamic_filename(file_path: str | Path) -> Optional[Dict]:
    """
    Parse DMR-IR dynamic filename.

    Expected example:
        T0067.1.1.D.2012-11-21.00.txt

    Returns
    -------
    dict or None
        Parsed components if valid, otherwise None.
    """
    file_path = Path(file_path)
    name = file_path.name

    if not is_frontal_dynamic_file(file_path):
        return None

    parts = name.split(".")

    # Expected:
    # patient_id . view . position . D . date . frame . txt
    if len(parts) < 7:
        return None

    try:
        patient_id = parts[0]
        view_1 = parts[1]
        view_2 = parts[2]
        acquisition_type = parts[3]
        date_string = parts[4]
        frame_index = int(parts[5])
        extension = parts[6].lower()
    except Exception:
        return None

    if acquisition_type != "D":
        return None

    if extension != "txt":
        return None

    if not (0 <= frame_index <= 19):
        return None

    return {
        "patient_id": patient_id,
        "view_1": view_1,
        "view_2": view_2,
        "acquisition_type": acquisition_type,
        "date": date_string,
        "frame_index": frame_index,
        "path": str(file_path),
    }


def find_frontal_dynamic_files(patient_dir: str | Path) -> List[Path]:
    """
    Find frontal dynamic files inside a patient folder.
    """
    patient_dir = Path(patient_dir)

    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    files = []
    for txt_file in patient_dir.glob("*.txt"):
        if parse_dmr_dynamic_filename(txt_file) is not None:
            files.append(txt_file)

    return sorted(files)


def group_dynamic_files_by_date(patient_dir: str | Path) -> Dict[str, List[Path]]:
    """
    Group frontal dynamic files by acquisition date.
    """
    date_groups = {}

    for file_path in find_frontal_dynamic_files(patient_dir):
        parsed = parse_dmr_dynamic_filename(file_path)
        if parsed is None:
            continue

        date_string = parsed["date"]
        date_groups.setdefault(date_string, []).append(file_path)

    # Sort each date group by frame index
    for date_string in date_groups:
        date_groups[date_string] = sorted(
            date_groups[date_string],
            key=lambda p: parse_dmr_dynamic_filename(p)["frame_index"]
        )

    return date_groups


def select_earliest_complete_sequence(
    patient_dir: str | Path,
    n_frames: int = DYNAMIC_FRAME_COUNT
) -> Dict:
    """
    Select earliest frontal dynamic date group for a patient.

    Returns a dictionary containing:
        selected_date
        selected_files
        status
        all_dates
        num_selected_files
    """
    patient_dir = Path(patient_dir)
    date_groups = group_dynamic_files_by_date(patient_dir)

    if len(date_groups) == 0:
        return {
            "selected_date": None,
            "selected_files": [],
            "status": "no_frontal_dynamic_files",
            "all_dates": [],
            "num_selected_files": 0,
        }

    sorted_dates = sorted(date_groups.keys())
    earliest_date = sorted_dates[0]
    selected_files = date_groups[earliest_date]

    selected_frame_indices = [
        parse_dmr_dynamic_filename(p)["frame_index"]
        for p in selected_files
    ]

    is_complete = (
        len(selected_files) == n_frames
        and selected_frame_indices == list(range(n_frames))
    )

    status = "retained_exact_20" if is_complete else "dropped_not_exact_20"

    return {
        "selected_date": earliest_date,
        "selected_files": selected_files,
        "status": status,
        "all_dates": sorted_dates,
        "num_selected_files": len(selected_files),
        "selected_frame_indices": selected_frame_indices,
    }


def load_patient_volume_from_files(file_paths: List[str | Path]) -> np.ndarray:
    """
    Load patient volume from a list of 20 ordered file paths.
    """
    file_paths = [Path(p) for p in file_paths]

    parsed = [parse_dmr_dynamic_filename(p) for p in file_paths]
    if any(item is None for item in parsed):
        raise ValueError("One or more file paths are not valid frontal dynamic DMR-IR files.")

    ordered_pairs = sorted(
        zip(file_paths, parsed),
        key=lambda pair: pair[1]["frame_index"]
    )

    ordered_file_paths = [pair[0] for pair in ordered_pairs]
    ordered_frame_indices = [pair[1]["frame_index"] for pair in ordered_pairs]

    if ordered_frame_indices != list(range(20)):
        raise ValueError(f"Expected frame indices 0..19 but got: {ordered_frame_indices}")

    frames = [read_thermal_txt(path) for path in ordered_file_paths]

    volume = np.stack(frames, axis=0).astype(np.float32)

    return volume


def load_patient_volume(patient_dir: str | Path) -> np.ndarray:
    """
    Load earliest complete frontal dynamic volume for a patient.
    """
    selected = select_earliest_complete_sequence(patient_dir)

    if selected["status"] != "retained_exact_20":
        raise ValueError(
            f"Patient does not have earliest complete 20-frame sequence. "
            f"Status: {selected['status']}, selected files: {selected['num_selected_files']}"
        )

    return load_patient_volume_from_files(selected["selected_files"])


def summarize_volume(volume: np.ndarray) -> Dict:
    """Return basic QC statistics for a patient volume."""
    return {
        "shape": tuple(volume.shape),
        "nan_count": int(np.isnan(volume).sum()),
        "min_temp": float(np.nanmin(volume)),
        "max_temp": float(np.nanmax(volume)),
        "mean_temp": float(np.nanmean(volume)),
        "std_temp": float(np.nanstd(volume)),
    }