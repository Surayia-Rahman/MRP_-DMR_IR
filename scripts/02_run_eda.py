"""
Run EDA for the final DMR-IR modeling cohort.

Outputs:
- Patient-level dynamic summary table
- Regional curve table
- Regional local feature table
- Report-quality EDA and feature-behavior figures
"""

from pathlib import Path
import sys
import json

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from src.data.load_data import load_config
from src.eda.eda_plots import run_eda


def main():
    config_path = PROJECT_DIR / "configs" / "config.yaml"
    config = load_config(config_path)

    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    logs_dir = Path(config["paths"]["outputs_dir"]) / "logs"

    index_path = tables_dir / "eda" / "dataset_index.csv"

    output_tables_dir = tables_dir / "eda"
    figures_eda_dir = figures_dir / "eda"
    figures_feature_dir = figures_dir / "feature_analysis"

    logs_dir.mkdir(parents=True, exist_ok=True)

    result = run_eda(
        index_path=index_path,
        output_tables_dir=output_tables_dir,
        figures_eda_dir=figures_eda_dir,
        figures_feature_dir=figures_feature_dir,
        max_patients=None,
    )

    log_path = logs_dir / "eda_log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nEDA completed.")
    print("Saved log:", log_path)


if __name__ == "__main__":
    main()