"""
Run temporal CNN experiments and same-input flat logistic baselines.
"""

from pathlib import Path
import sys
import json
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from src.data.load_data import load_config
from src.models.temporal_cnn_experiments import run_temporal_cnn_experiments


def main():
    config_path = PROJECT_DIR / "configs" / "config.yaml"
    config = load_config(config_path)

    feature_data_dir = Path(config["paths"]["feature_data_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    logs_dir = Path(config["paths"]["outputs_dir"]) / "logs"

    feature_table_path = feature_data_dir / "final_feature_table_144.csv"

    output_tables_dir = tables_dir / "temporal_cnn_experiments"
    output_figures_dir = figures_dir / "temporal_cnn_experiments"

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    result = run_temporal_cnn_experiments(
        feature_table_path=feature_table_path,
        output_tables_dir=output_tables_dir,
        output_figures_dir=output_figures_dir,
    )

    result["timestamp"] = str(datetime.now())

    log_path = logs_dir / "temporal_cnn_experiments_log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nTemporal CNN experiments completed.")
    print("Saved log:", log_path)


if __name__ == "__main__":
    main()