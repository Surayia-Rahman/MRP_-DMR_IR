"""
Run statistical testing for the final 144-feature table.

Outputs:
- Mann-Whitney U test results
- FDR-adjusted q-values
- Cohen's d effect sizes
- Feature-group summaries
- Report-quality statistical-test plots
"""

from pathlib import Path
import sys
import json
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

from src.data.load_data import load_config
from src.stats.statistical_tests import run_statistical_testing_pipeline


def main():
    config_path = PROJECT_DIR / "configs" / "config.yaml"
    config = load_config(config_path)

    feature_data_dir = Path(config["paths"]["feature_data_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    logs_dir = Path(config["paths"]["outputs_dir"]) / "logs"

    feature_table_path = feature_data_dir / "final_feature_table_144.csv"

    output_tables_dir = tables_dir / "statistical_tests"
    output_figures_dir = figures_dir / "statistical_tests"

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary = run_statistical_testing_pipeline(
        feature_table_path=feature_table_path,
        output_tables_dir=output_tables_dir,
        output_figures_dir=output_figures_dir,
    )

    summary["timestamp"] = str(datetime.now())

    log_path = logs_dir / "statistical_tests_log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nStatistical testing completed.")
    print("Saved log:", log_path)


if __name__ == "__main__":
    main()