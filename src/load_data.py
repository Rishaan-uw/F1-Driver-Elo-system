"""Load the raw F1 datasets without changing the source CSV files."""

from pathlib import Path
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# These are the columns used by the first version of the pipeline. Checking
# them here gives a clear error if a source file has the wrong schema.
REQUIRED_COLUMNS = {
    "races": {"raceId", "year", "round", "name", "date"},
    "results": {
        "resultId",
        "raceId",
        "driverId",
        "constructorId",
        "positionOrder",
        "laps",
    },
    "drivers": {"driverId", "forename", "surname"},
    "status": {"statusId", "status"},
    "qualifying": {"raceId", "driverId", "constructorId"},
    "sprint_results": {"raceId", "driverId", "constructorId"},
    "weather": {"raceId"},
}


def load_all_data(data_dir: Path = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Read every requested dataset and return it in a dictionary.

    Qualifying, sprint, and weather data are deliberately loaded but are not
    used by the version-one Grand Prix Elo calculation.
    """

    data_dir = Path(data_dir)
    file_names = {
        "races": "races.csv",
        "results": "results.csv",
        "drivers": "drivers.csv",
        "status": "status.csv",
        "qualifying": "qualifying.csv",
        "sprint_results": "sprint_results.csv",
        "weather": "weather.csv",
    }

    datasets: Dict[str, pd.DataFrame] = {}
    for dataset_name, file_name in file_names.items():
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Required data file not found: {file_path}")

        # Ergast-style files use \N for missing values.
        datasets[dataset_name] = pd.read_csv(
            file_path,
            na_values=[r"\N"],
            keep_default_na=True,
        )

        missing_columns = REQUIRED_COLUMNS[dataset_name] - set(
            datasets[dataset_name].columns
        )
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{file_name} is missing required columns: {missing}")

    return datasets
