"""Run the complete cleaning, matchup, and Elo pipeline."""

from pathlib import Path
from typing import Tuple

import pandas as pd

try:
    # These imports are used with: python -m src.run_elo
    from .clean_data import clean_race_results
    from .create_matchups import create_teammate_matchups
    from .elo import (
        ELO_SCALE,
        K_FACTOR,
        STARTING_ELO,
        calculate_elo_ratings,
    )
    from .load_data import DEFAULT_DATA_DIR, PROJECT_ROOT, load_all_data
except ImportError:
    # This fallback also lets beginners run: python src/run_elo.py
    from clean_data import clean_race_results
    from create_matchups import create_teammate_matchups
    from elo import ELO_SCALE, K_FACTOR, STARTING_ELO, calculate_elo_ratings
    from load_data import DEFAULT_DATA_DIR, PROJECT_ROOT, load_all_data


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def run_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the model, write the requested CSV outputs, and print checks."""

    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    output_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # All seven datasets are loaded here. Races, results, status, and drivers
    # feed this Elo version; the others remain available for later work.
    datasets = load_all_data(data_dir)

    clean_results, cleaning_stats = clean_race_results(
        datasets["races"],
        datasets["results"],
        datasets["status"],
    )
    matchups, matchup_stats = create_teammate_matchups(clean_results)
    final_ratings, elo_history = calculate_elo_ratings(
        matchups,
        datasets["drivers"],
        starting_elo=STARTING_ELO,
        k_factor=K_FACTOR,
        elo_scale=ELO_SCALE,
    )

    final_ratings, elo_history = _add_driver_names(
        final_ratings,
        elo_history,
        datasets["drivers"],
    )
    peak_ratings = _build_peak_ratings(final_ratings, elo_history)

    clean_results.to_csv(
        processed_dir / "clean_race_results.csv",
        index=False,
    )
    matchups.to_csv(
        processed_dir / "teammate_matchups.csv",
        index=False,
    )
    final_ratings.to_csv(
        output_dir / "final_driver_ratings.csv",
        index=False,
    )
    elo_history.to_csv(
        output_dir / "elo_history.csv",
        index=False,
    )
    peak_ratings.to_csv(
        output_dir / "peak_driver_ratings.csv",
        index=False,
    )

    _print_validation_report(cleaning_stats, matchup_stats)
    return final_ratings, elo_history


def _build_peak_ratings(
    final_ratings: pd.DataFrame,
    elo_history: pd.DataFrame,
) -> pd.DataFrame:
    """Find every driver's highest historical Elo and sort best to worst."""

    # A driver's peak is at least the starting rating, including drivers who
    # never had an Elo matchup or whose first result lowered their rating.
    peak_rows = []
    history_by_driver = {
        int(driver_id): group
        for driver_id, group in elo_history.groupby("driverId", sort=False)
    }

    for driver in final_ratings.itertuples(index=False):
        driver_id = int(driver.driverId)
        peak_elo = STARTING_ELO
        peak_race_id = pd.NA
        peak_date = pd.NaT
        peak_year = pd.NA
        peak_round = pd.NA
        peak_race_name = pd.NA

        driver_history = history_by_driver.get(driver_id)
        if driver_history is not None and not driver_history.empty:
            peak_index = driver_history["ratingAfter"].idxmax()
            peak_event = driver_history.loc[peak_index]

            # Use an event only when the driver rose above their initial Elo.
            # Otherwise, their true peak was the 1000 rating they started with.
            if float(peak_event["ratingAfter"]) > STARTING_ELO:
                peak_elo = float(peak_event["ratingAfter"])
                peak_race_id = int(peak_event["raceId"])
                peak_date = peak_event["date"]
                peak_year = int(peak_event["year"])
                peak_round = int(peak_event["round"])
                peak_race_name = peak_event["raceName"]

        peak_rows.append(
            {
                "driverId": driver_id,
                "driverName": driver.driverName,
                "forename": driver.forename,
                "surname": driver.surname,
                "peakElo": peak_elo,
                "peakRaceId": peak_race_id,
                "peakDate": peak_date,
                "peakYear": peak_year,
                "peakRound": peak_round,
                "peakRaceName": peak_race_name,
                "currentElo": float(driver.eloRating),
                "matchups": int(driver.matchups),
            }
        )

    peak_ratings = (
        pd.DataFrame(peak_rows)
        .sort_values(
            ["peakElo", "currentElo", "driverId"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    peak_ratings.insert(0, "rank", range(1, len(peak_ratings) + 1))
    return peak_ratings


def _add_driver_names(
    final_ratings: pd.DataFrame,
    elo_history: pd.DataFrame,
    drivers: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add readable names while retaining driverId as the stable identifier."""

    driver_names = drivers[["driverId", "forename", "surname"]].copy()
    driver_names["driverName"] = (
        driver_names["forename"].fillna("").astype(str).str.strip()
        + " "
        + driver_names["surname"].fillna("").astype(str).str.strip()
    ).str.strip()
    driver_names = driver_names[["driverId", "driverName", "forename", "surname"]]

    final_ratings = final_ratings.merge(
        driver_names,
        on="driverId",
        how="left",
        validate="one_to_one",
    )
    final_ratings.insert(0, "rank", range(1, len(final_ratings) + 1))
    final_columns = [
        "rank",
        "driverId",
        "driverName",
        "forename",
        "surname",
        "eloRating",
        "matchups",
        "wins",
        "losses",
    ]
    final_ratings = final_ratings[final_columns]

    elo_history = elo_history.merge(
        driver_names[["driverId", "driverName"]],
        on="driverId",
        how="left",
        validate="many_to_one",
    )
    history_columns = [
        "raceId",
        "date",
        "year",
        "round",
        "raceName",
        "driverId",
        "driverName",
        "ratingBefore",
        "ratingAfter",
        "ratingChange",
        "matchupsInRace",
        "winsInRace",
        "lossesInRace",
    ]
    return final_ratings, elo_history[history_columns]


def _print_validation_report(cleaning_stats: dict, matchup_stats: dict) -> None:
    """Print the required cleaning and matchup counts."""

    print("\nValidation report")
    print("-----------------")
    print(
        "Indianapolis 500 races removed: "
        f"{cleaning_stats['indianapolis_500_races_removed']}"
    )
    print(
        "Result rows removed with those races: "
        f"{cleaning_stats['indianapolis_500_result_rows_removed']}"
    )
    print(
        "Duplicate driver-race entries resolved: "
        f"{cleaning_stats['duplicate_driver_race_entries_resolved']}"
    )
    print(
        "Groups with fewer than two drivers: "
        f"{matchup_stats['groups_with_fewer_than_two_drivers']}"
    )
    print(
        "Multi-driver teammate groups: "
        f"{matchup_stats['multi_driver_teammate_groups']}"
    )
    print(
        "Tied-winner groups skipped: "
        f"{matchup_stats['tied_winner_groups_skipped']}"
    )
    print(
        "Non-finisher teammate comparisons skipped: "
        f"{matchup_stats['non_finisher_comparisons_skipped']}"
    )
    print(
        "Total Elo matchups generated: "
        f"{matchup_stats['total_elo_matchups_generated']}"
    )


if __name__ == "__main__":
    run_pipeline()
