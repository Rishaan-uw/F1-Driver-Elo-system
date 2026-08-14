"""Clean Grand Prix results before teammate matchups are created."""

from typing import Dict, Tuple

import pandas as pd


INDIANAPOLIS_500_NAME = "indianapolis 500"


def clean_race_results(
    races: pd.DataFrame,
    results: pd.DataFrame,
    status: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove Indianapolis 500 races and duplicate driver-race entries.

    A duplicate is resolved by keeping the lowest finishing position, then the
    most completed laps, then the lowest result ID.
    """

    races = races.copy()
    results = results.copy()
    status = status.copy()

    # This is intentionally an exact comparison after trimming and lowering.
    # It does not remove other races held in the United States or Indianapolis.
    normalized_names = races["name"].astype("string").str.strip().str.casefold()
    indy_mask = normalized_names.eq(INDIANAPOLIS_500_NAME).fillna(False)
    indy_race_ids = set(races.loc[indy_mask, "raceId"])

    result_indy_mask = results["raceId"].isin(indy_race_ids)
    indy_races_removed = int(indy_mask.sum())
    indy_result_rows_removed = int(result_indy_mask.sum())

    analyzed_races = races.loc[~indy_mask].copy()
    clean_results = results.loc[~result_indy_mask].copy()

    # Numeric values are needed for reliable sorting. "raise" prevents bad
    # input from quietly producing incorrect race rankings.
    numeric_columns = [
        "resultId",
        "raceId",
        "driverId",
        "constructorId",
        "positionOrder",
        "laps",
    ]
    for column in numeric_columns:
        clean_results[column] = pd.to_numeric(clean_results[column], errors="raise")

    if clean_results[numeric_columns].isna().any().any():
        missing = clean_results[numeric_columns].columns[
            clean_results[numeric_columns].isna().any()
        ].tolist()
        raise ValueError(f"Results contain missing required values in: {missing}")

    rows_before_deduplication = len(clean_results)
    clean_results = (
        clean_results.sort_values(
            ["raceId", "driverId", "positionOrder", "laps", "resultId"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["raceId", "driverId"], keep="first")
        .copy()
    )
    duplicate_entries_resolved = rows_before_deduplication - len(clean_results)

    if clean_results.duplicated(["raceId", "driverId"]).any():
        raise AssertionError(
            "Cleaning failed: (raceId, driverId) combinations are not unique."
        )

    if status["statusId"].duplicated().any():
        raise ValueError("status.csv contains duplicate statusId values.")
    clean_results = clean_results.merge(
        status[["statusId", "status"]],
        on="statusId",
        how="left",
        validate="many_to_one",
    )
    if clean_results["status"].isna().any():
        missing_status_ids = sorted(
            clean_results.loc[clean_results["status"].isna(), "statusId"].unique()
        )
        raise ValueError(
            "Results reference status IDs missing from status.csv: "
            f"{missing_status_ids}"
        )

    analyzed_races["date"] = pd.to_datetime(analyzed_races["date"], errors="raise")
    race_columns = analyzed_races[
        ["raceId", "date", "year", "round", "name"]
    ].rename(columns={"name": "raceName"})

    clean_results = clean_results.merge(
        race_columns,
        on="raceId",
        how="left",
        validate="many_to_one",
    )
    if clean_results["date"].isna().any():
        missing_races = sorted(clean_results.loc[clean_results["date"].isna(), "raceId"].unique())
        raise ValueError(f"Results reference races missing from races.csv: {missing_races}")

    # Keeping chronological order in this intermediate file makes it easier to
    # inspect and ensures downstream code receives deterministic input.
    clean_results = clean_results.sort_values(
        ["date", "year", "round", "raceId", "positionOrder", "resultId"],
        kind="mergesort",
    ).reset_index(drop=True)

    metadata_columns = ["raceId", "date", "year", "round", "raceName"]
    other_columns = [
        column for column in clean_results.columns if column not in metadata_columns
    ]
    clean_results = clean_results[metadata_columns + other_columns]

    stats = {
        "indianapolis_500_races_removed": indy_races_removed,
        "indianapolis_500_result_rows_removed": indy_result_rows_removed,
        "duplicate_driver_race_entries_resolved": int(
            duplicate_entries_resolved
        ),
    }
    return clean_results, stats
