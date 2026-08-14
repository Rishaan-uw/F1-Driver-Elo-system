"""Create winner-versus-teammate comparisons from cleaned race results."""

import re
from typing import Dict, List, Tuple

import pandas as pd


MATCHUP_COLUMNS = [
    "matchupId",
    "raceId",
    "date",
    "year",
    "round",
    "raceName",
    "constructorId",
    "winnerDriverId",
    "loserDriverId",
    "winnerPositionOrder",
    "loserPositionOrder",
    "winnerStatus",
    "loserStatus",
    "winnerActual",
    "loserActual",
]


def create_teammate_matchups(
    clean_results: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Compare each constructor's best finisher with every other teammate.

    Groups with tied best finishing positions are skipped rather than choosing
    a winner using an arbitrary secondary rule.
    """

    if clean_results.duplicated(["raceId", "driverId"]).any():
        raise ValueError(
            "Matchups require unique (raceId, driverId) combinations."
        )

    required_columns = {
        "raceId",
        "date",
        "year",
        "round",
        "raceName",
        "constructorId",
        "driverId",
        "positionOrder",
        "status",
    }
    missing_columns = required_columns - set(clean_results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Clean results are missing required columns: {missing}")

    ordered_results = clean_results.sort_values(
        [
            "date",
            "year",
            "round",
            "raceId",
            "constructorId",
            "positionOrder",
            "resultId",
        ],
        kind="mergesort",
    )

    matchup_rows: List[dict] = []
    small_group_count = 0
    multi_driver_group_count = 0
    tied_winner_group_count = 0
    non_finisher_comparison_count = 0

    groups = ordered_results.groupby(
        ["raceId", "constructorId"],
        sort=False,
        dropna=False,
    )
    for _group_key, group in groups:
        unique_driver_count = group["driverId"].nunique()
        if unique_driver_count < 2:
            small_group_count += 1
            continue
        if unique_driver_count > 2:
            multi_driver_group_count += 1

        group = group.sort_values(
            ["positionOrder", "resultId"],
            kind="mergesort",
        )
        best_position = group["positionOrder"].min()
        leaders = group.loc[group["positionOrder"].eq(best_position)]
        if leaders["driverId"].nunique() > 1:
            tied_winner_group_count += 1
            continue

        winner = leaders.iloc[0]
        losing_teammates = group.loc[~group["driverId"].eq(winner["driverId"])]

        # A group of N drivers creates N-1 comparisons: winner versus each
        # teammate. No loser-versus-loser rows are created. A driver who is one
        # or more laps down still finished; mechanical failures, accidents, and
        # all other non-finishing statuses remove only the affected comparison.
        for _, loser in losing_teammates.iterrows():
            if not (
                _is_finished_status(winner["status"])
                and _is_finished_status(loser["status"])
            ):
                non_finisher_comparison_count += 1
                continue

            matchup_rows.append(
                {
                    "raceId": int(winner["raceId"]),
                    "date": winner["date"],
                    "year": int(winner["year"]),
                    "round": int(winner["round"]),
                    "raceName": winner["raceName"],
                    "constructorId": int(winner["constructorId"]),
                    "winnerDriverId": int(winner["driverId"]),
                    "loserDriverId": int(loser["driverId"]),
                    "winnerPositionOrder": int(winner["positionOrder"]),
                    "loserPositionOrder": int(loser["positionOrder"]),
                    "winnerStatus": winner["status"],
                    "loserStatus": loser["status"],
                    "winnerActual": 1.0,
                    "loserActual": 0.0,
                }
            )

    matchups = pd.DataFrame(matchup_rows)
    if matchups.empty:
        matchups = pd.DataFrame(columns=MATCHUP_COLUMNS)
    else:
        matchups = matchups.sort_values(
            [
                "date",
                "year",
                "round",
                "raceId",
                "constructorId",
                "winnerDriverId",
                "loserDriverId",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
        matchups.insert(0, "matchupId", range(1, len(matchups) + 1))
        matchups = matchups[MATCHUP_COLUMNS]

    if not matchups.empty:
        if not matchups["winnerActual"].eq(1.0).all():
            raise AssertionError("Every matchup winner must have actual score 1.0.")
        if not matchups["loserActual"].eq(0.0).all():
            raise AssertionError("Every matchup loser must have actual score 0.0.")
        if matchups["winnerDriverId"].eq(matchups["loserDriverId"]).any():
            raise AssertionError("A driver cannot be matched against themselves.")

    stats = {
        "groups_with_fewer_than_two_drivers": small_group_count,
        "multi_driver_teammate_groups": multi_driver_group_count,
        "tied_winner_groups_skipped": tied_winner_group_count,
        "non_finisher_comparisons_skipped": non_finisher_comparison_count,
        "total_elo_matchups_generated": len(matchups),
    }
    return matchups, stats


def _is_finished_status(status: object) -> bool:
    """Return True for "Finished" and classified "+N Lap(s)" statuses."""

    if pd.isna(status):
        return False

    normalized_status = str(status).strip()
    if normalized_status.casefold() == "finished":
        return True

    # Ergast records lapped finishers as "+1 Lap", "+2 Laps", and so on.
    return re.fullmatch(
        r"\+\d+\s+Laps?",
        normalized_status,
        flags=re.IGNORECASE,
    ) is not None
