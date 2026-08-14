"""Core Elo rating calculations for F1 teammate matchups."""

from collections import defaultdict
from typing import DefaultDict, Dict, Tuple

import pandas as pd


# Keep the main model settings in one place so they are easy to tune later.
STARTING_ELO = 1000.0
K_FACTOR = 48.0
ELO_SCALE = 400.0


def expected_score(
    rating_a: float,
    rating_b: float,
    elo_scale: float = ELO_SCALE,
) -> float:
    """Return driver A's expected score against driver B."""

    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / elo_scale))


def calculate_elo_ratings(
    matchups: pd.DataFrame,
    drivers: pd.DataFrame,
    starting_elo: float = STARTING_ELO,
    k_factor: float = K_FACTOR,
    elo_scale: float = ELO_SCALE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process matchup rows chronologically and return ratings and history.

    Every comparison in a teammate group uses ratings from the start of that
    group. Changes are accumulated before being applied, so teammate row order
    cannot affect the result.
    """

    driver_ids = pd.to_numeric(drivers["driverId"], errors="raise").astype(int)
    if driver_ids.duplicated().any():
        raise ValueError("drivers.csv contains duplicate driverId values.")

    ratings: Dict[int, float] = {
        driver_id: float(starting_elo) for driver_id in driver_ids
    }
    matchup_counts: DefaultDict[int, int] = defaultdict(int)
    win_counts: DefaultDict[int, int] = defaultdict(int)
    loss_counts: DefaultDict[int, int] = defaultdict(int)
    history_rows = []

    if matchups.empty:
        final_ratings = _build_final_ratings(
            ratings,
            matchup_counts,
            win_counts,
            loss_counts,
        )
        return final_ratings, _empty_history()

    required_columns = {
        "raceId",
        "date",
        "year",
        "round",
        "raceName",
        "constructorId",
        "winnerDriverId",
        "loserDriverId",
        "winnerActual",
        "loserActual",
    }
    missing_columns = required_columns - set(matchups.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Matchups are missing required columns: {missing}")

    ordered_matchups = matchups.sort_values(
        ["date", "year", "round", "raceId", "matchupId"],
        kind="mergesort",
    )

    # raceId identifies a single event. sort=False preserves the chronological
    # order established immediately above.
    for _, race_matchups in ordered_matchups.groupby("raceId", sort=False):
        rating_changes: DefaultDict[int, float] = defaultdict(float)
        race_start_ratings: Dict[int, float] = {}
        race_matchup_counts: DefaultDict[int, int] = defaultdict(int)
        race_wins: DefaultDict[int, int] = defaultdict(int)
        race_losses: DefaultDict[int, int] = defaultdict(int)
        participating_drivers = set()

        # Dividing K among the losing teammates prevents a winner from
        # receiving several full-strength Elo rewards merely because a
        # historical constructor entered more than two drivers.
        constructor_groups = race_matchups.groupby("constructorId", sort=False)
        for _, teammate_group in constructor_groups:
            winner_ids = teammate_group["winnerDriverId"].unique()
            loser_ids = teammate_group["loserDriverId"].unique()
            if len(winner_ids) != 1:
                raise ValueError(
                    "Each race-constructor group must have exactly one winner."
                )
            if len(loser_ids) != len(teammate_group):
                raise ValueError(
                    "A race-constructor group contains duplicate loser matchups."
                )

            winner_id = int(winner_ids[0])
            number_of_losing_teammates = len(loser_ids)
            group_driver_ids = {winner_id, *(int(value) for value in loser_ids)}
            teammate_group_size = len(group_driver_ids)

            if number_of_losing_teammates == 0:
                raise ValueError(
                    "A teammate group must contain at least one losing driver."
                )
            if (
                number_of_losing_teammates
                != teammate_group_size - 1
            ):
                raise AssertionError(
                    "Losing-teammate count must equal teammate-group size minus one."
                )

            expected_effective_k = k_factor / number_of_losing_teammates
            effective_k = expected_effective_k
            if effective_k != expected_effective_k:
                raise AssertionError(
                    "Effective K must equal base K divided by losing teammates."
                )

            group_changes: DefaultDict[int, float] = defaultdict(float)

            for driver_id in group_driver_ids:
                if driver_id not in ratings:
                    raise ValueError(
                        f"Matchup references driverId {driver_id}, which is "
                        "missing from drivers.csv."
                    )
                race_start_ratings.setdefault(driver_id, ratings[driver_id])

            # Ratings are intentionally read before applying any group changes.
            # The winner therefore has the same pre-group rating against every
            # teammate, regardless of matchup row order.
            winner_rating = ratings[winner_id]
            for matchup in teammate_group.itertuples(index=False):
                loser_id = int(matchup.loserDriverId)
                expected_winner = expected_score(
                    winner_rating,
                    ratings[loser_id],
                    elo_scale,
                )
                change = effective_k * (1.0 - expected_winner)

                group_changes[winner_id] += change
                group_changes[loser_id] -= change
                race_matchup_counts[winner_id] += 1
                race_matchup_counts[loser_id] += 1
                race_wins[winner_id] += 1
                race_losses[loser_id] += 1

            sum_of_group_rating_changes = sum(group_changes.values())
            if abs(sum_of_group_rating_changes) >= 1e-9:
                raise AssertionError(
                    "Absolute teammate-group Elo change sum must be below 1e-9."
                )
            winner_total_gain = group_changes[winner_id]
            if winner_total_gain > k_factor + 1e-9:
                raise AssertionError(
                    "A group winner's total gain cannot exceed the base K-factor."
                )

            # Apply changes only after every winner-versus-loser comparison in
            # this constructor group has been calculated.
            for driver_id, change in group_changes.items():
                ratings[driver_id] += change
                rating_changes[driver_id] += change
            participating_drivers.update(group_driver_ids)

        race_info = race_matchups.iloc[0]
        for driver_id in sorted(participating_drivers):
            matchup_counts[driver_id] += race_matchup_counts[driver_id]
            win_counts[driver_id] += race_wins[driver_id]
            loss_counts[driver_id] += race_losses[driver_id]

            history_rows.append(
                {
                    "raceId": int(race_info["raceId"]),
                    "date": race_info["date"],
                    "year": int(race_info["year"]),
                    "round": int(race_info["round"]),
                    "raceName": race_info["raceName"],
                    "driverId": driver_id,
                    "ratingBefore": race_start_ratings[driver_id],
                    "ratingAfter": ratings[driver_id],
                    "ratingChange": rating_changes[driver_id],
                    "matchupsInRace": race_matchup_counts[driver_id],
                    "winsInRace": race_wins[driver_id],
                    "lossesInRace": race_losses[driver_id],
                }
            )

    final_ratings = _build_final_ratings(
        ratings,
        matchup_counts,
        win_counts,
        loss_counts,
    )
    history = pd.DataFrame(history_rows)
    return final_ratings, history


def _build_final_ratings(
    ratings: Dict[int, float],
    matchup_counts: Dict[int, int],
    win_counts: Dict[int, int],
    loss_counts: Dict[int, int],
) -> pd.DataFrame:
    """Put rating dictionaries into a sorted table."""

    rows = [
        {
            "driverId": driver_id,
            "eloRating": rating,
            "matchups": matchup_counts[driver_id],
            "wins": win_counts[driver_id],
            "losses": loss_counts[driver_id],
        }
        for driver_id, rating in ratings.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["eloRating", "driverId"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _empty_history() -> pd.DataFrame:
    """Return an empty history table with the normal output columns."""

    return pd.DataFrame(
        columns=[
            "raceId",
            "date",
            "year",
            "round",
            "raceName",
            "driverId",
            "ratingBefore",
            "ratingAfter",
            "ratingChange",
            "matchupsInRace",
            "winsInRace",
            "lossesInRace",
        ]
    )
