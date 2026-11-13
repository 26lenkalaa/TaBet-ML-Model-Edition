"""Download NBA team game logs from NBA.com (via nba_api) into basketball/data/raw/.

One CSV per season and season type, saved exactly as NBA.com returns it.
Each game appears twice (one row per team); clean.py combines them later.

By default it downloads every season from 2006-07 (where the free historical
odds start) up to the current season. Finished seasons are downloaded once;
the current season is re-downloaded every run so new games show up.

Run from the repo root:
    python3 basketball/src/ingest/fetch_games.py
    python3 basketball/src/ingest/fetch_games.py --first 2018-19 --last 2022-23
    python3 basketball/src/ingest/fetch_games.py --force   # re-download everything
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog  # type: ignore[import-untyped]


def find_basketball_dir() -> Path:
    """The 'basketball' folder this script lives in, wherever that is in the repo."""
    here = Path(__file__).resolve()
    for folder in here.parents:
        if folder.name == "basketball":
            return folder
    raise FileNotFoundError(f"{here} is not inside a folder named 'basketball'")


RAW_DIR = find_basketball_dir() / "data" / "raw"

FIRST_SEASON = "2018-19"   # free historical odds (Princeton database) start here
PLAYIN_FIRST_SEASON = "2020-21"

# NBA.com's names for each part of the season, keyed by the short name used in filenames.
SEASON_TYPES = {
    "regular": "Regular Season",
    "playin": "PlayIn",
    "playoffs": "Playoffs",
}

PAUSE_SECONDS = 2.0   # NBA.com blocks clients that send requests too fast
MAX_ATTEMPTS = 3      # NBA.com often times out; retry before giving up
TIMEOUT_SECONDS = 60


def season_name(start_year: int) -> str:
    """2022 -> '2022-23' (NBA.com's format)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def start_year(season: str) -> int:
    """'2022-23' -> 2022."""
    try:
        year = int(season[:4])
    except ValueError:
        raise ValueError(f"Season must look like 2022-23, got {season!r}") from None
    if season != season_name(year):
        raise ValueError(f"Season must look like 2022-23, got {season!r}")
    return year


def current_season(today: date | None = None) -> str:
    """The season in progress (or about to start). New seasons begin in October."""
    today = today or date.today()
    return season_name(today.year if today.month >= 8 else today.year - 1)


def seasons_between(first: str, last: str) -> list[str]:
    first_year, last_year = start_year(first), start_year(last)
    if first_year > last_year:
        raise ValueError(f"--first {first} is after --last {last}")
    return [season_name(year) for year in range(first_year, last_year + 1)]


def raw_path(season: str, season_type: str) -> Path:
    return RAW_DIR / f"games_{season}_{season_type}.csv"


def fetch_season(season: str, season_type: str) -> pd.DataFrame:
    """Ask NBA.com for every team game log in one season, retrying on errors."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=SEASON_TYPES[season_type],
                player_or_team_abbreviation="T",   # T = one row per team per game
                timeout=TIMEOUT_SECONDS,
            )
            frame: pd.DataFrame = log.get_data_frames()[0]
            return frame
        except Exception as error:  # nba_api raises plain requests errors
            if attempt == MAX_ATTEMPTS:
                raise
            wait = PAUSE_SECONDS * 2 ** attempt
            print(f"  attempt {attempt} failed ({error.__class__.__name__}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def check(frame: pd.DataFrame, season: str, season_type: str) -> None:
    """Catch obviously broken downloads before they reach clean.py."""
    if frame.empty:
        raise ValueError(f"No games returned for {season} {season_type}")

    per_game = frame.groupby("GAME_ID").size()
    if not (per_game == 2).all():
        bad = per_game[per_game != 2].index.tolist()[:5]
        raise ValueError(f"Games without exactly 2 team rows in {season}: {bad}")


def plan(seasons: list[str], types: list[str], force: bool, now: str) -> list[tuple[str, str]]:
    """Which (season, type) pairs to download this run."""
    jobs: list[tuple[str, str]] = []
    for season in seasons:
        for season_type in types:
            if season_type == "playin" and start_year(season) < start_year(PLAYIN_FIRST_SEASON):
                continue  # no play-in tournament before 2020-21
            if raw_path(season, season_type).exists() and not force and season != now:
                continue  # finished season already on disk
            jobs.append((season, season_type))
    return jobs


def main() -> None:
    now = current_season()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--first", default=FIRST_SEASON, help=f"first season (default {FIRST_SEASON})")
    parser.add_argument("--last", default=now, help=f"last season (default: current, {now})")
    parser.add_argument("--types", nargs="+", choices=SEASON_TYPES, default=list(SEASON_TYPES),
                        help="parts of the season to download (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="download again even if the file already exists")
    args = parser.parse_args()

    try:
        seasons = seasons_between(args.first, args.last)
    except ValueError as error:
        parser.error(str(error))

    jobs = plan(seasons, args.types, args.force, now)
    print(f"Seasons {seasons[0]} to {seasons[-1]}: {len(jobs)} file(s) to download "
          f"(~{len(jobs) * PAUSE_SECONDS / 60:.0f}+ min). Files already on disk are skipped.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for index, (season, season_type) in enumerate(jobs):
        if index:
            time.sleep(PAUSE_SECONDS)
        print(f"{season} {season_type}: downloading...")
        try:
            frame = fetch_season(season, season_type)
        except Exception as error:
            sys.exit(f"Failed on {season} {season_type}: {error}\n"
                     "Files saved so far are kept; rerun to continue where it stopped.")

        if frame.empty and season == now:
            print("  no games yet this season, skipping")
            continue
        try:
            check(frame, season, season_type)
        except ValueError as error:
            sys.exit(f"Bad data for {season} {season_type}: {error}")

        frame.to_csv(raw_path(season, season_type), index=False)
        print(f"  saved {len(frame) // 2} games")

    print(f"Done. Files are in {RAW_DIR}")


if __name__ == "__main__":
    main()
