"""Tests for fetch_games.py. NBA.com is never called: a fake endpoint stands in.

Run from the repo root:
    python3 -m pytest basketball/tests -v
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder
import src.ingest.fetch_games as fg  # noqa: E402

REAL_CURRENT_SEASON = fg.current_season   # saved before the fixture fakes it
CURRENT = "2026-27"


def team_rows(game_ids: list[str]) -> pd.DataFrame:
    """Two rows per game, like NBA.com returns."""
    rows = []
    for game_id in game_ids:
        rows.append({"GAME_ID": game_id, "TEAM_ABBREVIATION": "BOS", 
                     "MATCHUP": "BOS vs. NYK", "PTS": 110})
        rows.append({"GAME_ID": game_id, "TEAM_ABBREVIATION": "NYK",
                     "MATCHUP": "NYK @ BOS", "PTS": 104})
    return pd.DataFrame(rows)


class FakeLog:
    """Stands in for leaguegamelog.LeagueGameLog."""

    calls: list[dict[str, object]] = []
    fail_times = 0
    fail_season = ""
    frame = team_rows(["001", "002"])

    def __init__(self, **kwargs: object) -> None:
        FakeLog.calls.append(kwargs)
        if FakeLog.fail_times > 0:
            FakeLog.fail_times -= 1
            raise TimeoutError("NBA.com timed out")
        if kwargs["season"] == FakeLog.fail_season:
            raise TimeoutError("NBA.com is down")

    def get_data_frames(self) -> list[pd.DataFrame]:
        return [FakeLog.frame]


@pytest.fixture(autouse=True)
def fake_nba(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLog.calls = []
    FakeLog.fail_times = 0
    FakeLog.fail_season = ""
    FakeLog.frame = team_rows(["001", "002"])
    monkeypatch.setattr(fg.leaguegamelog, "LeagueGameLog", FakeLog)
    monkeypatch.setattr(fg.time, "sleep", lambda _: None)
    monkeypatch.setattr(fg, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fg, "current_season", lambda today=None: CURRENT)


def run(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["fetch_games.py", *args])
    fg.main()


def requested_seasons() -> list[object]:
    return [call["season"] for call in FakeLog.calls]


def requested_types() -> list[object]:
    return [call["season_type_all_star"] for call in FakeLog.calls]


# ---------- where files go ----------

def test_saves_inside_the_basketball_folder() -> None:
    assert fg.find_basketball_dir().name == "basketball"


# ---------- season helpers ----------

def test_seasons_between_includes_both_ends() -> None:
    assert fg.seasons_between("2018-19", "2021-22") == ["2018-19", "2019-20", "2020-21", "2021-22"]


def test_season_name_wraps_century() -> None:
    assert fg.season_name(1999) == "1999-00"


@pytest.mark.parametrize("bad", ["2022", "2022-24", "22-23", "abcd-ef"])
def test_bad_season_format_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        fg.start_year(bad)


@pytest.mark.parametrize("today, season", [
    (date(2026, 11, 13), "2026-27"),
    (date(2027, 3, 1), "2026-27"),
    (date(2027, 6, 15), "2026-27"),   # finals
    (date(2027, 9, 1), "2027-28"),
])
def test_current_season(today: date, season: str) -> None:
    assert REAL_CURRENT_SEASON(today) == season


# ---------- downloading ----------

def test_downloads_every_season_and_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run(monkeypatch, "--first", "2019-20", "--last", "2020-21")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "games_2019-20_playoffs.csv", "games_2019-20_regular.csv",
        "games_2020-21_playin.csv", "games_2020-21_playoffs.csv", "games_2020-21_regular.csv",
    ]


def test_no_play_in_request_before_2020_21(monkeypatch: pytest.MonkeyPatch) -> None:
    run(monkeypatch, "--first", "2018-19", "--last", "2019-20")
    assert "PlayIn" not in requested_types()


def test_default_range_runs_from_2018_to_current(monkeypatch: pytest.MonkeyPatch) -> None:
    run(monkeypatch, "--types", "regular")
    seasons = requested_seasons()
    assert seasons[0] == "2018-19"
    assert seasons[-1] == CURRENT
    assert len(seasons) == 9


def test_asks_nba_com_for_team_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    run(monkeypatch, "--first", "2022-23", "--last", "2022-23", "--types", "playoffs")
    assert FakeLog.calls == [{
        "season": "2022-23",
        "season_type_all_star": "Playoffs",
        "player_or_team_abbreviation": "T",
        "timeout": fg.TIMEOUT_SECONDS,
    }]


def test_skips_finished_seasons_on_disk_but_refreshes_current(monkeypatch: pytest.MonkeyPatch) -> None:
    args = ("--first", "2025-26", "--last", CURRENT, "--types", "regular")
    run(monkeypatch, *args)
    assert requested_seasons() == ["2025-26", CURRENT]
    FakeLog.calls = []
    run(monkeypatch, *args)
    assert requested_seasons() == [CURRENT]          # finished season skipped, current refreshed
    FakeLog.calls = []
    run(monkeypatch, *args, "--force")
    assert requested_seasons() == ["2025-26", CURRENT]


def test_current_season_with_no_games_is_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLog.frame = pd.DataFrame(columns=["GAME_ID"])
    run(monkeypatch, "--first", CURRENT, "--last", CURRENT, "--types", "playoffs")
    assert not any(tmp_path.iterdir())


def test_empty_finished_season_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeLog.frame = pd.DataFrame(columns=["GAME_ID"])
    with pytest.raises(SystemExit):
        run(monkeypatch, "--first", "2022-23", "--last", "2022-23", "--types", "regular")


def test_retries_after_timeouts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLog.fail_times = 2
    run(monkeypatch, "--first", "2022-23", "--last", "2022-23", "--types", "regular")
    assert len(FakeLog.calls) == 3
    assert (tmp_path / "games_2022-23_regular.csv").exists()


def test_failure_keeps_earlier_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLog.fail_season = "2022-23"
    with pytest.raises(SystemExit):
        run(monkeypatch, "--first", "2021-22", "--last", "2022-23", "--types", "regular")
    assert [p.name for p in tmp_path.iterdir()] == ["games_2021-22_regular.csv"]


def test_rejects_game_with_one_team_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLog.frame = team_rows(["001"]).iloc[:1]
    with pytest.raises(SystemExit):
        run(monkeypatch, "--first", "2022-23", "--last", "2022-23", "--types", "regular")
    assert not any(tmp_path.iterdir())


def test_first_after_last_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        run(monkeypatch, "--first", "2023-24", "--last", "2022-23")
