import pandas as pd

from src.data.splits import (
    chronological_train_test_split,
    rolling_origin_splits,
    summarize_split,
)


def make_split_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m4", "m1", "m3", "m2", "m5", "m6"],
            "match_date": [
                "2022-01-01",
                "2020-01-01",
                "2021-12-31",
                "2023-06-01",
                "2022-07-15",
                "2024-01-01",
            ],
            "result": [
                "draw",
                "team_a_win",
                "team_b_win",
                "team_a_win",
                "draw",
                "team_b_win",
            ],
        }
    )


def test_chronological_split_places_pre_test_start_dates_in_train() -> None:
    train_df, test_df = chronological_train_test_split(
        make_split_df(),
        test_start_date="2022-01-01",
    )

    assert train_df["match_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-01",
        "2021-12-31",
    ]
    assert test_df["match_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2022-01-01",
        "2022-07-15",
        "2023-06-01",
        "2024-01-01",
    ]


def test_rows_exactly_on_test_start_date_go_to_test() -> None:
    _, test_df = chronological_train_test_split(
        make_split_df(),
        test_start_date="2022-01-01",
    )

    assert "2022-01-01" in test_df["match_date"].dt.strftime("%Y-%m-%d").tolist()


def test_chronological_split_does_not_mutate_input_dataframe() -> None:
    df = make_split_df()
    original = df.copy(deep=True)

    chronological_train_test_split(df)

    pd.testing.assert_frame_equal(df, original)


def test_chronological_split_output_is_sorted_by_match_date() -> None:
    train_df, test_df = chronological_train_test_split(make_split_df())

    assert train_df["match_date"].is_monotonic_increasing
    assert test_df["match_date"].is_monotonic_increasing


def test_rolling_origin_splits_produce_non_overlapping_future_test_windows() -> None:
    splits = rolling_origin_splits(
        make_split_df(),
        initial_train_end_date="2020-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2023-12-31",
    )

    test_windows = [
        (split["test_start_date"], split["test_end_date"]) for split in splits
    ]

    assert test_windows == [
        ("2021-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
    ]


def test_rolling_origin_train_data_always_ends_before_test_data_begins() -> None:
    splits = rolling_origin_splits(
        make_split_df(),
        initial_train_end_date="2020-12-31",
        test_window_months=12,
        step_months=12,
        final_test_end_date="2023-12-31",
    )

    for split in splits:
        train_df = split["train_df"]
        test_df = split["test_df"]
        assert train_df["match_date"].max() < test_df["match_date"].min()


def test_summarize_split_returns_expected_counts_and_date_ranges() -> None:
    train_df, test_df = chronological_train_test_split(
        make_split_df(),
        test_start_date="2022-01-01",
    )

    summary = summarize_split(train_df, test_df)

    assert summary["train_count"] == 2
    assert summary["test_count"] == 4
    assert summary["train_start_date"] == "2020-01-01"
    assert summary["train_end_date"] == "2021-12-31"
    assert summary["test_start_date"] == "2022-01-01"
    assert summary["test_end_date"] == "2024-01-01"
    assert summary["train_result_distribution"] == {
        "team_a_win": 1,
        "team_b_win": 1,
    }
    assert summary["test_result_distribution"] == {
        "draw": 2,
        "team_a_win": 1,
        "team_b_win": 1,
    }
