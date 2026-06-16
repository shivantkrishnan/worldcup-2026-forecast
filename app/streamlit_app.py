"""Streamlit app shell for the World Cup 2026 match predictor."""

import streamlit as st


def main() -> None:
    """Render the initial dashboard shell."""
    st.set_page_config(
        page_title="World Cup 2026 Forecasting Dashboard",
        page_icon="WC",
        layout="wide",
    )

    st.title("World Cup 2026 Forecasting Dashboard")
    st.caption("Leakage-safe international football match outcome forecasting.")

    team_a = st.text_input("Team A", value="United States")
    team_b = st.text_input("Team B", value="Canada")

    st.subheader("Match Predictor")
    st.info(
        "The baseline model is not trained yet. The next milestone is to add "
        "cleaned data, leakage-safe features, and calibrated probabilities."
    )

    st.write(
        {
            "team_a": team_a,
            "team_b": team_b,
            "prediction_status": "not_trained",
        }
    )


if __name__ == "__main__":
    main()
