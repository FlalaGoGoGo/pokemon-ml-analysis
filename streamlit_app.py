from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pokemon_project import DEPLOY_BUNDLE_PATH, get_deploy_bundle, notebook_ready_tables, predict_battle, predict_types


APP_ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=True)
def load_bundle() -> dict:
    return get_deploy_bundle(DEPLOY_BUNDLE_PATH)


def _pokemon_options(master_df: pd.DataFrame) -> list[str]:
    ordered = master_df.sort_values(["dexnum_int", "name"]).copy()
    return [f"{int(row.dexnum_int):03d} - {row.name}" for row in ordered.itertuples()]


def _name_from_option(option: str) -> str:
    return option.split(" - ", 1)[1].strip()


def render_sidebar(bundle: dict) -> None:
    tables = notebook_ready_tables(bundle)
    st.sidebar.title("Pokemon ML Rebuild")
    st.sidebar.caption("The app prefers a prebuilt deploy artifact so Streamlit Cloud can start quickly.")
    st.sidebar.write(f"Bundle source: `{bundle.get('bundle_source', 'unknown')}`")
    st.sidebar.markdown("**Final models**")
    st.sidebar.write(f"Type prediction: `{bundle['type_bundle']['final_model_name']}`")
    st.sidebar.write(f"Battle prediction: `{bundle['battle_bundle']['final_model_name']}`")
    st.sidebar.markdown("**Core metrics**")
    st.sidebar.dataframe(
        tables["type_random"][["model", "micro_f1", "macro_f1", "exact_match"]].round(3),
        use_container_width=True,
        hide_index=True,
    )
    st.sidebar.dataframe(
        tables["battle_grouped"][["model", "accuracy", "roc_auc"]].round(3),
        use_container_width=True,
        hide_index=True,
    )


def render_type_page(bundle: dict) -> None:
    master_df = bundle["master_df"]
    options = _pokemon_options(master_df)
    st.title("Type Predictor")
    st.write(
        "Choose a Pokemon and the upgraded classifier chain model will predict its primary and secondary type "
        "from stats and structured metadata."
    )

    selection = st.selectbox("Pokemon", options, index=5)
    result = predict_types(_name_from_option(selection), bundle).payload

    left, right = st.columns([1, 2])
    with left:
        if result["image_url"]:
            st.image(result["image_url"], caption=result["name"], use_container_width=True)
        st.metric("True primary", result["true_primary"])
        st.metric("True secondary", result["true_secondary"])
    with right:
        st.metric("Predicted primary", result["predicted_primary"])
        st.metric("Predicted secondary", result["predicted_secondary"])
        st.caption(f"Model: {result['model_name']}")
        st.dataframe(result["probabilities"].head(8), hide_index=True, use_container_width=True)
        st.bar_chart(result["probabilities"].set_index("type").head(8))

    st.subheader("Profile Explanation")
    for line in result["explanation"]:
        st.write(f"- {line}")


def render_battle_page(bundle: dict) -> None:
    master_df = bundle["master_df"]
    options = _pokemon_options(master_df)
    st.title("Battle Predictor")
    st.write(
        "Pick two Pokemon and compare the upgraded 1v1 model's predicted winner, win probabilities, "
        "battle-history lookup, and the main matchup signals."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        option_a = st.selectbox("Pokemon A", options, index=5)
    with col_b:
        option_b = st.selectbox("Pokemon B", options, index=8)

    result = predict_battle(_name_from_option(option_a), _name_from_option(option_b), bundle).payload

    display_a, display_b = st.columns(2)
    with display_a:
        if result["pokemon_a"]["image_url"]:
            st.image(result["pokemon_a"]["image_url"], caption=result["pokemon_a"]["name"], use_container_width=True)
        st.write(
            f"**{result['pokemon_a']['name']}**  \n"
            f"Types: {result['pokemon_a']['type1']} / {result['pokemon_a']['type2']}  \n"
            f"Base total: {result['pokemon_a']['total']}"
        )
    with display_b:
        if result["pokemon_b"]["image_url"]:
            st.image(result["pokemon_b"]["image_url"], caption=result["pokemon_b"]["name"], use_container_width=True)
        st.write(
            f"**{result['pokemon_b']['name']}**  \n"
            f"Types: {result['pokemon_b']['type1']} / {result['pokemon_b']['type2']}  \n"
            f"Base total: {result['pokemon_b']['total']}"
        )

    win_col, history_col = st.columns([1, 1])
    with win_col:
        st.metric("Predicted winner", result["predicted_winner"])
        st.metric("P(A wins)", f"{result['win_prob_a']:.1%}")
        st.metric("P(B wins)", f"{result['win_prob_b']:.1%}")
    with history_col:
        st.caption("Historical single-battle lookup")
        history = result["history"]
        st.write(f"Recorded battles: {history['total_battles']}")
        if history["total_battles"]:
            st.write(
                f"A wins: {history['wins_a']} ({history['rate_a']:.1%})  \n"
                f"B wins: {history['wins_b']} ({history['rate_b']:.1%})"
            )
        else:
            st.write("No historical battles between this pair in the local dataset.")

    st.subheader("Feature Snapshot")
    st.dataframe(result["feature_snapshot"], hide_index=True, use_container_width=True)

    st.subheader("Matchup Explanation")
    for line in result["explanation"]:
        st.write(f"- {line}")


def main() -> None:
    st.set_page_config(page_title="Pokemon ML Rebuild", page_icon="⚔️", layout="wide")
    bundle = load_bundle()
    render_sidebar(bundle)
    page = st.radio("Page", ["Type Predictor", "Battle Predictor"], horizontal=True)
    if page == "Type Predictor":
        render_type_page(bundle)
    else:
        render_battle_page(bundle)


if __name__ == "__main__":
    main()
