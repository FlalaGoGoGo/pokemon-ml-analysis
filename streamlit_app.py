from __future__ import annotations

from html import escape
import unicodedata

import pandas as pd
import streamlit as st

from pokemon_project import DEPLOY_BUNDLE_PATH, get_deploy_bundle, notebook_ready_tables, predict_battle, predict_types


TYPE_STYLE_MAP = {
    "Normal": ("#C6C4B4", "#1B1B1B"),
    "Fire": ("#E76F51", "#FFF7E3"),
    "Water": ("#4F8FDC", "#FFF7E3"),
    "Electric": ("#F7D64A", "#1B1B1B"),
    "Grass": ("#67B85F", "#FFF7E3"),
    "Ice": ("#A7F0F2", "#10203C"),
    "Fighting": ("#B4473E", "#FFF7E3"),
    "Poison": ("#8B5FBF", "#FFF7E3"),
    "Ground": ("#C49B58", "#10203C"),
    "Flying": ("#7AA7FF", "#10203C"),
    "Psychic": ("#F36CA8", "#10203C"),
    "Bug": ("#9EB33E", "#10203C"),
    "Rock": ("#8A7B60", "#FFF7E3"),
    "Ghost": ("#5B5BA8", "#FFF7E3"),
    "Dragon": ("#3A5EDB", "#FFF7E3"),
    "Dark": ("#3D3A3A", "#FFF7E3"),
    "Steel": ("#8CA0AF", "#10203C"),
    "Fairy": ("#F3B3DE", "#10203C"),
    "None": ("#72808E", "#FFF7E3"),
}


@st.cache_resource(show_spinner=True)
def load_bundle() -> dict:
    return get_deploy_bundle(DEPLOY_BUNDLE_PATH)


def _pokemon_options(master_df: pd.DataFrame) -> list[str]:
    ordered = master_df.sort_values(["dexnum_int", "name"]).copy()
    return [f"{int(row.dexnum_int):03d} - {row.name}" for row in ordered.itertuples()]


def _name_from_option(option: str) -> str:
    return option.split(" - ", 1)[1].strip()


def _ascii_text(value: object) -> str:
    text = "" if value is None else str(value)
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalized or text


def _mount_html(html: str, container=None) -> None:
    target = container or st
    if hasattr(target, "html"):
        target.html(html)
    else:
        target.markdown(html, unsafe_allow_html=True)


def _render_type_badges(types: list[str]) -> str:
    badges = []
    for type_name in types:
        if not type_name or type_name == "None":
            continue
        bg_color, text_color = TYPE_STYLE_MAP.get(type_name, ("#72808E", "#FFF7E3"))
        badges.append(
            f"<span class='type-badge' style='background:{bg_color};color:{text_color};'>{escape(type_name.upper())}</span>"
        )
    return "".join(badges) if badges else "<span class='type-badge badge-empty'>NO SECOND TYPE</span>"


def _render_probability_meter(probability_df: pd.DataFrame, top_n: int = 6) -> str:
    rows = []
    for row in probability_df.head(top_n).itertuples(index=False):
        label = str(row.type)
        value = float(row.probability)
        bg_color, text_color = TYPE_STYLE_MAP.get(label, ("#72808E", "#FFF7E3"))
        rows.append(
            f"""
            <div class="meter-row">
              <div class="meter-row-label">{_render_type_badges([label])}</div>
              <div class="meter-track">
                <div class="meter-fill" style="width:{value * 100:.1f}%;background:{bg_color};"></div>
              </div>
              <div class="meter-row-value" style="color:{text_color};background:{bg_color};">{value:.1%}</div>
            </div>
            """
        )
    return "<div class='meter-stack'>" + "".join(rows) + "</div>"


def _render_commentary_lines(lines: list[str], panel_title: str, tone: str = "neutral") -> str:
    items = "".join(
        f"<div class='log-line {tone}'><span class='log-bullet'>&gt;</span>{escape(_ascii_text(line))}</div>" for line in lines
    )
    return f"""
    <div class="pixel-panel log-panel">
      <div class="panel-kicker">{escape(panel_title)}</div>
      <div class="log-shell">{items}</div>
    </div>
    """


def _render_stat_strip(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f"""
        <div class="mini-stat">
          <div class="mini-stat-label">{escape(_ascii_text(label))}</div>
          <div class="mini-stat-value">{escape(_ascii_text(value))}</div>
        </div>
        """
        for label, value in items
    )
    return f"<div class='mini-stat-grid'>{cells}</div>"


def _render_sidebar_table(df: pd.DataFrame, columns: list[str], value_formats: dict[str, str]) -> str:
    grid_template = " ".join(["1fr" for _ in columns])
    header = "".join(f"<div class='side-table-head'>{escape(column.replace('_', ' ').title())}</div>" for column in columns)
    rows = []
    for row in df.itertuples(index=False):
        cells = []
        for column in columns:
            value = getattr(row, column)
            if isinstance(value, float):
                fmt = value_formats.get(column, ".3f")
                display_value = format(value, fmt)
            else:
                display_value = _ascii_text(value)
            cells.append(f"<div class='side-table-cell'>{escape(display_value)}</div>")
        rows.append(f"<div class='side-table-row' style='grid-template-columns:{grid_template};'>" + "".join(cells) + "</div>")
    return (
        "<div class='side-table'>"
        + f"<div class='side-table-row side-table-header' style='grid-template-columns:{grid_template};'>"
        + header
        + "</div>"
        + "".join(rows)
        + "</div>"
    )


def _render_type_oof_summary_card(bundle: dict) -> str:
    summary_df = bundle.get("type_oof_summary")
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return """
        <div class="pixel-panel oof-panel">
          <div class="panel-kicker">BENCHMARK SNAPSHOT</div>
          <div class="panel-title">TYPE OOF SUMMARY</div>
          <div class="mode-copy">OOF summary is not available in the current bundle.</div>
        </div>
        """

    grouped_df = summary_df[summary_df["split_mode"] == "grouped"].copy()
    if grouped_df.empty:
        grouped_df = summary_df.copy()
    row = grouped_df.iloc[0]
    total = int(row["n_total"])
    all_correct_n = int(row["all_types_correct_n"])
    one_correct_n = int(row["one_type_correct_n"])
    zero_correct_n = int(row["zero_type_correct_n"])

    return f"""
    <div class="pixel-panel oof-panel">
      <div class="panel-kicker">BENCHMARK SNAPSHOT</div>
      <div class="panel-title">GROUPED OOF TYPE SUMMARY</div>
      {_render_stat_strip([
          ("TOTAL", str(total)),
          ("ALL CORRECT", str(all_correct_n)),
          ("ONE TYPE", str(one_correct_n)),
          ("ZERO TYPE", str(zero_correct_n)),
      ])}
      <div class="oof-meter-shell">
        <div class="oof-meter-row">
          <div class="oof-meter-label">ALL TYPES CORRECT</div>
          <div class="meter-track"><div class="meter-fill" style="width:{float(row['all_types_correct_pct']) * 100:.1f}%;background:#67B85F;"></div></div>
          <div class="meter-row-value" style="background:#67B85F;color:#FFF7E3;">{float(row['all_types_correct_pct']):.1%}</div>
        </div>
        <div class="oof-meter-row">
          <div class="oof-meter-label">ONE TYPE CORRECT</div>
          <div class="meter-track"><div class="meter-fill" style="width:{float(row['one_type_correct_pct']) * 100:.1f}%;background:#F7D64A;"></div></div>
          <div class="meter-row-value" style="background:#F7D64A;color:#10203C;">{float(row['one_type_correct_pct']):.1%}</div>
        </div>
        <div class="oof-meter-row">
          <div class="oof-meter-label">ZERO TYPE CORRECT</div>
          <div class="meter-track"><div class="meter-fill" style="width:{float(row['zero_type_correct_pct']) * 100:.1f}%;background:#D94A3A;"></div></div>
          <div class="meter-row-value" style="background:#D94A3A;color:#FFF7E3;">{float(row['zero_type_correct_pct']):.1%}</div>
        </div>
      </div>
      <div class="note-card">This card uses grouped OOF evaluation, so each Pokemon is predicted by a model that did not train on that Pokemon.</div>
    </div>
    """


def _render_app_shell(bundle: dict, page_label: str) -> None:
    type_reports = bundle.get("type_reports")
    battle_reports = bundle.get("battle_reports")

    if isinstance(type_reports, pd.DataFrame) and not type_reports.empty:
        type_random = type_reports[type_reports["split"] == "random"].copy()
        if type_random.empty:
            type_random = type_reports.copy()
        type_report = type_random.sort_values(["exact_match", "micro_f1"], ascending=[False, False]).iloc[0]
        type_exact_match = float(type_report.get("exact_match", 0.0))
    else:
        type_exact_match = 0.0

    if isinstance(battle_reports, pd.DataFrame) and not battle_reports.empty:
        battle_grouped = battle_reports[battle_reports["split"] == "grouped"].copy()
        if battle_grouped.empty:
            battle_grouped = battle_reports.copy()
        battle_report = battle_grouped.sort_values(["roc_auc", "accuracy"], ascending=[False, False]).iloc[0]
        battle_roc_auc = float(battle_report.get("roc_auc", 0.0))
    else:
        battle_roc_auc = 0.0

    app_html = f"""
    <div class="app-shell start-shell">
      <div class="status-ribbon">
        <span>TRAINER DEVICE ONLINE</span>
        <span>BUNDLE :: {escape(str(bundle.get('bundle_source', 'unknown')).upper())}</span>
        <span>MODE :: {escape(page_label.upper())}</span>
      </div>
      <div class="start-grid">
        <div class="hero-copy start-copy">
          <div class="panel-kicker">BOOT SEQUENCE COMPLETE</div>
          <div class="start-subkicker">TRAINER ANALYSIS SYSTEM v2.0</div>
          <h1>POKEMON ML ANALYSIS</h1>
          <div class="game-logo-bar">POKEDEX SCAN // BATTLE FORECAST // RETRO HUD</div>
          <p>
            A retro handheld game interface for type prediction and one versus one battle forecasting.
          </p>
          <div class="press-start">PRESS START TO ENTER {escape(page_label.upper())}</div>
        </div>
        <div class="boot-monitor">
          <div class="boot-screen">
            <div class="screen-title">MISSION STATUS</div>
            <div class="screen-line">TYPE MODEL :: {escape(_ascii_text(bundle['type_bundle']['final_model_name']).upper())}</div>
            <div class="screen-line">BATTLE MODEL :: {escape(_ascii_text(bundle['battle_bundle']['final_model_name']).upper())}</div>
            <div class="screen-line">TYPE EXACT MATCH :: {type_exact_match:.3f}</div>
            <div class="screen-line">BATTLE ROC AUC :: {battle_roc_auc:.3f}</div>
          </div>
        </div>
      </div>
      <div class="launch-strip">
        <span class="launch-chip chip-deploy">CURRENT APP :: DEPLOYMENT MODE</span>
        <span class="launch-chip chip-benchmark">BENCHMARK MODE :: HELD-OUT SPLIT + OOF STATS</span>
        <span class="launch-chip">PIXEL UI ACTIVE</span>
      </div>
    </div>
    """
    _mount_html(app_html)


def _render_mode_menu(page_label: str) -> None:
    menu_html = f"""
    <div class="pixel-panel mode-panel">
      <div class="panel-kicker">MODE SELECT</div>
      <div class="panel-title">PRESS START AND CHOOSE YOUR GAME SCREEN</div>
      <div class="mode-copy">Current selection :: {escape(page_label.upper())}</div>
      <div class="mode-legend">
        <span class="legend-badge benchmark">BENCHMARK MODE = NOTEBOOK SCORES FROM HELD-OUT OR OOF EVALUATION</span>
        <span class="legend-badge deployment">DEPLOYMENT MODE = LIVE APP PREDICTION FROM THE DEPLOYED MODEL</span>
      </div>
    </div>
    """
    _mount_html(menu_html)


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

        :root {
          --cartridge-cream: #FFF7E3;
          --shell-navy: #10203C;
          --shell-navy-dark: #0D1321;
          --screen-teal: #99E3D4;
          --battle-red: #D94A3A;
          --electric-yellow: #F7D64A;
          --grass-green: #67B85F;
          --steel-gray: #72808E;
          --ink: #1B1B1B;
          --panel-shadow: 6px 6px 0 #0D1321;
          --panel-border: 4px solid #10203C;
        }

        html, body, .stApp, [data-testid="stMarkdownContainer"], p, label, li, input, textarea {
          font-family: "VT323", "Courier New", monospace !important;
        }

        .app-shell *,
        .pixel-panel *,
        .sidebar-shell *,
        .note-card,
        .side-table *,
        .hero-data *,
        .fighter-card *,
        .log-shell *,
        .mini-stat * {
          font-family: "VT323", "Courier New", monospace !important;
        }

        h1, h2, h3, .panel-kicker, .status-ribbon, .hero-chip, .type-badge, .menu-note, .mini-stat-label {
          font-family: "Press Start 2P", "Courier New", monospace !important;
        }

        .stApp {
          background:
            linear-gradient(rgba(255,255,255,0.05) 50%, rgba(0,0,0,0.02) 50%),
            radial-gradient(circle at top left, rgba(153, 227, 212, 0.25), transparent 35%),
            radial-gradient(circle at bottom right, rgba(247, 214, 74, 0.18), transparent 30%),
            linear-gradient(180deg, #fdf7e6 0%, #f8f0d8 100%);
          background-size: 100% 6px, auto, auto, auto;
          color: var(--ink);
        }

        .stApp::before {
          content: "";
          position: fixed;
          inset: 0;
          background-image:
            linear-gradient(rgba(13,19,33,0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(13,19,33,0.05) 1px, transparent 1px);
          background-size: 18px 18px;
          pointer-events: none;
          opacity: 0.35;
          z-index: 0;
        }

        [data-testid="stHeader"] {
          background: transparent;
        }

        #MainMenu, footer {
          visibility: hidden;
        }

        .block-container {
          position: relative;
          z-index: 1;
          max-width: 1280px;
          padding-top: 1.5rem;
          padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
          background:
            linear-gradient(180deg, rgba(16,32,60,0.96), rgba(13,19,33,0.97));
          border-right: 6px solid #0A1020;
        }

        section[data-testid="stSidebar"] .block-container {
          padding-top: 1.4rem;
        }

        section[data-testid="stSidebar"] * {
          color: #FFF7E3 !important;
        }

        div[data-baseweb="select"] > div {
          background: rgba(255,247,227,0.92);
          border: var(--panel-border);
          border-radius: 0 !important;
          box-shadow: var(--panel-shadow);
          min-height: 62px;
        }

        div[data-baseweb="select"] input,
        div[data-baseweb="select"] [role="combobox"] {
          color: var(--shell-navy) !important;
          font-size: 1.35rem !important;
          letter-spacing: 0.02em;
          font-family: "VT323", "Courier New", monospace !important;
        }

        div[data-baseweb="popover"] [role="listbox"] {
          background: #FFF7E3;
          border: var(--panel-border);
          border-radius: 0 !important;
        }

        div[data-testid="stRadio"] > div {
          flex-direction: row;
          gap: 0.8rem;
        }

        div[data-testid="stRadio"] label {
          background: rgba(153,227,212,0.35);
          border: var(--panel-border);
          box-shadow: var(--panel-shadow);
          border-radius: 0 !important;
          padding: 0.8rem 1rem;
          min-width: 220px;
          justify-content: center;
        }

        div[data-testid="stRadio"] label > div:first-child {
          display: none;
        }

        div[data-testid="stRadio"] label p {
          font-size: 0.76rem !important;
          letter-spacing: 0.08em;
          color: var(--shell-navy) !important;
          margin: 0;
        }

        div[data-testid="stRadio"] label:has(input:checked) {
          background: var(--electric-yellow);
          transform: translate(2px, 2px);
          box-shadow: 3px 3px 0 #0D1321;
        }

        .app-shell,
        .pixel-panel,
        .battle-shell,
        .sidebar-shell {
          background: rgba(255, 247, 227, 0.92);
          border: var(--panel-border);
          box-shadow: var(--panel-shadow);
          position: relative;
          overflow: hidden;
        }

        .app-shell::after,
        .pixel-panel::after,
        .battle-shell::after,
        .sidebar-shell::after {
          content: "";
          position: absolute;
          inset: 8px;
          border: 2px solid rgba(16,32,60,0.16);
          pointer-events: none;
        }

        .app-shell {
          margin-bottom: 1.2rem;
        }

        .start-shell {
          overflow: visible;
        }

        .start-grid {
          display: grid;
          grid-template-columns: 1.7fr 1fr;
          gap: 1rem;
          align-items: stretch;
          padding: 1.25rem;
          background:
            linear-gradient(135deg, rgba(153,227,212,0.38), rgba(255,247,227,0.94) 55%),
            linear-gradient(180deg, rgba(16,32,60,0.03), rgba(16,32,60,0.0));
        }

        .start-copy {
          display: flex;
          flex-direction: column;
          justify-content: center;
          min-height: 300px;
        }

        .start-subkicker {
          display: inline-block;
          margin-bottom: 0.8rem;
          color: #243550;
          font-size: 1.25rem;
          letter-spacing: 0.08em;
        }

        .game-logo-bar {
          display: inline-block;
          margin: 0.25rem 0 0.8rem;
          padding: 0.7rem 0.85rem;
          border: 3px solid rgba(16,32,60,0.3);
          background: rgba(255,247,227,0.88);
          box-shadow: inset 0 0 0 3px rgba(16,32,60,0.08);
          color: var(--shell-navy);
          font-size: 1.2rem;
          letter-spacing: 0.05em;
        }

        .press-start {
          display: inline-block;
          margin-top: 1rem;
          padding: 0.85rem 1rem;
          background: var(--battle-red);
          color: var(--cartridge-cream);
          border: 4px solid #7F2218;
          box-shadow: 4px 4px 0 rgba(13,19,33,0.4);
          font-family: "Press Start 2P", "Courier New", monospace !important;
          font-size: 0.62rem;
          letter-spacing: 0.1em;
          animation: pulseStart 1.8s steps(2, end) infinite;
        }

        .boot-monitor {
          display: flex;
          align-items: stretch;
        }

        .boot-screen {
          width: 100%;
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.75rem;
          padding: 1rem;
          background:
            linear-gradient(180deg, rgba(16,32,60,0.08) 50%, rgba(16,32,60,0.02) 50%),
            linear-gradient(180deg, #bdf7ea, #8edacb);
          background-size: 100% 8px, auto;
          border: var(--panel-border);
          box-shadow: inset 0 0 0 4px rgba(255,255,255,0.35), var(--panel-shadow);
          min-height: 300px;
        }

        .screen-title {
          color: var(--battle-red);
          font-family: "Press Start 2P", "Courier New", monospace !important;
          font-size: 0.62rem;
          letter-spacing: 0.08em;
          margin-bottom: 0.3rem;
        }

        .screen-line {
          font-size: 1.5rem;
          line-height: 1.08;
          color: var(--shell-navy);
          padding: 0.45rem 0.55rem;
          border: 3px solid rgba(16,32,60,0.22);
          background: rgba(255,247,227,0.58);
        }

        .launch-strip {
          display: flex;
          flex-wrap: wrap;
          gap: 0.8rem;
          padding: 0 1.25rem 1.2rem;
          background: linear-gradient(180deg, rgba(255,247,227,0.95), rgba(255,247,227,0.86));
        }

        .launch-chip {
          display: inline-flex;
          align-items: center;
          padding: 0.6rem 0.8rem;
          border: 3px solid rgba(16,32,60,0.28);
          background: rgba(16,32,60,0.08);
          color: var(--shell-navy);
          font-family: "Press Start 2P", "Courier New", monospace !important;
          font-size: 0.54rem;
          letter-spacing: 0.06em;
        }

        .chip-benchmark {
          background: rgba(103,184,95,0.18);
          border-color: rgba(71,124,66,0.5);
        }

        .chip-deploy {
          background: rgba(217,74,58,0.16);
          border-color: rgba(151,44,34,0.5);
        }

        .status-ribbon {
          display: flex;
          justify-content: space-between;
          gap: 0.8rem;
          flex-wrap: wrap;
          background: var(--shell-navy);
          color: var(--cartridge-cream);
          padding: 0.8rem 1rem;
          font-size: 0.66rem;
          letter-spacing: 0.08em;
        }

        .app-hero {
          display: grid;
          grid-template-columns: 2.1fr 1fr;
          gap: 1rem;
          align-items: center;
          padding: 1.25rem;
          background: linear-gradient(135deg, rgba(153,227,212,0.35), rgba(255,247,227,0.95));
        }

        .hero-copy h1 {
          margin: 0.5rem 0 0.75rem;
          font-size: 1.28rem;
          line-height: 1.5;
          color: var(--shell-navy);
        }

        .hero-copy p {
          margin: 0;
          font-size: 1.5rem;
          line-height: 1.2;
          color: #243550;
        }

        .panel-kicker {
          font-size: 0.6rem;
          letter-spacing: 0.12em;
          color: var(--battle-red);
          margin-bottom: 0.4rem;
        }

        .hero-chips {
          display: flex;
          flex-direction: column;
          gap: 0.7rem;
        }

        .hero-chip {
          display: block;
          background: var(--shell-navy);
          color: var(--cartridge-cream);
          border: 3px solid #081123;
          box-shadow: 4px 4px 0 rgba(13,19,33,0.55);
          padding: 0.85rem 1rem;
          font-size: 0.6rem;
          letter-spacing: 0.08em;
          line-height: 1.6;
        }

        .page-shell {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .page-hero {
          display: grid;
          grid-template-columns: 1.35fr 0.8fr;
          gap: 1rem;
          align-items: center;
          padding: 1.2rem;
        }

        .page-hero.compact {
          grid-template-columns: 1fr;
        }

        .hero-screen {
          display: flex;
          gap: 1rem;
          align-items: center;
        }

        .hero-sprite {
          width: 180px;
          min-width: 180px;
          height: 180px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(180deg, #b9fff0, #85d2c3);
          border: var(--panel-border);
          box-shadow: inset 0 0 0 4px rgba(255,255,255,0.45);
        }

        .hero-sprite img,
        .fighter-sprite img {
          width: 100%;
          height: 100%;
          object-fit: contain;
          image-rendering: pixelated;
        }

        .hero-data h2,
        .fighter-card h3 {
          margin: 0.35rem 0 0.6rem;
          font-size: 1.1rem;
          line-height: 1.55;
          color: var(--shell-navy);
        }

        .hero-data p,
        .fighter-card p {
          font-size: 1.4rem;
          line-height: 1.18;
          margin: 0;
        }

        .hero-status {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          background: var(--battle-red);
          color: var(--cartridge-cream);
          padding: 0.4rem 0.7rem;
          border: 3px solid #7F2218;
          margin-bottom: 0.55rem;
          font-size: 0.58rem;
          letter-spacing: 0.08em;
        }

        .pixel-panel {
          padding: 1rem;
          margin-bottom: 1rem;
        }

        .oof-panel {
          background: linear-gradient(135deg, rgba(255,247,227,0.96), rgba(103,184,95,0.10));
        }

        .mode-panel {
          background: linear-gradient(135deg, rgba(255,247,227,0.95), rgba(153,227,212,0.28));
          margin-bottom: 0.35rem;
        }

        .mode-copy {
          font-size: 1.35rem;
          color: #243550;
          line-height: 1.1;
        }

        .mode-legend {
          display: flex;
          flex-direction: column;
          gap: 0.55rem;
          margin-top: 0.85rem;
        }

        .legend-badge {
          display: block;
          padding: 0.6rem 0.75rem;
          border: 3px solid rgba(16,32,60,0.24);
          font-size: 1.1rem;
          line-height: 1.12;
          color: #243550;
          background: rgba(16,32,60,0.06);
        }

        .legend-badge.benchmark {
          background: rgba(103,184,95,0.16);
          border-color: rgba(71,124,66,0.42);
        }

        .legend-badge.deployment {
          background: rgba(217,74,58,0.12);
          border-color: rgba(151,44,34,0.36);
        }

        .oof-meter-shell {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          margin-top: 0.9rem;
        }

        .oof-meter-row {
          display: grid;
          grid-template-columns: 160px 1fr 86px;
          gap: 0.7rem;
          align-items: center;
        }

        .oof-meter-label {
          font-size: 1.12rem;
          line-height: 1.05;
          color: var(--shell-navy);
        }

        .panel-title {
          margin: 0.2rem 0 0.8rem;
          color: var(--shell-navy);
          font-size: 0.9rem;
          line-height: 1.5;
        }

        .type-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 0.35rem 0.65rem;
          border: 3px solid rgba(0,0,0,0.28);
          box-shadow: 3px 3px 0 rgba(13,19,33,0.25);
          margin-right: 0.45rem;
          margin-bottom: 0.45rem;
          font-size: 0.56rem;
          letter-spacing: 0.06em;
          line-height: 1.35;
        }

        .badge-empty {
          background: var(--steel-gray);
          color: var(--cartridge-cream);
        }

        .mini-stat-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
          gap: 0.65rem;
          margin-top: 0.8rem;
        }

        .mini-stat {
          background: rgba(16,32,60,0.08);
          border: 3px solid rgba(16,32,60,0.3);
          padding: 0.55rem 0.65rem;
        }

        .mini-stat-label {
          font-size: 0.48rem;
          line-height: 1.4;
          color: var(--battle-red);
          margin-bottom: 0.25rem;
        }

        .mini-stat-value {
          font-size: 1.55rem;
          line-height: 1;
          color: var(--shell-navy);
        }

        .meter-stack {
          display: flex;
          flex-direction: column;
          gap: 0.7rem;
        }

        .meter-row {
          display: grid;
          grid-template-columns: 150px 1fr 86px;
          gap: 0.7rem;
          align-items: center;
        }

        .meter-track {
          height: 20px;
          background: rgba(16,32,60,0.13);
          border: 3px solid rgba(16,32,60,0.28);
          position: relative;
          overflow: hidden;
        }

        .meter-fill {
          height: 100%;
          box-shadow: inset 0 -3px 0 rgba(255,255,255,0.22);
        }

        .meter-row-value {
          text-align: center;
          border: 3px solid rgba(16,32,60,0.28);
          padding: 0.15rem 0.35rem;
          font-size: 1.35rem;
          line-height: 1;
        }

        .log-panel {
          background: linear-gradient(180deg, rgba(255,247,227,0.94), rgba(153,227,212,0.18));
        }

        .log-shell {
          background:
            linear-gradient(180deg, rgba(16,32,60,0.045) 50%, transparent 50%);
          background-size: 100% 10px;
          border: 3px solid rgba(16,32,60,0.22);
          padding: 0.75rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .log-line {
          font-size: 1.32rem;
          line-height: 1.1;
          display: flex;
          gap: 0.55rem;
          color: #243550;
        }

        .log-bullet {
          color: var(--battle-red);
        }

        .battle-hero {
          display: grid;
          grid-template-columns: 1fr 0.65fr 1fr;
          gap: 1rem;
          align-items: center;
          padding: 1rem;
        }

        .fighter-card {
          background: rgba(255,247,227,0.94);
          border: var(--panel-border);
          box-shadow: var(--panel-shadow);
          padding: 1rem;
        }

        .fighter-sprite {
          width: 100%;
          min-height: 170px;
          background: linear-gradient(180deg, rgba(153,227,212,0.6), rgba(255,247,227,0.95));
          border: 3px solid rgba(16,32,60,0.25);
          margin-bottom: 0.75rem;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .vs-core {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.9rem;
          text-align: center;
        }

        .vs-badge {
          width: 110px;
          height: 110px;
          border-radius: 50%;
          background: radial-gradient(circle at 30% 30%, #fff2b4, #F7D64A 55%, #C99E13 100%);
          border: 6px solid #10203C;
          box-shadow: 0 0 0 6px rgba(16,32,60,0.2), 6px 6px 0 #0D1321;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: "Press Start 2P", "Courier New", monospace !important;
          font-size: 1rem;
          color: #10203C;
        }

        .winner-banner {
          background: var(--battle-red);
          color: var(--cartridge-cream);
          border: 4px solid #7F2218;
          box-shadow: 4px 4px 0 rgba(13,19,33,0.45);
          padding: 0.75rem 0.95rem;
          width: 100%;
        }

        .winner-label {
          font-size: 0.55rem;
          letter-spacing: 0.08em;
          margin-bottom: 0.35rem;
        }

        .winner-name {
          font-size: 1.9rem;
          line-height: 1;
        }

        .hud-grid {
          display: grid;
          grid-template-columns: 1.1fr 0.9fr;
          gap: 1rem;
        }

        .battle-meter-shell {
          display: flex;
          flex-direction: column;
          gap: 0.85rem;
        }

        .battle-meter-row {
          display: grid;
          grid-template-columns: 120px 1fr 78px;
          gap: 0.7rem;
          align-items: center;
        }

        .battle-meter-label {
          font-size: 1.4rem;
          color: var(--shell-navy);
        }

        .feature-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 0.75rem;
        }

        .feature-tile {
          border: 3px solid rgba(16,32,60,0.28);
          padding: 0.7rem;
          background: rgba(16,32,60,0.08);
        }

        .feature-tile.positive {
          background: rgba(103,184,95,0.18);
          border-color: rgba(71,124,66,0.52);
        }

        .feature-tile.negative {
          background: rgba(217,74,58,0.16);
          border-color: rgba(151,44,34,0.48);
        }

        .feature-name {
          font-size: 0.54rem;
          letter-spacing: 0.08em;
          color: var(--battle-red);
          margin-bottom: 0.35rem;
        }

        .feature-value {
          font-size: 1.65rem;
          color: var(--shell-navy);
          line-height: 1;
        }

        .feature-lean {
          margin-top: 0.25rem;
          font-size: 1.15rem;
          color: #243550;
        }

        @keyframes pulseStart {
          0%, 100% { transform: translate(0, 0); }
          50% { transform: translate(2px, 2px); box-shadow: 2px 2px 0 rgba(13,19,33,0.4); }
        }

        .sidebar-shell {
          padding: 1rem;
          margin-bottom: 1rem;
          background: rgba(255,247,227,0.08);
          border-color: rgba(255,247,227,0.4);
          box-shadow: 5px 5px 0 rgba(0,0,0,0.38);
        }

        .sidebar-shell::after {
          border-color: rgba(255,247,227,0.14);
        }

        .console-title {
          margin: 0.25rem 0 0.8rem;
          font-size: 0.75rem;
          line-height: 1.5;
        }

        .console-copy {
          font-size: 1.25rem;
          line-height: 1.12;
          margin-bottom: 0.75rem;
          color: rgba(255,247,227,0.92);
        }

        .side-table {
          display: flex;
          flex-direction: column;
          gap: 0.2rem;
          margin-top: 0.55rem;
        }

        .side-table-row {
          display: grid;
          grid-template-columns: 1.7fr 0.9fr 0.9fr 0.9fr;
          gap: 0.35rem;
        }

        .side-table-header .side-table-head {
          font-size: 0.5rem;
          color: rgba(247,214,74,0.95);
          letter-spacing: 0.08em;
        }

        .side-table-cell {
          padding: 0.25rem 0.3rem;
          background: rgba(255,247,227,0.08);
          border: 2px solid rgba(255,247,227,0.12);
          font-size: 1.05rem;
          line-height: 1.05;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .note-card {
          background: rgba(16,32,60,0.12);
          border: 3px solid rgba(255,247,227,0.15);
          padding: 0.55rem 0.65rem;
          font-size: 1.15rem;
          line-height: 1.1;
          margin-top: 0.65rem;
        }

        @media (max-width: 1100px) {
          .app-hero,
          .start-grid,
          .page-hero,
          .hud-grid,
          .battle-hero {
            grid-template-columns: 1fr;
          }

          .hero-screen {
            flex-direction: column;
            align-items: flex-start;
          }
        }

        @media (max-width: 780px) {
          .hero-sprite,
          .fighter-sprite {
            width: 100%;
            min-width: 100%;
            height: auto;
            min-height: 150px;
          }

          .meter-row,
          .oof-meter-row,
          .battle-meter-row {
            grid-template-columns: 1fr;
          }

          .side-table-row {
            grid-template-columns: 1.4fr 1fr 1fr 1fr;
          }

          div[data-testid="stRadio"] > div {
            flex-direction: column;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(bundle: dict) -> None:
    tables = notebook_ready_tables(bundle)
    _mount_html(
        f"""
        <div class="sidebar-shell">
          <div class="panel-kicker">TRAINER CONSOLE</div>
          <div class="console-title">POKEMON ML ANALYSIS</div>
          <div class="console-copy">
            Retro device styling wrapped around the same local prediction bundle.
          </div>
          <div class="note-card">Bundle source :: {escape(str(bundle.get('bundle_source', 'unknown')).upper())}</div>
          <div class="note-card">Type model :: {escape(bundle['type_bundle']['final_model_name'])}</div>
          <div class="note-card">Battle model :: {escape(bundle['battle_bundle']['final_model_name'])}</div>
        </div>
        """,
        container=st.sidebar,
    )

    _mount_html(
        f"""
        <div class="sidebar-shell">
          <div class="panel-kicker">TYPE LEADERBOARD</div>
          {_render_sidebar_table(
              tables["type_random"][["model", "micro_f1", "macro_f1", "exact_match"]].round(3),
              ["model", "micro_f1", "macro_f1", "exact_match"],
              {"micro_f1": ".3f", "macro_f1": ".3f", "exact_match": ".3f"},
          )}
        </div>
        """,
        container=st.sidebar,
    )

    _mount_html(
        f"""
        <div class="sidebar-shell">
          <div class="panel-kicker">BATTLE LEADERBOARD</div>
          {_render_sidebar_table(
              tables["battle_grouped"][["model", "accuracy", "roc_auc"]].round(3),
              ["model", "accuracy", "roc_auc"],
              {"accuracy": ".3f", "roc_auc": ".3f"},
          )}
        </div>
        """,
        container=st.sidebar,
    )

    _mount_html(_render_type_oof_summary_card(bundle), container=st.sidebar)


def _render_type_overview_card(row: pd.Series, result: dict) -> str:
    ability_text = row["ability1"]
    if row.get("ability2") and row["ability2"] != "Unknown":
        ability_text += f" / {row['ability2']}"
    return f"""
    <div class="pixel-panel">
      <div class="panel-kicker">GROUND TRUTH</div>
      <div class="panel-title">TRUE TYPE FROM DATASET</div>
      <div>{_render_type_badges([result['true_primary'], result['true_secondary']])}</div>
      {_render_stat_strip([
          ("DEX", f"#{int(row['dexnum_int']):03d}"),
          ("SPECIES", str(row['species']).upper()),
          ("GEN", f"{int(row['generation'])}"),
          ("BST", f"{int(row['total'])}"),
      ])}
      <div class="note-card">ABILITY :: {escape(ability_text.upper())}</div>
      <div class="note-card">SCAN CLASS :: {"DUAL-TYPE" if result['true_secondary'] != 'None' else "SINGLE-TYPE"}</div>
    </div>
    """


def _render_type_prediction_card(result: dict) -> str:
    top_probability = float(result["probabilities"].iloc[0]["probability"])
    confidence_state = "LOCKED IN" if top_probability >= 0.8 else "CHECK SIGNAL"
    return f"""
    <div class="pixel-panel">
      <div class="panel-kicker">MODEL PREDICTION</div>
      <div class="panel-title">PREDICTED TYPE FROM ML MODEL</div>
      <div>{_render_type_badges([result['predicted_primary'], result['predicted_secondary']])}</div>
      {_render_stat_strip([
          ("MODEL", result["model_name"].replace("ClassifierChain ", "").upper()),
          ("TOP CONF", f"{top_probability:.1%}"),
          ("STATE", confidence_state),
      ])}
      <div class="note-card">PREDICTED SECONDARY :: {escape(result['predicted_secondary'].upper())}</div>
    </div>
    """


def render_type_page(bundle: dict) -> None:
    master_df = bundle["master_df"]
    options = _pokemon_options(master_df)
    selection = st.selectbox("Choose a Pokemon to scan", options, index=5)
    result = predict_types(_name_from_option(selection), bundle).payload
    row = master_df[master_df["name"] == result["name"]].iloc[0]

    hero_html = f"""
    <div class="app-shell page-hero">
      <div class="hero-screen">
        <div class="hero-sprite">
          <img src="{escape(result['image_url'])}" alt="{escape(result['name'])} sprite" />
        </div>
        <div class="hero-data">
          <div class="hero-status">POKEDEX SCAN MODE :: ACTIVE</div>
          <div class="panel-kicker">GROUND TRUTH VS MODEL PREDICTION</div>
          <h2>DEX #{int(result['dexnum_int']):03d} :: {escape(result['name'].upper())}</h2>
          <p>Reading stat signature, species tag, growth profile, and ability fingerprint.</p>
          {_render_stat_strip([
              ("TRUE PRIMARY", result["true_primary"].upper()),
              ("TRUE SECONDARY", result["true_secondary"].upper()),
              ("MODEL", result["model_name"].replace("ClassifierChain ", "").upper()),
          ])}
        </div>
      </div>
      <div class="pixel-panel">
        <div class="panel-kicker">TYPE SIGNAL HUD</div>
        <div class="panel-title">RANKED TYPE CONFIDENCE</div>
        {_render_probability_meter(result["probabilities"])}
      </div>
    </div>
    """
    _mount_html(hero_html)

    left_col, right_col = st.columns([1, 1])
    with left_col:
        _mount_html(_render_type_overview_card(row, result))
    with right_col:
        _mount_html(_render_type_prediction_card(result))

    _mount_html(_render_commentary_lines(result["explanation"], "PROFESSOR NOTES", tone="type"))


def _render_battle_header(result: dict) -> str:
    pokemon_a = result["pokemon_a"]
    pokemon_b = result["pokemon_b"]
    return f"""
    <div class="app-shell battle-hero">
      <div class="fighter-card">
        <div class="panel-kicker">COMBATANT A</div>
        <div class="fighter-sprite"><img src="{escape(pokemon_a['image_url'])}" alt="{escape(pokemon_a['name'])} sprite" /></div>
        <h3>{escape(pokemon_a['name'].upper())}</h3>
        <div>{_render_type_badges([pokemon_a['type1'], pokemon_a['type2']])}</div>
        {_render_stat_strip([
            ("DEX", f"#{pokemon_a['dexnum_int']:03d}"),
            ("BST", str(pokemon_a["total"])),
        ])}
      </div>
      <div class="vs-core">
        <div class="vs-badge">VS</div>
        <div class="winner-banner">
          <div class="winner-label">PREDICTED WINNER</div>
          <div class="winner-name">{escape(result['predicted_winner'].upper())}</div>
        </div>
      </div>
      <div class="fighter-card">
        <div class="panel-kicker">COMBATANT B</div>
        <div class="fighter-sprite"><img src="{escape(pokemon_b['image_url'])}" alt="{escape(pokemon_b['name'])} sprite" /></div>
        <h3>{escape(pokemon_b['name'].upper())}</h3>
        <div>{_render_type_badges([pokemon_b['type1'], pokemon_b['type2']])}</div>
        {_render_stat_strip([
            ("DEX", f"#{pokemon_b['dexnum_int']:03d}"),
            ("BST", str(pokemon_b["total"])),
        ])}
      </div>
    </div>
    """


def _render_battle_probability_panel(result: dict) -> str:
    pokemon_a = result["pokemon_a"]["name"]
    pokemon_b = result["pokemon_b"]["name"]
    history = result["history"]
    history_lines = (
        [
            f"Recorded battles: {history['total_battles']}",
            f"{pokemon_a} wins: {history['wins_a']} ({history['rate_a']:.1%})" if history["rate_a"] is not None else f"{pokemon_a} wins: 0",
            f"{pokemon_b} wins: {history['wins_b']} ({history['rate_b']:.1%})" if history["rate_b"] is not None else f"{pokemon_b} wins: 0",
        ]
        if history["total_battles"]
        else ["Recorded battles: 0", "No historical battle record in the local dataset.", "Use the model HUD as the primary signal."]
    )

    history_log = _render_commentary_lines(history_lines, "BATTLE RECORD", tone="battle")
    return f"""
    <div class="hud-grid">
      <div class="pixel-panel">
        <div class="panel-kicker">WIN PROBABILITY HUD</div>
        <div class="panel-title">ARENA FORECAST</div>
        <div class="battle-meter-shell">
          <div class="battle-meter-row">
            <div class="battle-meter-label">{escape(pokemon_a.upper())}</div>
            <div class="meter-track"><div class="meter-fill" style="width:{result['win_prob_a'] * 100:.1f}%;background:#67B85F;"></div></div>
            <div class="meter-row-value" style="background:#67B85F;color:#FFF7E3;">{result['win_prob_a']:.1%}</div>
          </div>
          <div class="battle-meter-row">
            <div class="battle-meter-label">{escape(pokemon_b.upper())}</div>
            <div class="meter-track"><div class="meter-fill" style="width:{result['win_prob_b'] * 100:.1f}%;background:#D94A3A;"></div></div>
            <div class="meter-row-value" style="background:#D94A3A;color:#FFF7E3;">{result['win_prob_b']:.1%}</div>
          </div>
        </div>
        {_render_stat_strip([
            ("MODEL", result["model_name"].upper()),
            ("LEAD", result["predicted_winner"].upper()),
            ("HIST", str(history["total_battles"])),
        ])}
      </div>
      {history_log}
    </div>
    """


def _render_feature_snapshot(df: pd.DataFrame) -> str:
    tiles = []
    for row in df.itertuples(index=False):
        feature_name = str(row.feature).replace("_", " ").upper()
        value = float(row.value)
        if value > 0:
            tile_class = "positive"
            leaning = "A EDGE"
        elif value < 0:
            tile_class = "negative"
            leaning = "B EDGE"
        else:
            tile_class = ""
            leaning = "EVEN"
        tiles.append(
            f"""
            <div class="feature-tile {tile_class}">
              <div class="feature-name">{escape(feature_name)}</div>
              <div class="feature-value">{value:+.2f}</div>
              <div class="feature-lean">{leaning}</div>
            </div>
            """
        )
    return f"""
    <div class="pixel-panel">
      <div class="panel-kicker">MATCHUP SNAPSHOT</div>
      <div class="panel-title">KEY BATTLE SIGNALS</div>
      <div class="feature-grid">{''.join(tiles)}</div>
    </div>
    """


def render_battle_page(bundle: dict) -> None:
    master_df = bundle["master_df"]
    options = _pokemon_options(master_df)

    col_a, col_b = st.columns(2)
    with col_a:
        option_a = st.selectbox("Choose Pokemon A", options, index=5)
    with col_b:
        option_b = st.selectbox("Choose Pokemon B", options, index=8)

    result = predict_battle(_name_from_option(option_a), _name_from_option(option_b), bundle).payload
    _mount_html(_render_battle_header(result))
    _mount_html(_render_battle_probability_panel(result))
    _mount_html(_render_feature_snapshot(result["feature_snapshot"]))
    _mount_html(_render_commentary_lines(result["explanation"], "BATTLE COMMENTARY", tone="battle"))


def main() -> None:
    st.set_page_config(page_title="Pokemon ML Analysis", layout="wide")
    inject_global_styles()
    bundle = load_bundle()
    render_sidebar(bundle)

    options = ["Pokedex Scan", "Battle HUD"]
    current_page = st.session_state.get("mode_selector", options[0])
    if current_page not in options:
        current_page = options[0]

    _render_app = "Pokedex Scan" if current_page == "Pokedex Scan" else "Battle HUD"
    _render_app_shell(bundle, _render_app)
    _render_mode_menu(_render_app)

    page = st.radio(
        "Select a mode",
        options,
        horizontal=True,
        label_visibility="collapsed",
        key="mode_selector",
        index=options.index(current_page),
    )
    _render_app = "Pokedex Scan" if page == "Pokedex Scan" else "Battle HUD"

    if page == "Pokedex Scan":
        render_type_page(bundle)
    else:
        render_battle_page(bundle)


if __name__ == "__main__":
    main()
