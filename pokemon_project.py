from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
POKEAPI_CACHE = ROOT / "pokeapi_enrichment_cache.json"
SMOGON_CACHE = ROOT / "smogon_usage_cache.json"
ARTIFACTS_DIR = ROOT / "artifacts"
DEPLOY_BUNDLE_PATH = ARTIFACTS_DIR / "streamlit_cloud_bundle.joblib"

TYPE_ORDER = [
    "Normal",
    "Fire",
    "Water",
    "Electric",
    "Grass",
    "Ice",
    "Fighting",
    "Poison",
    "Ground",
    "Flying",
    "Psychic",
    "Bug",
    "Rock",
    "Ghost",
    "Dragon",
    "Dark",
    "Steel",
    "Fairy",
]

CANONICAL_TYPE_CHART: dict[str, dict[str, float]] = {
    "Normal": {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
    "Fire": {
        "Fire": 0.5,
        "Water": 0.5,
        "Grass": 2.0,
        "Ice": 2.0,
        "Bug": 2.0,
        "Rock": 0.5,
        "Dragon": 0.5,
        "Steel": 2.0,
    },
    "Water": {
        "Fire": 2.0,
        "Water": 0.5,
        "Grass": 0.5,
        "Ground": 2.0,
        "Rock": 2.0,
        "Dragon": 0.5,
    },
    "Electric": {
        "Water": 2.0,
        "Electric": 0.5,
        "Grass": 0.5,
        "Ground": 0.0,
        "Flying": 2.0,
        "Dragon": 0.5,
    },
    "Grass": {
        "Fire": 0.5,
        "Water": 2.0,
        "Grass": 0.5,
        "Poison": 0.5,
        "Ground": 2.0,
        "Flying": 0.5,
        "Bug": 0.5,
        "Rock": 2.0,
        "Dragon": 0.5,
        "Steel": 0.5,
    },
    "Ice": {
        "Fire": 0.5,
        "Water": 0.5,
        "Grass": 2.0,
        "Ground": 2.0,
        "Flying": 2.0,
        "Dragon": 2.0,
        "Steel": 0.5,
        "Ice": 0.5,
    },
    "Fighting": {
        "Normal": 2.0,
        "Ice": 2.0,
        "Rock": 2.0,
        "Dark": 2.0,
        "Steel": 2.0,
        "Poison": 0.5,
        "Flying": 0.5,
        "Psychic": 0.5,
        "Bug": 0.5,
        "Ghost": 0.0,
        "Fairy": 0.5,
    },
    "Poison": {
        "Grass": 2.0,
        "Fairy": 2.0,
        "Poison": 0.5,
        "Ground": 0.5,
        "Rock": 0.5,
        "Ghost": 0.5,
        "Steel": 0.0,
    },
    "Ground": {
        "Fire": 2.0,
        "Electric": 2.0,
        "Grass": 0.5,
        "Poison": 2.0,
        "Flying": 0.0,
        "Bug": 0.5,
        "Rock": 2.0,
        "Steel": 2.0,
    },
    "Flying": {
        "Grass": 2.0,
        "Electric": 0.5,
        "Fighting": 2.0,
        "Bug": 2.0,
        "Rock": 0.5,
        "Steel": 0.5,
    },
    "Psychic": {
        "Fighting": 2.0,
        "Poison": 2.0,
        "Psychic": 0.5,
        "Dark": 0.0,
        "Steel": 0.5,
    },
    "Bug": {
        "Grass": 2.0,
        "Fire": 0.5,
        "Fighting": 0.5,
        "Poison": 0.5,
        "Flying": 0.5,
        "Psychic": 2.0,
        "Ghost": 0.5,
        "Dark": 2.0,
        "Steel": 0.5,
        "Fairy": 0.5,
    },
    "Rock": {
        "Fire": 2.0,
        "Ice": 2.0,
        "Fighting": 0.5,
        "Ground": 0.5,
        "Flying": 2.0,
        "Bug": 2.0,
        "Steel": 0.5,
    },
    "Ghost": {"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5},
    "Dragon": {"Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0},
    "Dark": {
        "Fighting": 0.5,
        "Psychic": 2.0,
        "Ghost": 2.0,
        "Dark": 0.5,
        "Fairy": 0.5,
    },
    "Steel": {
        "Fire": 0.5,
        "Water": 0.5,
        "Electric": 0.5,
        "Ice": 2.0,
        "Rock": 2.0,
        "Fairy": 2.0,
        "Steel": 0.5,
    },
    "Fairy": {
        "Fire": 0.5,
        "Fighting": 2.0,
        "Poison": 0.5,
        "Dragon": 2.0,
        "Dark": 2.0,
        "Steel": 0.5,
    },
}

LOCAL_NUMERIC_COLUMNS = [
    "generation",
    "height",
    "weight",
    "hp",
    "attack",
    "defense",
    "sp_atk",
    "sp_def",
    "speed",
    "total",
    "catch_rate",
    "percent_male",
    "percent_female",
    "egg_cycles",
    "base_friendship",
    "base_exp",
]

TYPE_BASE_CATEGORICAL_COLUMNS = ["growth_rate", "egg_group1", "egg_group2", "special_group"]
TYPE_RICH_CATEGORICAL_COLUMNS = TYPE_BASE_CATEGORICAL_COLUMNS + [
    "species",
    "ability1",
    "ability2",
    "hidden_ability",
    "ev_yield",
]

OPTIONAL_EXTERNAL_NUMERIC_COLUMNS = [
    "evolution_chain_id",
    "move_count",
    "pokemon_api_ability_count",
    "is_legendary",
    "is_mythical",
]

OPTIONAL_EXTERNAL_CATEGORICAL_COLUMNS = [
    "pokeapi_color",
    "pokeapi_shape",
    "pokeapi_habitat",
]

TYPE_MODEL_CANDIDATES = [
    ("OVR Logistic", "baseline", "ovr_logistic"),
    ("ClassifierChain Logistic (C=10)", "rich", "chain_logistic"),
    ("ExtraTrees MultiOutput", "rich", "extra_trees"),
]
TYPE_MODEL_LOOKUP = {
    display_name: {"feature_key": feature_key, "model_key": model_key}
    for display_name, feature_key, model_key in TYPE_MODEL_CANDIDATES
}


@dataclass
class PredictionResult:
    payload: dict[str, Any]


def _safe_json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text in {"—", "nan", "NaN", "None"}:
        return None
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_local_data(root: Path | None = None) -> dict[str, pd.DataFrame]:
    base = root or ROOT
    return {
        "pokemon": pd.read_csv(base / "IS510_Pokemon_List.csv"),
        "single_combats": pd.read_csv(base / "IS510_Pokemon_Single_Combats.csv"),
        "team_combats": pd.read_csv(base / "IS510_Pokemon_Team_Combat.csv"),
        "team_ids": pd.read_csv(base / "IS510_Pokemon_ID_Each_Team.csv"),
        "type_matchup": pd.read_csv(base / "IS510_Pokemon_Type_Matchup_Data.csv"),
        "legacy_cn": pd.read_csv(base / "pokemon_data.csv"),
    }


def summarize_local_data(local_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in local_data.items():
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": len(df.columns),
                "missing_cells": int(df.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def audit_matchup_coverage(local_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    single_combats = local_data["single_combats"].copy()
    matchup = local_data["type_matchup"].copy()
    matchup_ids = (
        matchup["Number"].astype(str).str.extract(r"(\d+)")[0].dropna().astype(int).tolist()
    )
    matchup_id_set = set(matchup_ids)
    both_covered = (
        single_combats["First_pokemon"].isin(matchup_id_set)
        & single_combats["Second_pokemon"].isin(matchup_id_set)
    )
    unique_combat_ids = pd.unique(single_combats[["First_pokemon", "Second_pokemon", "Winner"]].values.ravel())
    rows = [
        {
            "original_single_combats_rows": int(len(single_combats)),
            "rows_after_old_incomplete_matchup_join": int(both_covered.sum()),
            "rows_lost_by_old_join": int(len(single_combats) - both_covered.sum()),
            "coverage_ratio": float(both_covered.mean()),
            "unique_pokemon_ids_in_combats": int(len(unique_combat_ids)),
            "unique_ids_covered_by_matchup_csv": int(sum(int(p) in matchup_id_set for p in unique_combat_ids)),
        }
    ]
    return pd.DataFrame(rows)


def clean_pokemon_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df[df["name"].notna() & df["type1"].notna()].copy()
    df["type1"] = df["type1"].astype(str)
    df["type2"] = df["type2"].fillna("None").astype(str)
    for column in LOCAL_NUMERIC_COLUMNS + ["dexnum"]:
        df[column] = df[column].apply(_parse_number)
    df["dexnum_int"] = df["dexnum"].astype(int)
    for column in ["ability1", "ability2", "hidden_ability", "species", "ev_yield", "url"]:
        df[column] = df[column].fillna("Unknown").astype(str)
    for column in ["growth_rate", "egg_group1", "egg_group2", "special_group"]:
        df[column] = df[column].fillna("Unknown").astype(str)
    df["percent_male"] = df["percent_male"].fillna(0.0)
    df["percent_female"] = df["percent_female"].fillna(0.0)
    df["image_url"] = df["url"].replace("Unknown", "").fillna("")
    df["bulk_score"] = df["hp"] + df["defense"] + df["sp_def"]
    df["offense_score"] = df["attack"] + df["sp_atk"] + df["speed"]
    df["physical_bias"] = df["attack"] - df["sp_atk"]
    df["special_bias"] = df["sp_atk"] - df["attack"]
    df["speed_rank_pct"] = df["speed"].rank(pct=True)
    df["single_type_flag"] = (df["type2"] == "None").astype(int)
    return df.reset_index(drop=True)


def fetch_pokeapi_enrichment(
    dex_numbers: list[int] | None = None,
    cache_path: Path | None = None,
    sleep_seconds: float = 0.05,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_file = cache_path or POKEAPI_CACHE
    cache = _safe_json_load(cache_file)
    session = requests.Session()
    dex_list = dex_numbers or []
    rows: list[dict[str, Any]] = []

    for dex in dex_list:
        key = str(int(dex))
        if force_refresh or key not in cache:
            species_url = f"https://pokeapi.co/api/v2/pokemon-species/{dex}/"
            pokemon_url = f"https://pokeapi.co/api/v2/pokemon/{dex}/"
            species_resp = session.get(species_url, timeout=30)
            pokemon_resp = session.get(pokemon_url, timeout=30)
            species_resp.raise_for_status()
            pokemon_resp.raise_for_status()
            species_payload = species_resp.json()
            pokemon_payload = pokemon_resp.json()
            flavor_text = ""
            for entry in species_payload.get("flavor_text_entries", []):
                if entry.get("language", {}).get("name") == "en":
                    flavor_text = entry.get("flavor_text", "").replace("\n", " ").replace("\f", " ").strip()
                    break
            evolution_chain_url = species_payload.get("evolution_chain", {}).get("url", "")
            evolution_chain_id = _parse_number(evolution_chain_url.rstrip("/").split("/")[-1])
            cache[key] = {
                "dexnum_int": int(dex),
                "is_legendary": int(species_payload.get("is_legendary", False)),
                "is_mythical": int(species_payload.get("is_mythical", False)),
                "pokeapi_color": species_payload.get("color", {}).get("name", "unknown"),
                "pokeapi_shape": species_payload.get("shape", {}).get("name", "unknown"),
                "pokeapi_habitat": species_payload.get("habitat", {}).get("name", "unknown"),
                "evolution_chain_id": evolution_chain_id or 0,
                "pokemon_api_ability_count": len(pokemon_payload.get("abilities", [])),
                "move_count": len(pokemon_payload.get("moves", [])),
                "flavor_text_en": flavor_text,
            }
            time.sleep(sleep_seconds)
        rows.append(cache[key])

    if dex_list:
        _safe_json_dump(cache_file, cache)

    return pd.DataFrame(rows)


def _default_smogon_month() -> str:
    now = pd.Timestamp.now(tz="America/Los_Angeles")
    previous_month = now - pd.offsets.MonthBegin(1)
    return previous_month.strftime("%Y-%m")


def fetch_smogon_usage_stats(
    month: str | None = None,
    format_name: str = "gen9ou-1500",
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_file = cache_path or SMOGON_CACHE
    cache = _safe_json_load(cache_file)
    target_month = month or _default_smogon_month()
    cache_key = f"{target_month}:{format_name}"

    if force_refresh or cache_key not in cache:
        url = f"https://www.smogon.com/stats/{target_month}/{format_name}.txt"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        usage_rows = []
        for line in response.text.splitlines():
            match = re.match(r"\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([0-9.]+)%\s*\|", line)
            if not match:
                continue
            usage_rows.append(
                {
                    "rank": int(match.group(1)),
                    "pokemon_name": match.group(2).strip(),
                    "usage_pct": float(match.group(3)),
                    "format_name": format_name,
                    "month": target_month,
                }
            )
        cache[cache_key] = usage_rows
        _safe_json_dump(cache_file, cache)

    return pd.DataFrame(cache.get(cache_key, []))


def build_master_table(
    root: Path | None = None,
    include_external: bool = False,
    pokeapi_cache_path: Path | None = None,
    smogon_cache_path: Path | None = None,
) -> pd.DataFrame:
    local_data = load_local_data(root)
    master_df = clean_pokemon_table(local_data["pokemon"])

    if include_external and (pokeapi_cache_path or POKEAPI_CACHE.exists()):
        cache_source = pokeapi_cache_path or POKEAPI_CACHE
        cache = _safe_json_load(cache_source)
        if cache:
            pokeapi_df = pd.DataFrame(cache.values())
            if not pokeapi_df.empty:
                master_df = master_df.merge(pokeapi_df, how="left", on="dexnum_int")

    if include_external and (smogon_cache_path or SMOGON_CACHE.exists()):
        cache_source = smogon_cache_path or SMOGON_CACHE
        usage_cache = _safe_json_load(cache_source)
        usage_rows: list[dict[str, Any]] = []
        for rows in usage_cache.values():
            usage_rows.extend(rows)
        usage_df = pd.DataFrame(usage_rows)
        if not usage_df.empty:
            usage_df = usage_df.sort_values(["month", "usage_pct"], ascending=[False, False]).drop_duplicates(
                "pokemon_name"
            )
            usage_df["pokemon_name"] = usage_df["pokemon_name"].astype(str)
            master_df = master_df.merge(
                usage_df[["pokemon_name", "usage_pct"]].rename(columns={"pokemon_name": "name"}),
                how="left",
                on="name",
            )

    for column in OPTIONAL_EXTERNAL_NUMERIC_COLUMNS:
        if column in master_df.columns:
            master_df[column] = master_df[column].fillna(0.0)
    for column in OPTIONAL_EXTERNAL_CATEGORICAL_COLUMNS:
        if column in master_df.columns:
            master_df[column] = master_df[column].fillna("unknown").astype(str)

    return master_df


def _available_optional_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns and df[column].notna().any()]


def _type_feature_sets(master_df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    rich_numeric = LOCAL_NUMERIC_COLUMNS + _available_optional_columns(master_df, OPTIONAL_EXTERNAL_NUMERIC_COLUMNS)
    rich_categorical = TYPE_RICH_CATEGORICAL_COLUMNS + _available_optional_columns(
        master_df, OPTIONAL_EXTERNAL_CATEGORICAL_COLUMNS
    )
    return {
        "baseline": {
            "numeric": LOCAL_NUMERIC_COLUMNS,
            "categorical": TYPE_BASE_CATEGORICAL_COLUMNS,
        },
        "rich": {
            "numeric": rich_numeric,
            "categorical": rich_categorical,
        },
    }


def _build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def _fit_type_estimator(
    model_key: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Any:
    if model_key == "ovr_logistic":
        estimator = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", OneVsRestClassifier(LogisticRegression(max_iter=500))),
            ]
        )
        estimator.fit(X_train, y_train)
        return estimator

    if model_key == "chain_logistic":
        transformed_train = preprocessor.fit_transform(X_train)
        estimator = ClassifierChain(
            LogisticRegression(max_iter=800, C=10.0),
            order="random",
            random_state=1,
        )
        estimator.fit(transformed_train, y_train)
        return estimator

    estimator = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                MultiOutputClassifier(
                    ExtraTreesClassifier(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )
    estimator.fit(X_train, y_train)
    return estimator


def _predict_type_labels(
    estimator: Any,
    preprocessor: ColumnTransformer,
    model_key: str,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    if model_key == "chain_logistic":
        transformed_eval = preprocessor.transform(X_eval)
        return estimator.predict(transformed_eval)
    return estimator.predict(X_eval)


def _predict_type_probabilities(
    estimator: Any,
    preprocessor: ColumnTransformer,
    model_key: str,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    if model_key == "chain_logistic":
        transformed_eval = preprocessor.transform(X_eval)
        raw_output = estimator.predict_proba(transformed_eval)
    else:
        raw_output = estimator.predict_proba(X_eval)
    return _normalize_multilabel_proba(raw_output)


def _type_split_indices(master_df: pd.DataFrame, split_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if split_mode == "random":
        train_idx, test_idx = train_test_split(master_df.index.to_numpy(), test_size=0.2, random_state=42)
        return np.array(train_idx), np.array(test_idx)
    groups = master_df["species"].fillna(master_df["dexnum_int"].astype(str))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(master_df, groups=groups))
    return train_idx, test_idx


def _normalize_multilabel_proba(raw_output: Any) -> np.ndarray:
    if isinstance(raw_output, list):
        columns = []
        for entry in raw_output:
            array = np.asarray(entry)
            if array.ndim == 2 and array.shape[1] == 2:
                columns.append(array[:, 1])
            else:
                columns.append(array.ravel())
        return np.column_stack(columns)
    array = np.asarray(raw_output)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[:, :, 1]
    return array


def evaluate_type_models(master_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_sets = _type_feature_sets(master_df)
    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(master_df[["type1", "type2"]].values.tolist())
    reports: list[dict[str, Any]] = []
    fitted_models: dict[str, dict[str, Any]] = {}

    for split_mode in ["random", "grouped"]:
        train_idx, test_idx = _type_split_indices(master_df, split_mode)
        X_train = master_df.iloc[train_idx]
        X_test = master_df.iloc[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        for display_name, feature_key, model_key in TYPE_MODEL_CANDIDATES:
            columns = feature_sets[feature_key]
            preprocessor = _build_preprocessor(columns["numeric"], columns["categorical"])
            estimator = _fit_type_estimator(model_key, preprocessor, X_train, y_train)
            y_pred = _predict_type_labels(estimator, preprocessor, model_key, X_test)

            report = {
                "task": "type_prediction",
                "model": display_name,
                "split": split_mode,
                "micro_f1": float(f1_score(y_test, y_pred, average="micro")),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "hamming_loss": float(hamming_loss(y_test, y_pred)),
                "exact_match": float(accuracy_score(y_test, y_pred)),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
            reports.append(report)
            fitted_models.setdefault(display_name, {})[split_mode] = {
                "estimator": estimator,
                "preprocessor": preprocessor,
                "feature_key": feature_key,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }

    reports_df = pd.DataFrame(reports).sort_values(["split", "exact_match", "micro_f1"], ascending=[True, False, False])
    random_reports = reports_df[reports_df["split"] == "random"].copy()
    random_reports = random_reports.sort_values(["exact_match", "micro_f1"], ascending=[False, False]).reset_index(drop=True)
    best_row = random_reports.iloc[0]
    if len(random_reports) > 1:
        runner_up = random_reports.iloc[1]
        if (
            best_row["model"] != "ClassifierChain Logistic (C=10)"
            and abs(best_row["exact_match"] - runner_up["exact_match"]) < 0.01
            and "ClassifierChain Logistic (C=10)" in set(random_reports["model"])
        ):
            best_row = random_reports[random_reports["model"] == "ClassifierChain Logistic (C=10)"].iloc[0]

    final_model = fitted_models[best_row["model"]]["random"]
    return reports_df.reset_index(drop=True), {
        "mlb": mlb,
        "feature_sets": feature_sets,
        "final_model_name": best_row["model"],
        "final_estimator": final_model["estimator"],
        "final_preprocessor": final_model["preprocessor"],
        "final_feature_key": final_model["feature_key"],
        "final_threshold": 0.30,
    }


def type_oof_statistics(
    master_df: pd.DataFrame,
    type_bundle: dict[str, Any],
    split_mode: str = "grouped",
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_name = type_bundle["final_model_name"]
    model_spec = TYPE_MODEL_LOOKUP[model_name]
    feature_key = type_bundle.get("final_feature_key", model_spec["feature_key"])
    model_key = model_spec["model_key"]
    feature_sets = type_bundle.get("feature_sets", _type_feature_sets(master_df))
    mlb = type_bundle["mlb"]
    threshold = float(type_bundle.get("final_threshold", 0.30))
    classes = mlb.classes_
    y_all = mlb.transform(master_df[["type1", "type2"]].values.tolist())
    columns = feature_sets[feature_key]

    if split_mode == "grouped":
        groups = master_df["species"].fillna(master_df["dexnum_int"].astype(str))
        fold_count = max(2, min(n_splits, int(groups.nunique())))
        splitter = GroupKFold(n_splits=fold_count)
        fold_iterator = splitter.split(master_df, y_all, groups=groups)
    else:
        fold_count = max(2, min(n_splits, len(master_df)))
        splitter = KFold(n_splits=fold_count, shuffle=True, random_state=42)
        fold_iterator = splitter.split(master_df, y_all)

    detail_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(fold_iterator, start=1):
        X_train = master_df.iloc[train_idx]
        X_test = master_df.iloc[test_idx]
        y_train = y_all[train_idx]
        preprocessor = _build_preprocessor(columns["numeric"], columns["categorical"])
        estimator = _fit_type_estimator(model_key, preprocessor, X_train, y_train)
        fold_probabilities = _predict_type_probabilities(estimator, preprocessor, model_key, X_test)

        for row, probabilities in zip(X_test.itertuples(index=False), fold_probabilities):
            pred_primary, pred_secondary = _decode_type_prediction(probabilities, classes, threshold)
            true_set = {label for label in [row.type1, row.type2] if label != "None"}
            predicted_set = {label for label in [pred_primary, pred_secondary] if label != "None"}
            overlap_n = len(true_set & predicted_set)
            all_types_correct = true_set == predicted_set
            ordered_match = pred_primary == row.type1 and pred_secondary == row.type2
            if all_types_correct:
                bucket = "all_types_correct"
            elif overlap_n == 1:
                bucket = "one_type_correct"
            else:
                bucket = "zero_type_correct"

            detail_rows.append(
                {
                    "split_mode": split_mode,
                    "fold": fold_id,
                    "dexnum_int": int(row.dexnum_int),
                    "name": row.name,
                    "true_primary": row.type1,
                    "true_secondary": row.type2,
                    "predicted_primary": pred_primary,
                    "predicted_secondary": pred_secondary,
                    "all_types_correct": bool(all_types_correct),
                    "ordered_match": bool(ordered_match),
                    "matched_type_count": int(overlap_n),
                    "result_bucket": bucket,
                    "single_type_truth": bool(row.type2 == "None"),
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["fold", "dexnum_int"]).reset_index(drop=True)
    n_total = len(detail_df)
    all_types_correct_n = int((detail_df["result_bucket"] == "all_types_correct").sum())
    one_type_correct_n = int((detail_df["result_bucket"] == "one_type_correct").sum())
    zero_type_correct_n = int((detail_df["result_bucket"] == "zero_type_correct").sum())

    summary_df = pd.DataFrame(
        [
            {
                "split_mode": split_mode,
                "model": model_name,
                "feature_key": feature_key,
                "n_splits": int(fold_count),
                "n_total": int(n_total),
                "all_types_correct_n": all_types_correct_n,
                "all_types_correct_pct": float(all_types_correct_n / n_total) if n_total else 0.0,
                "one_type_correct_n": one_type_correct_n,
                "one_type_correct_pct": float(one_type_correct_n / n_total) if n_total else 0.0,
                "zero_type_correct_n": zero_type_correct_n,
                "zero_type_correct_pct": float(zero_type_correct_n / n_total) if n_total else 0.0,
                "ordered_match_n": int(detail_df["ordered_match"].sum()),
                "ordered_match_pct": float(detail_df["ordered_match"].mean()) if n_total else 0.0,
            }
        ]
    )
    return summary_df, detail_df


def _type_multiplier(move_type: str, defender_types: list[str]) -> float:
    multiplier = 1.0
    for defender_type in defender_types:
        if defender_type and defender_type != "None":
            multiplier *= CANONICAL_TYPE_CHART.get(move_type, {}).get(defender_type, 1.0)
    return multiplier


def _best_stab_multiplier(attacker_type1: str, attacker_type2: str, defender_type1: str, defender_type2: str) -> float:
    attacker_types = [t for t in [attacker_type1, attacker_type2] if isinstance(t, str) and t != "None"]
    defender_types = [t for t in [defender_type1, defender_type2] if isinstance(t, str) and t != "None"]
    if not attacker_types:
        return 1.0
    return max(_type_multiplier(move_type, defender_types) for move_type in attacker_types)


def _weakness_summary(defender_type1: str, defender_type2: str) -> dict[str, float]:
    defender_types = [t for t in [defender_type1, defender_type2] if isinstance(t, str) and t != "None"]
    multipliers = [_type_multiplier(move_type, defender_types) for move_type in TYPE_ORDER]
    return {
        "weak_sum": float(sum(multipliers)),
        "weak_max": float(max(multipliers)),
        "weak_min": float(min(multipliers)),
        "immune_n": float(sum(multiplier == 0.0 for multiplier in multipliers)),
        "x4_n": float(sum(multiplier == 4.0 for multiplier in multipliers)),
        "x2_n": float(sum(multiplier == 2.0 for multiplier in multipliers)),
        "half_n": float(sum(multiplier == 0.5 for multiplier in multipliers)),
        "quarter_n": float(sum(multiplier == 0.25 for multiplier in multipliers)),
    }


def build_battle_dataset(master_df: pd.DataFrame, single_combats_df: pd.DataFrame) -> pd.DataFrame:
    numeric_lookup_columns = [
        "dexnum_int",
        "name",
        "generation",
        "height",
        "weight",
        "hp",
        "attack",
        "defense",
        "sp_atk",
        "sp_def",
        "speed",
        "total",
        "catch_rate",
        "base_friendship",
        "base_exp",
        "type1",
        "type2",
        "ability1",
        "ability2",
        "hidden_ability",
        "species",
        "special_group",
        "image_url",
    ]

    lookup_df = (
        master_df[master_df["dexnum_int"].between(1, 800)][numeric_lookup_columns]
        .sort_values("dexnum_int")
        .drop_duplicates("dexnum_int")
        .copy()
    )

    combat_df = single_combats_df.copy()
    combat_df["First_wins"] = (combat_df["Winner"] == combat_df["First_pokemon"]).astype(int)
    combat_df = combat_df.merge(lookup_df.add_prefix("A_"), left_on="First_pokemon", right_on="A_dexnum_int", how="inner")
    combat_df = combat_df.merge(lookup_df.add_prefix("B_"), left_on="Second_pokemon", right_on="B_dexnum_int", how="inner")

    for column in [
        "generation",
        "height",
        "weight",
        "hp",
        "attack",
        "defense",
        "sp_atk",
        "sp_def",
        "speed",
        "total",
        "catch_rate",
        "base_friendship",
        "base_exp",
    ]:
        combat_df[f"{column}_diff"] = combat_df[f"A_{column}"] - combat_df[f"B_{column}"]

    combat_df["speed_edge"] = np.sign(combat_df["speed_diff"]).astype(int)
    combat_df["A_best_stab"] = [
        _best_stab_multiplier(a1, a2, b1, b2)
        for a1, a2, b1, b2 in zip(
            combat_df["A_type1"], combat_df["A_type2"], combat_df["B_type1"], combat_df["B_type2"]
        )
    ]
    combat_df["B_best_stab"] = [
        _best_stab_multiplier(b1, b2, a1, a2)
        for a1, a2, b1, b2 in zip(
            combat_df["A_type1"], combat_df["A_type2"], combat_df["B_type1"], combat_df["B_type2"]
        )
    ]
    combat_df["stab_diff"] = combat_df["A_best_stab"] - combat_df["B_best_stab"]

    for side in ["A", "B"]:
        summaries = [
            _weakness_summary(type1, type2)
            for type1, type2 in zip(combat_df[f"{side}_type1"], combat_df[f"{side}_type2"])
        ]
        summary_df = pd.DataFrame(summaries).add_prefix(f"{side}_")
        combat_df = pd.concat([combat_df.reset_index(drop=True), summary_df.reset_index(drop=True)], axis=1)

    for summary_name in ["weak_sum", "weak_max", "weak_min", "immune_n", "x4_n", "x2_n", "half_n", "quarter_n"]:
        combat_df[f"{summary_name}_diff"] = combat_df[f"A_{summary_name}"] - combat_df[f"B_{summary_name}"]

    combat_df["unordered_pair"] = combat_df.apply(
        lambda row: tuple(sorted((int(row["First_pokemon"]), int(row["Second_pokemon"])))),
        axis=1,
    )
    return combat_df


def battle_feature_columns() -> tuple[list[str], list[str], list[str]]:
    categorical = [
        "A_type1",
        "A_type2",
        "B_type1",
        "B_type2",
        "A_ability1",
        "A_ability2",
        "A_hidden_ability",
        "B_ability1",
        "B_ability2",
        "B_hidden_ability",
        "A_species",
        "B_species",
        "A_special_group",
        "B_special_group",
    ]
    baseline = [
        "hp_diff",
        "attack_diff",
        "defense_diff",
        "sp_atk_diff",
        "sp_def_diff",
        "speed_diff",
        "total_diff",
        "A_best_stab",
        "B_best_stab",
        "stab_diff",
    ]
    full = [
        "A_generation",
        "B_generation",
        "A_height",
        "B_height",
        "A_weight",
        "B_weight",
        "A_hp",
        "B_hp",
        "A_attack",
        "B_attack",
        "A_defense",
        "B_defense",
        "A_sp_atk",
        "B_sp_atk",
        "A_sp_def",
        "B_sp_def",
        "A_speed",
        "B_speed",
        "A_total",
        "B_total",
        "generation_diff",
        "height_diff",
        "weight_diff",
        "hp_diff",
        "attack_diff",
        "defense_diff",
        "sp_atk_diff",
        "sp_def_diff",
        "speed_diff",
        "total_diff",
        "speed_edge",
        "A_best_stab",
        "B_best_stab",
        "stab_diff",
        "A_weak_sum",
        "B_weak_sum",
        "A_weak_max",
        "B_weak_max",
        "A_immune_n",
        "B_immune_n",
        "A_x4_n",
        "B_x4_n",
        "A_x2_n",
        "B_x2_n",
        "A_half_n",
        "B_half_n",
        "A_quarter_n",
        "B_quarter_n",
        "weak_sum_diff",
        "weak_max_diff",
        "weak_min_diff",
        "immune_n_diff",
        "x4_n_diff",
        "x2_n_diff",
        "half_n_diff",
        "quarter_n_diff",
    ] + categorical
    return baseline, full, categorical


def _battle_split(dataset: pd.DataFrame, split_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if split_mode == "random":
        train_idx, test_idx = train_test_split(
            dataset.index.to_numpy(),
            test_size=0.2,
            random_state=42,
            stratify=dataset["First_wins"],
        )
        return np.array(train_idx), np.array(test_idx)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(dataset, dataset["First_wins"], groups=dataset["unordered_pair"]))
    return train_idx, test_idx


def evaluate_battle_models(battle_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_columns, full_columns, categorical_columns = battle_feature_columns()
    reports: list[dict[str, Any]] = []
    fitted_models: dict[str, dict[str, Any]] = {}

    candidates = [
        ("Logistic Regression", "baseline", LogisticRegression(max_iter=700)),
        (
            "Random Forest",
            "full",
            RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        ),
        (
            "Extra Trees",
            "full",
            ExtraTreesClassifier(n_estimators=350, random_state=42, n_jobs=-1),
        ),
    ]

    for split_mode in ["random", "grouped"]:
        train_idx, test_idx = _battle_split(battle_df, split_mode)
        train_df = battle_df.iloc[train_idx]
        test_df = battle_df.iloc[test_idx]
        y_train = train_df["First_wins"]
        y_test = test_df["First_wins"]

        for display_name, feature_key, estimator in candidates:
            feature_columns = baseline_columns if feature_key == "baseline" else full_columns
            numeric_columns = [column for column in feature_columns if column not in categorical_columns]
            selected_categorical = [column for column in feature_columns if column in categorical_columns]
            preprocessor = _build_preprocessor(numeric_columns, selected_categorical)
            model = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            model.fit(train_df[feature_columns], y_train)
            y_pred = model.predict(test_df[feature_columns])
            y_proba = model.predict_proba(test_df[feature_columns])[:, 1]
            report = {
                "task": "battle_prediction",
                "model": display_name,
                "split": split_mode,
                "feature_set": feature_key,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
            reports.append(report)
            fitted_models.setdefault(display_name, {})[split_mode] = {
                "estimator": model,
                "feature_columns": feature_columns,
            }

    reports_df = pd.DataFrame(reports).sort_values(["split", "roc_auc", "accuracy"], ascending=[True, False, False])
    grouped_reports = reports_df[reports_df["split"] == "grouped"].sort_values(
        ["roc_auc", "accuracy"], ascending=[False, False]
    )
    best_row = grouped_reports.iloc[0]
    rf_row = grouped_reports[grouped_reports["model"] == "Random Forest"]
    if not rf_row.empty and abs(best_row["roc_auc"] - rf_row.iloc[0]["roc_auc"]) < 0.005:
        best_row = rf_row.iloc[0]

    final_model = fitted_models[best_row["model"]]["grouped"]
    return reports_df.reset_index(drop=True), {
        "final_model_name": best_row["model"],
        "final_estimator": final_model["estimator"],
        "final_feature_columns": final_model["feature_columns"],
    }


def build_battle_history_table(single_combats_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in single_combats_df.itertuples(index=False):
        dex_a = int(min(row.First_pokemon, row.Second_pokemon))
        dex_b = int(max(row.First_pokemon, row.Second_pokemon))
        winner = int(row.Winner)
        if winner == dex_a:
            win_a, win_b = 1, 0
        elif winner == dex_b:
            win_a, win_b = 0, 1
        else:
            win_a, win_b = 0, 0
        rows.append(
            {
                "dex_a": dex_a,
                "dex_b": dex_b,
                "wins_a": win_a,
                "wins_b": win_b,
                "total_battles": 1,
            }
        )
    history_df = pd.DataFrame(rows)
    history_df = (
        history_df.groupby(["dex_a", "dex_b"], as_index=False)[["wins_a", "wins_b", "total_battles"]]
        .sum()
        .sort_values(["dex_a", "dex_b"])
        .reset_index(drop=True)
    )
    return history_df


def train_project_bundle(root: Path | None = None, include_external: bool = False) -> dict[str, Any]:
    base = root or ROOT
    local_data = load_local_data(base)
    master_df = build_master_table(base, include_external=include_external)
    type_reports, type_bundle = evaluate_type_models(master_df)
    battle_df = build_battle_dataset(master_df, local_data["single_combats"])
    battle_reports, battle_bundle = evaluate_battle_models(battle_df)
    battle_history_df = build_battle_history_table(local_data["single_combats"])

    return {
        "root": str(base),
        "local_data_summary": summarize_local_data(local_data),
        "matchup_audit": audit_matchup_coverage(local_data),
        "master_df": master_df,
        "battle_df": battle_df,
        "battle_history_df": battle_history_df,
        "type_reports": type_reports,
        "battle_reports": battle_reports,
        "type_bundle": type_bundle,
        "battle_bundle": battle_bundle,
        "bundle_source": "trained_from_local_data",
    }


@lru_cache(maxsize=1)
def get_project_bundle(include_external: bool = False) -> dict[str, Any]:
    return train_project_bundle(ROOT, include_external=include_external)


def _train_deploy_battle_bundle(battle_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_columns, full_columns, categorical_columns = battle_feature_columns()
    grouped_train_idx, grouped_test_idx = _battle_split(battle_df, "grouped")
    random_train_idx, random_test_idx = _battle_split(battle_df, "random")

    def _fit_with_indices(train_idx: np.ndarray, test_idx: np.ndarray, split_name: str) -> tuple[dict[str, Any], Pipeline]:
        train_df = battle_df.iloc[train_idx]
        test_df = battle_df.iloc[test_idx]
        y_train = train_df["First_wins"]
        y_test = test_df["First_wins"]
        numeric_columns = [column for column in full_columns if column not in categorical_columns]
        selected_categorical = [column for column in full_columns if column in categorical_columns]
        preprocessor = _build_preprocessor(numeric_columns, selected_categorical)
        estimator = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(n_estimators=180, random_state=42, n_jobs=-1)),
            ]
        )
        estimator.fit(train_df[full_columns], y_train)
        y_pred = estimator.predict(test_df[full_columns])
        y_proba = estimator.predict_proba(test_df[full_columns])[:, 1]
        report = {
            "task": "battle_prediction",
            "model": "Random Forest (Deploy)",
            "split": split_name,
            "feature_set": "full",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        }
        return report, estimator

    grouped_report, grouped_estimator = _fit_with_indices(grouped_train_idx, grouped_test_idx, "grouped")
    random_report, _ = _fit_with_indices(random_train_idx, random_test_idx, "random")
    reports_df = pd.DataFrame([grouped_report, random_report]).sort_values(["split"])
    return reports_df.reset_index(drop=True), {
        "final_model_name": "Random Forest (Deploy)",
        "final_estimator": grouped_estimator,
        "final_feature_columns": full_columns,
        "deploy_ready": True,
    }


def train_deploy_bundle(root: Path | None = None, include_external: bool = False) -> dict[str, Any]:
    base = root or ROOT
    local_data = load_local_data(base)
    master_df = build_master_table(base, include_external=include_external)
    type_reports, type_bundle = evaluate_type_models(master_df)
    battle_df = build_battle_dataset(master_df, local_data["single_combats"])
    battle_reports, battle_bundle = _train_deploy_battle_bundle(battle_df)
    battle_history_df = build_battle_history_table(local_data["single_combats"])

    minimal_master_columns = [
        "dexnum_int",
        "name",
        "type1",
        "type2",
        "species",
        "height",
        "weight",
        "hp",
        "attack",
        "defense",
        "sp_atk",
        "sp_def",
        "speed",
        "total",
        "catch_rate",
        "base_friendship",
        "base_exp",
        "growth_rate",
        "egg_group1",
        "egg_group2",
        "special_group",
        "ability1",
        "ability2",
        "hidden_ability",
        "ev_yield",
        "percent_male",
        "percent_female",
        "egg_cycles",
        "generation",
        "image_url",
        "bulk_score",
        "offense_score",
        "physical_bias",
        "special_bias",
        "speed_rank_pct",
        "single_type_flag",
    ] + _available_optional_columns(master_df, OPTIONAL_EXTERNAL_NUMERIC_COLUMNS + OPTIONAL_EXTERNAL_CATEGORICAL_COLUMNS)

    return {
        "root": str(base),
        "master_df": master_df[minimal_master_columns].copy(),
        "battle_history_df": battle_history_df,
        "type_reports": type_reports,
        "battle_reports": battle_reports,
        "type_bundle": type_bundle,
        "battle_bundle": battle_bundle,
        "matchup_audit": audit_matchup_coverage(local_data),
        "bundle_source": "deploy_artifact",
    }


def save_deploy_bundle(bundle: dict[str, Any], path: Path | None = None, compress: int = 3) -> Path:
    target = path or DEPLOY_BUNDLE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=compress)
    return target


def load_deploy_bundle(path: Path | None = None) -> dict[str, Any]:
    target = path or DEPLOY_BUNDLE_PATH
    return joblib.load(target)


@lru_cache(maxsize=1)
def get_deploy_bundle(path: Path | None = None) -> dict[str, Any]:
    target = path or DEPLOY_BUNDLE_PATH
    if target.exists():
        return load_deploy_bundle(target)
    bundle = train_deploy_bundle(ROOT, include_external=False)
    save_deploy_bundle(bundle, target)
    return bundle


def _decode_type_prediction(probabilities: np.ndarray, classes: np.ndarray, threshold: float) -> tuple[str, str]:
    probability_map = {label: float(prob) for label, prob in zip(classes, probabilities)}
    non_none = [(label, prob) for label, prob in probability_map.items() if label != "None"]
    non_none.sort(key=lambda item: item[1], reverse=True)
    primary_type = non_none[0][0]
    secondary_type = "None"
    for label, prob in non_none[1:]:
        if label != primary_type and prob >= threshold:
            secondary_type = label
            break
    return primary_type, secondary_type


def _type_profile_explanation(row: pd.Series, predicted_primary: str, predicted_secondary: str) -> list[str]:
    numeric_focus = []
    baseline_means = {
        "hp": row.get("hp"),
        "attack": row.get("attack"),
        "defense": row.get("defense"),
        "sp_atk": row.get("sp_atk"),
        "sp_def": row.get("sp_def"),
        "speed": row.get("speed"),
    }
    strongest_stats = sorted(baseline_means.items(), key=lambda item: item[1] if item[1] is not None else -math.inf, reverse=True)[:2]
    for stat_name, stat_value in strongest_stats:
        numeric_focus.append(f"{stat_name.replace('_', ' ').title()} is a standout stat at {int(stat_value)}.")

    explanation = numeric_focus
    explanation.append(
        f"Ability profile: {row.get('ability1', 'Unknown')}"
        + (f" / {row.get('ability2')}" if row.get("ability2") not in {None, 'Unknown'} else "")
        + "."
    )
    explanation.append(f"Species tag in the dataset: {row.get('species', 'Unknown')}.")
    explanation.append(
        f"Model leans toward {predicted_primary}"
        + (f" + {predicted_secondary}" if predicted_secondary != "None" else "")
        + " because this profile is similar to known Pokemon with matching stat and ability patterns."
    )
    return explanation


def predict_types(pokemon_name_or_feature_row: str | pd.Series, bundle: dict[str, Any] | None = None) -> PredictionResult:
    project_bundle = bundle or get_project_bundle()
    master_df = project_bundle["master_df"]
    type_bundle = project_bundle["type_bundle"]

    if isinstance(pokemon_name_or_feature_row, pd.Series):
        row = pokemon_name_or_feature_row
    else:
        row_df = master_df[master_df["name"] == pokemon_name_or_feature_row]
        if row_df.empty:
            raise ValueError(f"Pokemon '{pokemon_name_or_feature_row}' was not found in the master table.")
        row = row_df.iloc[0]

    estimator = type_bundle["final_estimator"]
    threshold = type_bundle["final_threshold"]
    classes = type_bundle["mlb"].classes_
    feature_key = type_bundle["final_feature_key"]

    if isinstance(estimator, ClassifierChain):
        preprocessor = type_bundle["final_preprocessor"]
        transformed = preprocessor.transform(row.to_frame().T)
        raw_proba = estimator.predict_proba(transformed)
    else:
        raw_proba = estimator.predict_proba(row.to_frame().T)

    probabilities = _normalize_multilabel_proba(raw_proba)[0]
    predicted_primary, predicted_secondary = _decode_type_prediction(probabilities, classes, threshold)
    probability_table = (
        pd.DataFrame({"type": classes, "probability": probabilities})
        .query("type != 'None'")
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    return PredictionResult(
        {
            "name": row["name"],
            "dexnum_int": int(row["dexnum_int"]),
            "image_url": row.get("image_url", ""),
            "true_primary": row["type1"],
            "true_secondary": row["type2"],
            "predicted_primary": predicted_primary,
            "predicted_secondary": predicted_secondary,
            "probabilities": probability_table,
            "model_name": type_bundle["final_model_name"],
            "explanation": _type_profile_explanation(row, predicted_primary, predicted_secondary),
            "feature_key": feature_key,
        }
    )


def _battle_explanation(row: pd.Series, probability_a: float, probability_b: float) -> list[str]:
    explanation: list[str] = []
    if row["A_best_stab"] > row["B_best_stab"]:
        explanation.append(
            f"Pokemon A has the stronger immediate STAB matchup ({row['A_best_stab']:.2f}x vs {row['B_best_stab']:.2f}x)."
        )
    elif row["B_best_stab"] > row["A_best_stab"]:
        explanation.append(
            f"Pokemon B has the stronger immediate STAB matchup ({row['B_best_stab']:.2f}x vs {row['A_best_stab']:.2f}x)."
        )
    else:
        explanation.append("Neither side has a clear STAB matchup edge, so base stats matter more here.")

    stat_diffs = {
        "HP": row["hp_diff"],
        "Attack": row["attack_diff"],
        "Defense": row["defense_diff"],
        "Sp. Atk": row["sp_atk_diff"],
        "Sp. Def": row["sp_def_diff"],
        "Speed": row["speed_diff"],
        "Total": row["total_diff"],
    }
    top_swings = sorted(stat_diffs.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    for stat_name, diff in top_swings:
        favored_side = "A" if diff > 0 else "B"
        explanation.append(f"{stat_name} advantage favors Pokemon {favored_side} by {abs(int(diff))} points.")

    explanation.append(
        f"Final model probabilities: Pokemon A {probability_a:.1%}, Pokemon B {probability_b:.1%}."
    )
    return explanation


def predict_battle(
    pokemon_a: str | int,
    pokemon_b: str | int,
    bundle: dict[str, Any] | None = None,
) -> PredictionResult:
    project_bundle = bundle or get_project_bundle()
    master_df = project_bundle["master_df"]
    battle_df = project_bundle.get("battle_df")
    battle_history_df = project_bundle.get("battle_history_df")
    battle_bundle = project_bundle["battle_bundle"]

    def _lookup(identifier: str | int) -> pd.Series:
        if isinstance(identifier, int):
            subset = master_df[master_df["dexnum_int"] == identifier]
        else:
            subset = master_df[master_df["name"] == identifier]
        if subset.empty:
            raise ValueError(f"Pokemon '{identifier}' was not found in the master table.")
        return subset.iloc[0]

    row_a = _lookup(pokemon_a)
    row_b = _lookup(pokemon_b)
    single_row = build_battle_dataset(
        master_df=pd.concat([row_a.to_frame().T, row_b.to_frame().T], ignore_index=True),
        single_combats_df=pd.DataFrame(
            [{"First_pokemon": int(row_a["dexnum_int"]), "Second_pokemon": int(row_b["dexnum_int"]), "Winner": int(row_a["dexnum_int"])}]
        ),
    ).iloc[0]

    feature_columns = battle_bundle["final_feature_columns"]
    estimator = battle_bundle["final_estimator"]
    feature_frame = pd.DataFrame([single_row[feature_columns].to_dict()])
    probability_a = float(estimator.predict_proba(feature_frame)[0, 1])
    probability_b = 1.0 - probability_a
    predicted_winner = row_a["name"] if probability_a >= 0.5 else row_b["name"]

    dex_a = int(min(row_a["dexnum_int"], row_b["dexnum_int"]))
    dex_b = int(max(row_a["dexnum_int"], row_b["dexnum_int"]))
    if battle_history_df is not None:
        history_row = battle_history_df[
            (battle_history_df["dex_a"] == dex_a) & (battle_history_df["dex_b"] == dex_b)
        ]
        if history_row.empty:
            total_history = 0
            wins_a = 0
            wins_b = 0
        else:
            history_row = history_row.iloc[0]
            wins_low = int(history_row["wins_a"])
            wins_high = int(history_row["wins_b"])
            total_history = int(history_row["total_battles"])
            if int(row_a["dexnum_int"]) == dex_a:
                wins_a, wins_b = wins_low, wins_high
            else:
                wins_a, wins_b = wins_high, wins_low
    elif battle_df is not None:
        history_ab = battle_df[
            (battle_df["First_pokemon"] == int(row_a["dexnum_int"]))
            & (battle_df["Second_pokemon"] == int(row_b["dexnum_int"]))
        ]
        history_ba = battle_df[
            (battle_df["First_pokemon"] == int(row_b["dexnum_int"]))
            & (battle_df["Second_pokemon"] == int(row_a["dexnum_int"]))
        ]
        total_history = len(history_ab) + len(history_ba)
        wins_a = int(
            (history_ab["Winner"] == int(row_a["dexnum_int"])).sum()
            + (history_ba["Winner"] == int(row_a["dexnum_int"])).sum()
        )
        wins_b = int(
            (history_ab["Winner"] == int(row_b["dexnum_int"])).sum()
            + (history_ba["Winner"] == int(row_b["dexnum_int"])).sum()
        )
    else:
        total_history = 0
        wins_a = 0
        wins_b = 0

    return PredictionResult(
        {
            "pokemon_a": {
                "name": row_a["name"],
                "dexnum_int": int(row_a["dexnum_int"]),
                "type1": row_a["type1"],
                "type2": row_a["type2"],
                "image_url": row_a.get("image_url", ""),
                "total": int(row_a["total"]),
            },
            "pokemon_b": {
                "name": row_b["name"],
                "dexnum_int": int(row_b["dexnum_int"]),
                "type1": row_b["type1"],
                "type2": row_b["type2"],
                "image_url": row_b.get("image_url", ""),
                "total": int(row_b["total"]),
            },
            "predicted_winner": predicted_winner,
            "win_prob_a": probability_a,
            "win_prob_b": probability_b,
            "history": {
                "total_battles": int(total_history),
                "wins_a": wins_a,
                "wins_b": wins_b,
                "rate_a": float(wins_a / total_history) if total_history else None,
                "rate_b": float(wins_b / total_history) if total_history else None,
            },
            "model_name": battle_bundle["final_model_name"],
            "explanation": _battle_explanation(single_row, probability_a, probability_b),
            "feature_snapshot": pd.DataFrame(
                [
                    {
                        "feature": "A_best_stab",
                        "value": single_row["A_best_stab"],
                    },
                    {
                        "feature": "B_best_stab",
                        "value": single_row["B_best_stab"],
                    },
                    {
                        "feature": "total_diff",
                        "value": single_row["total_diff"],
                    },
                    {
                        "feature": "speed_diff",
                        "value": single_row["speed_diff"],
                    },
                    {
                        "feature": "weak_sum_diff",
                        "value": single_row["weak_sum_diff"],
                    },
                ]
            ),
        }
    )


def notebook_ready_tables(bundle: dict[str, Any]) -> dict[str, pd.DataFrame]:
    type_reports = bundle["type_reports"].copy()
    battle_reports = bundle["battle_reports"].copy()
    return {
        "type_random": type_reports[type_reports["split"] == "random"].reset_index(drop=True),
        "type_grouped": type_reports[type_reports["split"] == "grouped"].reset_index(drop=True),
        "battle_random": battle_reports[battle_reports["split"] == "random"].reset_index(drop=True),
        "battle_grouped": battle_reports[battle_reports["split"] == "grouped"].reset_index(drop=True),
    }
