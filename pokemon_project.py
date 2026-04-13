from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler

from canonical_pokemon import (
    ARTIFACTS_DIR,
    CANONICAL_DIR,
    build_case_study_candidates,
    ensure_canonical_tables,
    ensure_image_embeddings,
)


ROOT = Path(__file__).resolve().parent
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
    "Fire": {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0, "Bug": 2.0, "Rock": 0.5, "Dragon": 0.5, "Steel": 2.0},
    "Water": {"Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0, "Rock": 2.0, "Dragon": 0.5},
    "Electric": {"Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0, "Flying": 2.0, "Dragon": 0.5},
    "Grass": {"Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5, "Ground": 2.0, "Flying": 0.5, "Bug": 0.5, "Rock": 2.0, "Dragon": 0.5, "Steel": 0.5},
    "Ice": {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ground": 2.0, "Flying": 2.0, "Dragon": 2.0, "Steel": 0.5, "Ice": 0.5},
    "Fighting": {"Normal": 2.0, "Ice": 2.0, "Rock": 2.0, "Dark": 2.0, "Steel": 2.0, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Ghost": 0.0, "Fairy": 0.5},
    "Poison": {"Grass": 2.0, "Fairy": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0.0},
    "Ground": {"Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0, "Flying": 0.0, "Bug": 0.5, "Rock": 2.0, "Steel": 2.0},
    "Flying": {"Grass": 2.0, "Electric": 0.5, "Fighting": 2.0, "Bug": 2.0, "Rock": 0.5, "Steel": 0.5},
    "Psychic": {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5, "Dark": 0.0, "Steel": 0.5},
    "Bug": {"Grass": 2.0, "Fire": 0.5, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0, "Steel": 0.5, "Fairy": 0.5},
    "Rock": {"Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5, "Flying": 2.0, "Bug": 2.0, "Steel": 0.5},
    "Ghost": {"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5},
    "Dragon": {"Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0},
    "Dark": {"Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5, "Fairy": 0.5},
    "Steel": {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0, "Rock": 2.0, "Fairy": 2.0, "Steel": 0.5},
    "Fairy": {"Fire": 0.5, "Fighting": 2.0, "Poison": 0.5, "Dragon": 2.0, "Dark": 2.0, "Steel": 0.5},
}

LEGACY_NUMERIC_COLUMNS = [
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

LEGACY_CATEGORICAL_COLUMNS = ["growth_rate", "egg_group1", "egg_group2", "special_group", "species", "ability1", "ability2", "hidden_ability"]

TYPE_STRUCTURED_NUMERIC_COLUMNS = [
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
    "capture_rate",
    "percent_male",
    "percent_female",
    "egg_cycles",
    "base_happiness",
    "base_exp",
    "evolution_chain_id",
    "evolution_stage",
    "move_count",
    "pokemon_api_ability_count",
    "is_baby",
    "is_legendary",
    "is_mythical",
    "is_mega",
    "is_battle_only",
    "single_type_flag",
    "bulk_score",
    "offense_score",
    "physical_bias",
    "special_bias",
    "speed_rank_pct",
]

TYPE_STRUCTURED_CATEGORICAL_COLUMNS = [
    "growth_rate",
    "egg_group1",
    "egg_group2",
    "ability1",
    "ability2",
    "hidden_ability",
    "color",
    "shape",
    "habitat",
    "region_tag",
    "form_group",
    "special_group",
]

TEXT_COLUMNS = ["display_name", "species_display_name", "genus_en", "flavor_text_en", "flavor_text_corpus_en", "ability_summary_en", "text_corpus_en"]

TYPE_MODEL_CANDIDATES = [
    {"model": "Structured OVR Logistic", "feature_mode": "structured", "model_key": "ovr_logistic", "deploy_priority": 2},
    {"model": "Structured ClassifierChain Logistic (C=10)", "feature_mode": "structured", "model_key": "chain_logistic", "deploy_priority": 1},
    {"model": "Structured ExtraTrees", "feature_mode": "structured", "model_key": "extra_trees", "deploy_priority": 3},
    {"model": "Structured + Text OVR Logistic", "feature_mode": "structured_text", "model_key": "ovr_logistic", "deploy_priority": 2},
    {"model": "Structured + Text ClassifierChain Logistic (C=10)", "feature_mode": "structured_text", "model_key": "chain_logistic", "deploy_priority": 1},
    {"model": "Multimodal Logistic Regression", "feature_mode": "multimodal", "model_key": "ovr_logistic", "deploy_priority": 4},
    {"model": "Multimodal ExtraTrees", "feature_mode": "multimodal", "model_key": "extra_trees", "deploy_priority": 5},
]

TYPE_MODEL_LOOKUP = {row["model"]: row for row in TYPE_MODEL_CANDIDATES}


@dataclass
class PredictionResult:
    payload: dict[str, Any]


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
        rows.append({"dataset": name, "rows": len(df), "columns": len(df.columns), "missing_cells": int(df.isna().sum().sum())})
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def audit_matchup_coverage(local_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    single_combats = local_data["single_combats"].copy()
    matchup = local_data["type_matchup"].copy()
    matchup_ids = matchup["Number"].astype(str).str.extract(r"(\d+)")[0].dropna().astype(int).tolist()
    matchup_id_set = set(matchup_ids)
    both_covered = single_combats["First_pokemon"].isin(matchup_id_set) & single_combats["Second_pokemon"].isin(matchup_id_set)
    unique_combat_ids = pd.unique(single_combats[["First_pokemon", "Second_pokemon", "Winner"]].values.ravel())
    return pd.DataFrame(
        [
            {
                "original_single_combats_rows": int(len(single_combats)),
                "rows_after_old_incomplete_matchup_join": int(both_covered.sum()),
                "rows_lost_by_old_join": int(len(single_combats) - both_covered.sum()),
                "coverage_ratio": float(both_covered.mean()),
                "unique_pokemon_ids_in_combats": int(len(unique_combat_ids)),
                "unique_ids_covered_by_matchup_csv": int(sum(int(p) in matchup_id_set for p in unique_combat_ids)),
            }
        ]
    )


def clean_legacy_pokemon_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df[df["name"].notna() & df["type1"].notna()].copy()
    for column in LEGACY_NUMERIC_COLUMNS + ["dexnum"]:
        df[column] = df[column].apply(_parse_number)
    df["dexnum_int"] = df["dexnum"].astype(int)
    df["type1"] = df["type1"].astype(str)
    df["type2"] = df["type2"].fillna("None").astype(str)
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
    df["single_type_flag"] = df["type2"].eq("None").astype(int)
    return df.reset_index(drop=True)


def build_master_table(root: Path | None = None, include_external: bool = False, **_: Any) -> pd.DataFrame:
    _ = include_external
    return _sanitize_type_master(ensure_canonical_tables(root)["pokemon_master"].copy())


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _move_count_columns(master_df: pd.DataFrame) -> list[str]:
    return sorted([column for column in master_df.columns if column.startswith("move_type_count_") or column.startswith("move_class_count_")])


def _structured_columns(master_df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = _available_columns(master_df, TYPE_STRUCTURED_NUMERIC_COLUMNS + _move_count_columns(master_df))
    categorical = _available_columns(master_df, TYPE_STRUCTURED_CATEGORICAL_COLUMNS)
    return {"numeric": numeric, "categorical": categorical}


def _sanitize_type_master(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()
    string_fill_map = {
        "type1": "Unknown",
        "type2": "None",
        "display_name": "",
        "canonical_slug": "",
        "species_slug": "",
        "species_display_name": "",
        "genus_en": "",
        "flavor_text_en": "",
        "flavor_text_corpus_en": "",
        "ability_summary_en": "",
        "text_corpus_en": "",
        "image_url": "",
        "official_artwork_url": "",
        "artwork_variant_url": "",
        "sprite_url": "",
        "official_pokedex_url": "",
        "validation_status": "unknown",
        "growth_rate": "Unknown",
        "egg_group1": "Unknown",
        "egg_group2": "Unknown",
        "ability1": "Unknown",
        "ability2": "Unknown",
        "hidden_ability": "Unknown",
        "color": "Unknown",
        "shape": "Unknown",
        "habitat": "Unknown",
        "region_tag": "None",
        "form_group": "Standard",
        "special_group": "Ordinary",
    }
    for column, fill_value in string_fill_map.items():
        if column in df.columns:
            df[column] = df[column].fillna(fill_value).astype(str)
    return df


def _build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
        ]
    )


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


def _assemble_type_feature_bundle(
    train_df: pd.DataFrame,
    feature_mode: str,
    image_embedding_df: pd.DataFrame | None,
) -> dict[str, Any]:
    structured = _structured_columns(train_df)
    preprocessor = _build_preprocessor(structured["numeric"], structured["categorical"])
    X_structured = preprocessor.fit_transform(train_df[structured["numeric"] + structured["categorical"]])
    pieces: list[sparse.csr_matrix] = [sparse.csr_matrix(X_structured)]
    bundle: dict[str, Any] = {
        "feature_mode": feature_mode,
        "structured_columns": structured,
        "preprocessor": preprocessor,
        "uses_text": feature_mode in {"structured_text", "multimodal"},
        "uses_image": feature_mode == "multimodal",
    }

    if bundle["uses_text"]:
        text_series = train_df["text_corpus_en"].fillna("")
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
        X_text = vectorizer.fit_transform(text_series)
        text_svd = None
        if X_text.shape[0] >= 3 and X_text.shape[1] >= 3:
            n_components = max(2, min(128, X_text.shape[0] - 1, X_text.shape[1] - 1))
            text_svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_text_dense = text_svd.fit_transform(X_text)
            pieces.append(sparse.csr_matrix(X_text_dense))
        else:
            pieces.append(sparse.csr_matrix(X_text))
        bundle["text_vectorizer"] = vectorizer
        bundle["text_svd"] = text_svd

    if bundle["uses_image"]:
        if image_embedding_df is None:
            raise ValueError("Multimodal models require image embeddings.")
        image_lookup = image_embedding_df.set_index("canonical_slug")
        image_columns = [column for column in image_embedding_df.columns if column.startswith("img_")]
        X_image = image_lookup.loc[train_df["canonical_slug"], image_columns].to_numpy(dtype=np.float32)
        pieces.append(sparse.csr_matrix(X_image))
        bundle["image_lookup"] = image_lookup
        bundle["image_columns"] = image_columns

    bundle["matrix"] = sparse.hstack(pieces).tocsr()
    return bundle


def _transform_type_features(df: pd.DataFrame, feature_bundle: dict[str, Any]) -> sparse.csr_matrix:
    structured = feature_bundle["structured_columns"]
    pieces: list[sparse.csr_matrix] = [
        sparse.csr_matrix(feature_bundle["preprocessor"].transform(df[structured["numeric"] + structured["categorical"]]))
    ]
    if feature_bundle.get("uses_text"):
        X_text = feature_bundle["text_vectorizer"].transform(df["text_corpus_en"].fillna(""))
        if feature_bundle.get("text_svd") is not None:
            X_text = sparse.csr_matrix(feature_bundle["text_svd"].transform(X_text))
        else:
            X_text = sparse.csr_matrix(X_text)
        pieces.append(X_text)
    if feature_bundle.get("uses_image"):
        image_lookup = feature_bundle["image_lookup"]
        image_columns = feature_bundle["image_columns"]
        X_image = image_lookup.loc[df["canonical_slug"], image_columns].to_numpy(dtype=np.float32)
        pieces.append(sparse.csr_matrix(X_image))
    return sparse.hstack(pieces).tocsr()


def _fit_type_estimator(model_key: str, X_train: sparse.csr_matrix, y_train: np.ndarray) -> Any:
    if model_key == "ovr_logistic":
        estimator = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    elif model_key == "chain_logistic":
        estimator = ClassifierChain(LogisticRegression(max_iter=1200, C=10.0), order="random", random_state=42)
    else:
        estimator = MultiOutputClassifier(ExtraTreesClassifier(n_estimators=180, random_state=42, n_jobs=-1))
    try:
        estimator.fit(X_train, y_train)
    except ValueError:
        if model_key != "chain_logistic":
            raise
        estimator = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        estimator.fit(X_train, y_train)
    return estimator


def _predict_type_labels(estimator: Any, X_eval: sparse.csr_matrix) -> np.ndarray:
    return estimator.predict(X_eval)


def _predict_type_probabilities(estimator: Any, X_eval: sparse.csr_matrix) -> np.ndarray:
    return _normalize_multilabel_proba(estimator.predict_proba(X_eval))


def _label_matrix(master_df: pd.DataFrame, mlb: MultiLabelBinarizer | None = None) -> tuple[np.ndarray, MultiLabelBinarizer]:
    local_mlb = mlb or MultiLabelBinarizer()
    labels_df = master_df[["type1", "type2"]].fillna("None").astype(str)
    label_values = labels_df.values.tolist()
    y_all = local_mlb.fit_transform(label_values) if mlb is None else local_mlb.transform(label_values)
    return y_all, local_mlb


def _group_values(master_df: pd.DataFrame, split_mode: str) -> pd.Series:
    if split_mode == "species_group":
        return master_df["species_slug"].fillna(master_df["canonical_slug"])
    evolution_groups = master_df["evolution_chain_id"].fillna(0).astype(int).astype(str)
    return np.where(evolution_groups == "0", master_df["canonical_slug"], evolution_groups)


def _type_split_indices(master_df: pd.DataFrame, split_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if split_mode == "random":
        train_idx, test_idx = train_test_split(master_df.index.to_numpy(), test_size=0.2, random_state=42)
        return np.array(train_idx), np.array(test_idx)
    groups = _group_values(master_df, split_mode)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(master_df, groups=groups))
    return train_idx, test_idx


def evaluate_legacy_type_models(legacy_master_df: pd.DataFrame) -> pd.DataFrame:
    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(legacy_master_df[["type1", "type2"]].values.tolist())
    train_idx, test_idx = train_test_split(legacy_master_df.index.to_numpy(), test_size=0.2, random_state=42)
    X_train = legacy_master_df.iloc[train_idx]
    X_test = legacy_master_df.iloc[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    preprocessor = _build_preprocessor(LEGACY_NUMERIC_COLUMNS, LEGACY_CATEGORICAL_COLUMNS)

    candidates = [
        ("Legacy OVR Logistic", OneVsRestClassifier(LogisticRegression(max_iter=700))),
        ("Legacy ClassifierChain Logistic (C=10)", ClassifierChain(LogisticRegression(max_iter=800, C=10.0), order="random", random_state=42)),
        ("Legacy ExtraTrees", MultiOutputClassifier(ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1))),
    ]

    reports = []
    X_train_processed = preprocessor.fit_transform(X_train[LEGACY_NUMERIC_COLUMNS + LEGACY_CATEGORICAL_COLUMNS])
    X_test_processed = preprocessor.transform(X_test[LEGACY_NUMERIC_COLUMNS + LEGACY_CATEGORICAL_COLUMNS])
    for model_name, estimator in candidates:
        estimator.fit(X_train_processed, y_train)
        y_pred = estimator.predict(X_test_processed)
        reports.append(
            {
                "task": "legacy_type_prediction",
                "model": model_name,
                "split": "legacy_random",
                "micro_f1": float(f1_score(y_test, y_pred, average="micro")),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "hamming_loss": float(hamming_loss(y_test, y_pred)),
                "exact_match": float(accuracy_score(y_test, y_pred)),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            }
        )
    return pd.DataFrame(reports).sort_values(["exact_match", "micro_f1"], ascending=[False, False]).reset_index(drop=True)


def _candidate_oof_summary(
    master_df: pd.DataFrame,
    candidate: dict[str, Any],
    y_all: np.ndarray,
    mlb: MultiLabelBinarizer,
    image_embedding_df: pd.DataFrame | None,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _group_values(master_df, "evolution_group")
    unique_groups = pd.Series(groups).nunique()
    fold_count = max(2, min(n_splits, int(unique_groups)))
    splitter = GroupKFold(n_splits=fold_count)
    detail_rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(master_df, y_all, groups=groups), start=1):
        train_df = master_df.iloc[train_idx]
        test_df = master_df.iloc[test_idx]
        feature_bundle = _assemble_type_feature_bundle(train_df, candidate["feature_mode"], image_embedding_df)
        estimator = _fit_type_estimator(candidate["model_key"], feature_bundle["matrix"], y_all[train_idx])
        X_eval = _transform_type_features(test_df, feature_bundle)
        y_pred = _predict_type_labels(estimator, X_eval)
        probabilities = _predict_type_probabilities(estimator, X_eval)

        for row, pred, proba in zip(test_df.itertuples(index=False), y_pred, probabilities):
            pred_primary, pred_secondary = _decode_type_prediction(proba, mlb.classes_, 0.30)
            true_primary = "Unknown" if pd.isna(row.type1) else str(row.type1)
            true_secondary = "None" if pd.isna(row.type2) else str(row.type2)
            true_set = {label for label in [true_primary, true_secondary] if label != "None"}
            predicted_set = {label for label in [pred_primary, pred_secondary] if label != "None"}
            overlap_n = len(true_set & predicted_set)
            if true_set == predicted_set:
                bucket = "all_types_correct"
            elif overlap_n == 1:
                bucket = "one_type_correct"
            else:
                bucket = "zero_type_correct"
            detail_rows.append(
                {
                    "model": candidate["model"],
                    "feature_mode": candidate["feature_mode"],
                    "fold": fold,
                    "canonical_slug": row.canonical_slug,
                    "display_name": row.display_name,
                    "species_slug": row.species_slug,
                    "evolution_chain_id": row.evolution_chain_id,
                    "true_primary": true_primary,
                    "true_secondary": true_secondary,
                    "predicted_primary": pred_primary,
                    "predicted_secondary": pred_secondary,
                    "matched_type_count": overlap_n,
                    "result_bucket": bucket,
                    "ordered_match": bool(pred_primary == row.type1 and pred_secondary == row.type2),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    y_true = mlb.transform(detail_df[["true_primary", "true_secondary"]].fillna("None").astype(str).values.tolist())
    y_pred = mlb.transform(detail_df[["predicted_primary", "predicted_secondary"]].fillna("None").astype(str).values.tolist())
    summary_df = pd.DataFrame(
        [
            {
                "model": candidate["model"],
                "feature_mode": candidate["feature_mode"],
                "split": "evolution_group_oof",
                "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "exact_match": float(accuracy_score(y_true, y_pred)),
                "all_types_correct_n": int((detail_df["result_bucket"] == "all_types_correct").sum()),
                "one_type_correct_n": int((detail_df["result_bucket"] == "one_type_correct").sum()),
                "zero_type_correct_n": int((detail_df["result_bucket"] == "zero_type_correct").sum()),
                "n_total": int(len(detail_df)),
                "ordered_match_n": int(detail_df["ordered_match"].sum()),
                "deploy_priority": int(candidate["deploy_priority"]),
            }
        ]
    )
    return summary_df, detail_df


def _select_type_model(oof_summary_df: pd.DataFrame) -> pd.Series:
    ranked = oof_summary_df.sort_values(["exact_match", "micro_f1"], ascending=[False, False]).reset_index(drop=True)
    best = ranked.iloc[0]
    nearby = ranked[ranked["exact_match"] >= best["exact_match"] - 0.01].copy()
    non_multimodal = nearby[nearby["feature_mode"] != "multimodal"].copy()
    if not non_multimodal.empty:
        nearby = non_multimodal
    nearby = nearby.sort_values(["deploy_priority", "micro_f1"], ascending=[True, False]).reset_index(drop=True)
    return nearby.iloc[0]


def evaluate_type_models(
    master_df: pd.DataFrame,
    image_embedding_df: pd.DataFrame | None = None,
    run_image_experiment: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    candidates = TYPE_MODEL_CANDIDATES if run_image_experiment else [row for row in TYPE_MODEL_CANDIDATES if row["feature_mode"] != "multimodal"]
    y_all, mlb = _label_matrix(master_df)

    split_reports: list[dict[str, Any]] = []
    for split_mode in ["random", "species_group"]:
        print(f"[type] holdout split: {split_mode}")
        train_idx, test_idx = _type_split_indices(master_df, split_mode)
        train_df = master_df.iloc[train_idx]
        test_df = master_df.iloc[test_idx]
        for candidate in candidates:
            print(f"[type] fitting {candidate['model']} on {split_mode}")
            feature_bundle = _assemble_type_feature_bundle(train_df, candidate["feature_mode"], image_embedding_df)
            estimator = _fit_type_estimator(candidate["model_key"], feature_bundle["matrix"], y_all[train_idx])
            X_eval = _transform_type_features(test_df, feature_bundle)
            y_pred = _predict_type_labels(estimator, X_eval)
            split_reports.append(
                {
                    "task": "type_prediction",
                    "model": candidate["model"],
                    "feature_mode": candidate["feature_mode"],
                    "split": split_mode,
                    "micro_f1": float(f1_score(y_all[test_idx], y_pred, average="micro")),
                    "macro_f1": float(f1_score(y_all[test_idx], y_pred, average="macro")),
                    "hamming_loss": float(hamming_loss(y_all[test_idx], y_pred)),
                    "exact_match": float(accuracy_score(y_all[test_idx], y_pred)),
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                }
            )

    oof_summaries = []
    oof_details = []
    for candidate in candidates:
        print(f"[type] evolution-group OOF: {candidate['model']}")
        summary_df, detail_df = _candidate_oof_summary(master_df, candidate, y_all, mlb, image_embedding_df)
        oof_summaries.append(summary_df)
        oof_details.append(detail_df)

    oof_summary_df = pd.concat(oof_summaries, ignore_index=True).sort_values(["exact_match", "micro_f1"], ascending=[False, False]).reset_index(drop=True)
    oof_detail_df = pd.concat(oof_details, ignore_index=True).sort_values(["model", "fold", "canonical_slug"]).reset_index(drop=True)
    selected_row = _select_type_model(oof_summary_df)
    selected_candidate = TYPE_MODEL_LOOKUP[str(selected_row["model"])]

    final_feature_bundle = _assemble_type_feature_bundle(master_df, selected_candidate["feature_mode"], image_embedding_df)
    final_estimator = _fit_type_estimator(selected_candidate["model_key"], final_feature_bundle["matrix"], y_all)

    reports_df = pd.DataFrame(split_reports).sort_values(["split", "exact_match", "micro_f1"], ascending=[True, False, False]).reset_index(drop=True)
    type_bundle = {
        "mlb": mlb,
        "final_model_name": selected_candidate["model"],
        "final_feature_mode": selected_candidate["feature_mode"],
        "final_model_key": selected_candidate["model_key"],
        "final_estimator": final_estimator,
        "final_feature_bundle": final_feature_bundle,
        "final_threshold": 0.30,
        "selection_summary": oof_summary_df.copy(),
        "uses_image": bool(selected_candidate["feature_mode"] == "multimodal"),
    }
    return reports_df, oof_summary_df, type_bundle, oof_summary_df.copy(), oof_detail_df


def type_oof_statistics(
    master_df: pd.DataFrame,
    type_bundle: dict[str, Any],
    split_mode: str = "evolution_group",
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mlb = type_bundle["mlb"]
    y_all = mlb.transform(master_df[["type1", "type2"]].values.tolist())
    feature_mode = type_bundle["final_feature_mode"]
    model_key = type_bundle["final_model_key"]
    image_lookup = None
    if type_bundle["final_feature_bundle"].get("uses_image"):
        lookup = type_bundle["final_feature_bundle"]["image_lookup"].reset_index()
        image_lookup = lookup
    if split_mode == "random":
        fold_count = max(2, min(n_splits, len(master_df)))
        splitter = KFold(n_splits=fold_count, shuffle=True, random_state=42)
        iterator = splitter.split(master_df, y_all)
    else:
        groups = _group_values(master_df, split_mode)
        fold_count = max(2, min(n_splits, int(pd.Series(groups).nunique())))
        splitter = GroupKFold(n_splits=fold_count)
        iterator = splitter.split(master_df, y_all, groups=groups)

    detail_rows: list[dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(iterator, start=1):
        train_df = master_df.iloc[train_idx]
        test_df = master_df.iloc[test_idx]
        feature_bundle = _assemble_type_feature_bundle(train_df, feature_mode, image_lookup)
        estimator = _fit_type_estimator(model_key, feature_bundle["matrix"], y_all[train_idx])
        probabilities = _predict_type_probabilities(estimator, _transform_type_features(test_df, feature_bundle))
        for row, proba in zip(test_df.itertuples(index=False), probabilities):
            pred_primary, pred_secondary = _decode_type_prediction(proba, mlb.classes_, type_bundle.get("final_threshold", 0.30))
            true_primary = "Unknown" if pd.isna(row.type1) else str(row.type1)
            true_secondary = "None" if pd.isna(row.type2) else str(row.type2)
            true_set = {label for label in [true_primary, true_secondary] if label != "None"}
            predicted_set = {label for label in [pred_primary, pred_secondary] if label != "None"}
            overlap_n = len(true_set & predicted_set)
            bucket = "all_types_correct" if true_set == predicted_set else ("one_type_correct" if overlap_n == 1 else "zero_type_correct")
            detail_rows.append(
                {
                    "split_mode": split_mode,
                    "fold": fold_id,
                    "canonical_slug": row.canonical_slug,
                    "display_name": row.display_name,
                    "true_primary": true_primary,
                    "true_secondary": true_secondary,
                    "predicted_primary": pred_primary,
                    "predicted_secondary": pred_secondary,
                    "all_types_correct": bucket == "all_types_correct",
                    "ordered_match": pred_primary == row.type1 and pred_secondary == row.type2,
                    "matched_type_count": overlap_n,
                    "result_bucket": bucket,
                }
            )
    detail_df = pd.DataFrame(detail_rows).sort_values(["fold", "canonical_slug"]).reset_index(drop=True)
    n_total = len(detail_df)
    all_types_correct_n = int((detail_df["result_bucket"] == "all_types_correct").sum())
    one_type_correct_n = int((detail_df["result_bucket"] == "one_type_correct").sum())
    zero_type_correct_n = int((detail_df["result_bucket"] == "zero_type_correct").sum())
    summary_df = pd.DataFrame(
        [
            {
                "split_mode": split_mode,
                "model": type_bundle["final_model_name"],
                "feature_mode": type_bundle["final_feature_mode"],
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


def build_type_oof_summary_table(master_df: pd.DataFrame, type_bundle: dict[str, Any]) -> pd.DataFrame:
    summaries = []
    for split_mode in ["random", "species_group", "evolution_group"]:
        summary_df, _ = type_oof_statistics(master_df, type_bundle, split_mode=split_mode, n_splits=5)
        summaries.append(summary_df)
    return pd.concat(summaries, ignore_index=True)


def _flatten_evidence_panels(panels: dict[str, list[str]]) -> list[str]:
    lines = []
    for title, items in panels.items():
        lines.append(f"{title}:")
        lines.extend(items)
    return lines


def _type_profile_explanation(row: pd.Series, result: dict[str, Any], feature_mode: str) -> dict[str, list[str]]:
    strongest_stats = sorted(
        [("HP", row.get("hp")), ("Attack", row.get("attack")), ("Defense", row.get("defense")), ("Sp. Atk", row.get("sp_atk")), ("Sp. Def", row.get("sp_def")), ("Speed", row.get("speed"))],
        key=lambda item: item[1] if item[1] is not None else -math.inf,
        reverse=True,
    )[:2]
    structured_lines = [f"{name} is a standout stat at {int(value)}." for name, value in strongest_stats if value is not None]
    structured_lines.append(f"Ability profile: {row.get('ability1', 'Unknown')}" + (f" / {row.get('ability2')}" if row.get("ability2") not in {None, 'Unknown'} else "") + ".")
    structured_lines.append(f"Move pool size is {int(row.get('move_count', 0))}, with color/shape cues recorded as {row.get('color', 'Unknown')} / {row.get('shape', 'Unknown')}.")

    text_lines = []
    if row.get("genus_en"):
        text_lines.append(f"Genus tag: {row['genus_en']}.")
    if row.get("flavor_text_en"):
        text_lines.append(f"Pokedex text cue: {row['flavor_text_en'][:110]}...")
    if not text_lines:
        text_lines.append("No strong text cue was available for this specimen.")

    if feature_mode == "multimodal":
        image_lines = ["Image embeddings from the recorded artwork and sprite were used in this experimental model."]
    else:
        image_lines = ["Image embeddings were evaluated in the notebook, but the deployed model does not rely on them."]

    return {"STRUCTURED SIGNALS": structured_lines, "TEXT SIGNALS": text_lines, "IMAGE SIGNALS": image_lines}


def predict_types(pokemon_name_or_feature_row: str | pd.Series, bundle: dict[str, Any] | None = None) -> PredictionResult:
    project_bundle = bundle or get_project_bundle()
    master_df = project_bundle["type_master_df"]
    type_bundle = project_bundle["type_bundle"]

    if isinstance(pokemon_name_or_feature_row, pd.Series):
        row = pokemon_name_or_feature_row
    else:
        row_df = master_df[(master_df["display_name"] == pokemon_name_or_feature_row) | (master_df["canonical_slug"] == pokemon_name_or_feature_row)]
        if row_df.empty:
            raise ValueError(f"Pokemon '{pokemon_name_or_feature_row}' was not found in the canonical master table.")
        row = row_df.iloc[0]

    feature_bundle = type_bundle["final_feature_bundle"]
    estimator = type_bundle["final_estimator"]
    threshold = float(type_bundle.get("final_threshold", 0.30))
    X_eval = _transform_type_features(row.to_frame().T, feature_bundle)
    probabilities = _predict_type_probabilities(estimator, X_eval)[0]
    classes = type_bundle["mlb"].classes_
    predicted_primary, predicted_secondary = _decode_type_prediction(probabilities, classes, threshold)
    probability_table = (
        pd.DataFrame({"type": classes, "probability": probabilities})
        .query("type != 'None'")
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    evidence_panels = _type_profile_explanation(row, {}, type_bundle["final_feature_mode"])
    return PredictionResult(
        {
            "name": row["display_name"],
            "canonical_slug": row["canonical_slug"],
            "dexnum_int": int(row["dexnum_int"]),
            "image_url": row.get("image_url", ""),
            "official_artwork_url": row.get("official_artwork_url", ""),
            "sprite_url": row.get("sprite_url", ""),
            "official_pokedex_url": row.get("official_pokedex_url", ""),
            "validation_status": row.get("validation_status", "unknown"),
            "source_label": "canonical_hybrid_pokeapi_plus_official_refs",
            "true_primary": row["type1"],
            "true_secondary": row["type2"],
            "predicted_primary": predicted_primary,
            "predicted_secondary": predicted_secondary,
            "probabilities": probability_table,
            "model_name": type_bundle["final_model_name"],
            "feature_mode": type_bundle["final_feature_mode"],
            "explanation": _flatten_evidence_panels(evidence_panels),
            "evidence_panels": evidence_panels,
            "provenance": {
                "species_display_name": row.get("species_display_name", row.get("display_name")),
                "validation_status": row.get("validation_status", "unknown"),
                "official_pokedex_url": row.get("official_pokedex_url", ""),
                "official_artwork_url": row.get("official_artwork_url", ""),
            },
        }
    )


def notebook_ready_tables(bundle: dict[str, Any]) -> dict[str, pd.DataFrame]:
    type_reports = bundle["type_reports"].copy()
    battle_reports = bundle["battle_reports"].copy()
    return {
        "type_random": type_reports[type_reports["split"] == "random"].reset_index(drop=True),
        "type_grouped": type_reports[type_reports["split"] == "species_group"].reset_index(drop=True),
        "type_evolution_oof": bundle["type_oof_benchmark"].copy(),
        "battle_random": battle_reports[battle_reports["split"] == "random"].reset_index(drop=True),
        "battle_grouped": battle_reports[battle_reports["split"] == "grouped"].reset_index(drop=True),
    }


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
    lookup_columns = [
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
    lookup_df = master_df[lookup_columns].sort_values("dexnum_int").drop_duplicates("dexnum_int").copy()
    combat_df = single_combats_df.copy()
    combat_df["First_wins"] = (combat_df["Winner"] == combat_df["First_pokemon"]).astype(int)
    combat_df = combat_df.merge(lookup_df.add_prefix("A_"), left_on="First_pokemon", right_on="A_dexnum_int", how="inner")
    combat_df = combat_df.merge(lookup_df.add_prefix("B_"), left_on="Second_pokemon", right_on="B_dexnum_int", how="inner")
    for column in ["generation", "height", "weight", "hp", "attack", "defense", "sp_atk", "sp_def", "speed", "total", "catch_rate", "base_friendship", "base_exp"]:
        combat_df[f"{column}_diff"] = combat_df[f"A_{column}"] - combat_df[f"B_{column}"]
    combat_df["speed_edge"] = np.sign(combat_df["speed_diff"]).astype(int)
    combat_df["A_best_stab"] = [_best_stab_multiplier(a1, a2, b1, b2) for a1, a2, b1, b2 in zip(combat_df["A_type1"], combat_df["A_type2"], combat_df["B_type1"], combat_df["B_type2"])]
    combat_df["B_best_stab"] = [_best_stab_multiplier(b1, b2, a1, a2) for a1, a2, b1, b2 in zip(combat_df["A_type1"], combat_df["A_type2"], combat_df["B_type1"], combat_df["B_type2"])]
    combat_df["stab_diff"] = combat_df["A_best_stab"] - combat_df["B_best_stab"]
    for side in ["A", "B"]:
        summaries = [_weakness_summary(type1, type2) for type1, type2 in zip(combat_df[f"{side}_type1"], combat_df[f"{side}_type2"])]
        combat_df = pd.concat([combat_df.reset_index(drop=True), pd.DataFrame(summaries).add_prefix(f"{side}_").reset_index(drop=True)], axis=1)
    for summary_name in ["weak_sum", "weak_max", "weak_min", "immune_n", "x4_n", "x2_n", "half_n", "quarter_n"]:
        combat_df[f"{summary_name}_diff"] = combat_df[f"A_{summary_name}"] - combat_df[f"B_{summary_name}"]
    combat_df["unordered_pair"] = combat_df.apply(lambda row: tuple(sorted((int(row["First_pokemon"]), int(row["Second_pokemon"])))), axis=1)
    return combat_df


def battle_feature_columns() -> tuple[list[str], list[str], list[str]]:
    categorical = ["A_type1", "A_type2", "B_type1", "B_type2", "A_ability1", "A_ability2", "A_hidden_ability", "B_ability1", "B_ability2", "B_hidden_ability", "A_species", "B_species", "A_special_group", "B_special_group"]
    baseline = ["hp_diff", "attack_diff", "defense_diff", "sp_atk_diff", "sp_def_diff", "speed_diff", "total_diff", "A_best_stab", "B_best_stab", "stab_diff"]
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
        train_idx, test_idx = train_test_split(dataset.index.to_numpy(), test_size=0.2, random_state=42, stratify=dataset["First_wins"])
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
        ("Random Forest", "full", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ("Extra Trees", "full", ExtraTreesClassifier(n_estimators=350, random_state=42, n_jobs=-1)),
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
            reports.append(
                {
                    "task": "battle_prediction",
                    "model": display_name,
                    "split": split_mode,
                    "feature_set": feature_key,
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "roc_auc": float(roc_auc_score(y_test, y_proba)),
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                }
            )
            fitted_models.setdefault(display_name, {})[split_mode] = {"estimator": model, "feature_columns": feature_columns}
    reports_df = pd.DataFrame(reports).sort_values(["split", "roc_auc", "accuracy"], ascending=[True, False, False])
    grouped_reports = reports_df[reports_df["split"] == "grouped"].sort_values(["roc_auc", "accuracy"], ascending=[False, False])
    best_row = grouped_reports.iloc[0]
    rf_row = grouped_reports[grouped_reports["model"] == "Random Forest"]
    if not rf_row.empty and abs(best_row["roc_auc"] - rf_row.iloc[0]["roc_auc"]) < 0.005:
        best_row = rf_row.iloc[0]
    final_model = fitted_models[best_row["model"]]["grouped"]
    return reports_df.reset_index(drop=True), {"final_model_name": best_row["model"], "final_estimator": final_model["estimator"], "final_feature_columns": final_model["feature_columns"]}


def build_battle_history_table(single_combats_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
        rows.append({"dex_a": dex_a, "dex_b": dex_b, "wins_a": win_a, "wins_b": win_b, "total_battles": 1})
    return pd.DataFrame(rows).groupby(["dex_a", "dex_b"], as_index=False)[["wins_a", "wins_b", "total_battles"]].sum().sort_values(["dex_a", "dex_b"]).reset_index(drop=True)


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
        estimator = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=180, random_state=42, n_jobs=-1))])
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
    reports_df = pd.DataFrame([grouped_report, random_report]).sort_values(["split"]).reset_index(drop=True)
    return reports_df, {"final_model_name": "Random Forest (Deploy)", "final_estimator": grouped_estimator, "final_feature_columns": full_columns, "deploy_ready": True}


def _battle_explanation(row: pd.Series, probability_a: float, probability_b: float) -> list[str]:
    explanation = []
    if row["A_best_stab"] > row["B_best_stab"]:
        explanation.append(f"Pokemon A has the stronger immediate STAB matchup ({row['A_best_stab']:.2f}x vs {row['B_best_stab']:.2f}x).")
    elif row["B_best_stab"] > row["A_best_stab"]:
        explanation.append(f"Pokemon B has the stronger immediate STAB matchup ({row['B_best_stab']:.2f}x vs {row['A_best_stab']:.2f}x).")
    else:
        explanation.append("Neither side has a clear STAB matchup edge, so base stats matter more here.")
    stat_diffs = {"HP": row["hp_diff"], "Attack": row["attack_diff"], "Defense": row["defense_diff"], "Sp. Atk": row["sp_atk_diff"], "Sp. Def": row["sp_def_diff"], "Speed": row["speed_diff"], "Total": row["total_diff"]}
    top_swings = sorted(stat_diffs.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    for stat_name, diff in top_swings:
        favored_side = "A" if diff > 0 else "B"
        explanation.append(f"{stat_name} advantage favors Pokemon {favored_side} by {abs(int(diff))} points.")
    explanation.append(f"Final model probabilities: Pokemon A {probability_a:.1%}, Pokemon B {probability_b:.1%}.")
    return explanation


def predict_battle(pokemon_a: str | int, pokemon_b: str | int, bundle: dict[str, Any] | None = None) -> PredictionResult:
    project_bundle = bundle or get_project_bundle()
    master_df = project_bundle["battle_master_df"]
    battle_df = project_bundle.get("battle_df")
    battle_history_df = project_bundle.get("battle_history_df")
    battle_bundle = project_bundle["battle_bundle"]

    def _lookup(identifier: str | int) -> pd.Series:
        if isinstance(identifier, int):
            subset = master_df[master_df["dexnum_int"] == identifier]
        else:
            subset = master_df[master_df["name"] == identifier]
        if subset.empty:
            raise ValueError(f"Pokemon '{identifier}' was not found in the battle master table.")
        return subset.iloc[0]

    row_a = _lookup(pokemon_a)
    row_b = _lookup(pokemon_b)
    single_row = build_battle_dataset(
        master_df=pd.concat([row_a.to_frame().T, row_b.to_frame().T], ignore_index=True),
        single_combats_df=pd.DataFrame([{"First_pokemon": int(row_a["dexnum_int"]), "Second_pokemon": int(row_b["dexnum_int"]), "Winner": int(row_a["dexnum_int"])}]),
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
        history_row = battle_history_df[(battle_history_df["dex_a"] == dex_a) & (battle_history_df["dex_b"] == dex_b)]
        if history_row.empty:
            total_history = wins_a = wins_b = 0
        else:
            history_row = history_row.iloc[0]
            wins_low = int(history_row["wins_a"])
            wins_high = int(history_row["wins_b"])
            total_history = int(history_row["total_battles"])
            wins_a, wins_b = (wins_low, wins_high) if int(row_a["dexnum_int"]) == dex_a else (wins_high, wins_low)
    elif battle_df is not None:
        history_ab = battle_df[(battle_df["First_pokemon"] == int(row_a["dexnum_int"])) & (battle_df["Second_pokemon"] == int(row_b["dexnum_int"]))]
        history_ba = battle_df[(battle_df["First_pokemon"] == int(row_b["dexnum_int"])) & (battle_df["Second_pokemon"] == int(row_a["dexnum_int"]))]
        total_history = len(history_ab) + len(history_ba)
        wins_a = int((history_ab["Winner"] == int(row_a["dexnum_int"])).sum() + (history_ba["Winner"] == int(row_a["dexnum_int"])).sum())
        wins_b = int((history_ab["Winner"] == int(row_b["dexnum_int"])).sum() + (history_ba["Winner"] == int(row_b["dexnum_int"])).sum())
    else:
        total_history = wins_a = wins_b = 0

    return PredictionResult(
        {
            "pokemon_a": {"name": row_a["name"], "dexnum_int": int(row_a["dexnum_int"]), "type1": row_a["type1"], "type2": row_a["type2"], "image_url": row_a.get("image_url", ""), "total": int(row_a["total"])},
            "pokemon_b": {"name": row_b["name"], "dexnum_int": int(row_b["dexnum_int"]), "type1": row_b["type1"], "type2": row_b["type2"], "image_url": row_b.get("image_url", ""), "total": int(row_b["total"])},
            "predicted_winner": predicted_winner,
            "win_prob_a": probability_a,
            "win_prob_b": probability_b,
            "history": {"total_battles": int(total_history), "wins_a": wins_a, "wins_b": wins_b, "rate_a": float(wins_a / total_history) if total_history else None, "rate_b": float(wins_b / total_history) if total_history else None},
            "model_name": battle_bundle["final_model_name"],
            "explanation": _battle_explanation(single_row, probability_a, probability_b),
            "feature_snapshot": pd.DataFrame([{"feature": "A_best_stab", "value": single_row["A_best_stab"]}, {"feature": "B_best_stab", "value": single_row["B_best_stab"]}, {"feature": "total_diff", "value": single_row["total_diff"]}, {"feature": "speed_diff", "value": single_row["speed_diff"]}, {"feature": "weak_sum_diff", "value": single_row["weak_sum_diff"]}]),
        }
    )


def train_project_bundle(root: Path | None = None, include_external: bool = False) -> dict[str, Any]:
    base = root or ROOT
    _ = include_external
    local_data = load_local_data(base)
    canonical_tables = ensure_canonical_tables(base)
    type_master_df = _sanitize_type_master(canonical_tables["pokemon_master"].copy())
    image_embedding_df = ensure_image_embeddings(type_master_df)
    type_reports, type_selection, type_bundle, type_oof_benchmark, type_oof_details = evaluate_type_models(type_master_df, image_embedding_df=image_embedding_df, run_image_experiment=True)
    type_oof_summary = build_type_oof_summary_table(type_master_df, type_bundle)

    legacy_type_reports = evaluate_legacy_type_models(clean_legacy_pokemon_table(local_data["pokemon"]))
    battle_master_df = clean_legacy_pokemon_table(local_data["pokemon"])
    battle_df = build_battle_dataset(battle_master_df, local_data["single_combats"])
    battle_reports, battle_bundle = evaluate_battle_models(battle_df)
    battle_history_df = build_battle_history_table(local_data["single_combats"])
    case_study_targets = build_case_study_candidates(type_master_df)
    case_study_results = case_study_targets.merge(
        type_oof_details[type_oof_details["model"] == type_bundle["final_model_name"]][["canonical_slug", "predicted_primary", "predicted_secondary", "matched_type_count", "result_bucket"]],
        how="left",
        on="canonical_slug",
    )

    return {
        "root": str(base),
        "bundle_source": "canonical_hybrid_training_bundle",
        "local_data_summary": summarize_local_data(local_data),
        "matchup_audit": audit_matchup_coverage(local_data),
        "type_master_df": type_master_df,
        "battle_master_df": battle_master_df,
        "battle_df": battle_df,
        "battle_history_df": battle_history_df,
        "type_reports": type_reports,
        "type_selection": type_selection,
        "type_oof_benchmark": type_oof_benchmark,
        "type_oof_summary": type_oof_summary,
        "type_oof_details": type_oof_details,
        "legacy_type_reports": legacy_type_reports,
        "battle_reports": battle_reports,
        "type_bundle": type_bundle,
        "battle_bundle": battle_bundle,
        "official_validation_report": canonical_tables["official_validation_report"].copy(),
        "pokemon_media_manifest": canonical_tables["pokemon_media"].copy(),
        "pokemon_text_corpus": canonical_tables["pokemon_text_corpus"].copy(),
        "case_study_results": case_study_results,
        "canonical_tables_dir": str(CANONICAL_DIR),
    }


@lru_cache(maxsize=1)
def get_project_bundle(include_external: bool = False) -> dict[str, Any]:
    return train_project_bundle(ROOT, include_external=include_external)


def train_deploy_bundle(root: Path | None = None, include_external: bool = False) -> dict[str, Any]:
    base = root or ROOT
    _ = include_external
    local_data = load_local_data(base)
    canonical_tables = ensure_canonical_tables(base)
    type_master_df = _sanitize_type_master(canonical_tables["pokemon_master"].copy())
    image_embedding_df = ensure_image_embeddings(type_master_df)
    type_reports, type_selection, type_bundle, type_oof_benchmark, _ = evaluate_type_models(type_master_df, image_embedding_df=image_embedding_df, run_image_experiment=True)
    type_oof_summary = build_type_oof_summary_table(type_master_df, type_bundle)
    battle_master_df = clean_legacy_pokemon_table(local_data["pokemon"])
    battle_df = build_battle_dataset(battle_master_df, local_data["single_combats"])
    battle_reports, battle_bundle = _train_deploy_battle_bundle(battle_df)
    battle_history_df = build_battle_history_table(local_data["single_combats"])

    return {
        "root": str(base),
        "bundle_source": "deploy_artifact",
        "type_master_df": type_master_df,
        "battle_master_df": battle_master_df,
        "battle_history_df": battle_history_df,
        "type_reports": type_reports,
        "type_selection": type_selection,
        "type_oof_benchmark": type_oof_benchmark,
        "type_oof_summary": type_oof_summary,
        "battle_reports": battle_reports,
        "type_bundle": type_bundle,
        "battle_bundle": battle_bundle,
        "matchup_audit": audit_matchup_coverage(local_data),
        "official_validation_report": canonical_tables["official_validation_report"].copy(),
        "pokemon_media_manifest": canonical_tables["pokemon_media"].copy(),
        "pokemon_text_corpus": canonical_tables["pokemon_text_corpus"].copy(),
        "canonical_tables_dir": str(CANONICAL_DIR),
    }


def save_deploy_bundle(bundle: dict[str, Any], path: Path | None = None, compress: int = 3) -> Path:
    target = path or DEPLOY_BUNDLE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=compress)
    return target


def load_deploy_bundle(path: Path | None = None) -> dict[str, Any]:
    return joblib.load(path or DEPLOY_BUNDLE_PATH)


@lru_cache(maxsize=1)
def get_deploy_bundle(path: Path | None = None) -> dict[str, Any]:
    target = path or DEPLOY_BUNDLE_PATH
    if target.exists():
        return load_deploy_bundle(target)
    bundle = train_deploy_bundle(ROOT, include_external=False)
    save_deploy_bundle(bundle, target)
    return bundle
