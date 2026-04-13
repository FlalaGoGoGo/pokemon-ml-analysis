from __future__ import annotations

import json
import re
import ssl
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
CANONICAL_DIR = ARTIFACTS_DIR / "canonical"
RAW_CACHE_DIR = CANONICAL_DIR / "raw_cache"
IMAGE_EMBEDDINGS_PATH = ARTIFACTS_DIR / "canonical_image_embeddings.joblib"

CANONICAL_TABLE_PATHS = {
    "pokemon": CANONICAL_DIR / "pokemon.csv",
    "pokemon_species": CANONICAL_DIR / "pokemon_species.csv",
    "pokemon_forms": CANONICAL_DIR / "pokemon_forms.csv",
    "pokemon_types": CANONICAL_DIR / "pokemon_types.csv",
    "pokemon_abilities": CANONICAL_DIR / "pokemon_abilities.csv",
    "pokemon_moves": CANONICAL_DIR / "pokemon_moves.csv",
    "pokemon_flavor_texts": CANONICAL_DIR / "pokemon_flavor_texts.csv",
    "pokemon_evolution_edges": CANONICAL_DIR / "pokemon_evolution_edges.csv",
    "pokemon_media": CANONICAL_DIR / "pokemon_media_manifest.csv",
    "pokemon_text_corpus": CANONICAL_DIR / "pokemon_text_corpus.csv",
    "official_validation_report": CANONICAL_DIR / "official_validation_report.csv",
    "move_details": CANONICAL_DIR / "move_details.csv",
    "ability_details": CANONICAL_DIR / "ability_details.csv",
    "pokemon_master": CANONICAL_DIR / "pokemon_master.csv",
}

RESOURCE_CACHE_PATHS = {
    "pokemon_index": RAW_CACHE_DIR / "pokemon_index.json",
    "pokemon": RAW_CACHE_DIR / "pokemon_payloads.json",
    "species": RAW_CACHE_DIR / "species_payloads.json",
    "forms": RAW_CACHE_DIR / "form_payloads.json",
    "abilities": RAW_CACHE_DIR / "ability_payloads.json",
    "moves": RAW_CACHE_DIR / "move_payloads.json",
    "evolution": RAW_CACHE_DIR / "evolution_payloads.json",
}

REGION_TAGS = {
    "alola": "Alola",
    "galar": "Galar",
    "hisui": "Hisui",
    "paldea": "Paldea",
}

FORM_GROUP_TOKENS = {
    "mega": "Mega",
    "gmax": "Gigantamax",
    "totem": "Totem",
    "origin": "Origin",
    "therian": "Therian",
    "incarnate": "Incarnate",
    "attack": "Attack",
    "defense": "Defense",
    "speed": "Speed",
    "school": "School",
    "sunshine": "Sunshine",
    "dusk": "Dusk",
    "dawn": "Dawn",
    "midday": "Midday",
    "midnight": "Midnight",
    "crowned": "Crowned",
    "eternamax": "Eternamax",
    "bloodmoon": "Bloodmoon",
    "cornerstone": "Cornerstone",
    "hearthflame": "Hearthflame",
    "wellspring": "Wellspring",
    "teal-mask": "Teal Mask",
}


def _safe_json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def _safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def _slug_to_title(slug: str) -> str:
    text = slug.replace("-f", " female").replace("-m", " male")
    parts = [part for part in re.split(r"[-_]+", text) if part]
    return " ".join(part.upper() if len(part) == 1 else part.capitalize() for part in parts)


def _english_name(entries: list[dict[str, Any]], fallback: str) -> str:
    for entry in entries:
        if entry.get("language", {}).get("name") == "en":
            return str(entry.get("name", fallback))
    return fallback


def _clean_flavor_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ").replace("\f", " ")).strip()


def _generation_number(generation_name: str) -> int:
    match = re.search(r"generation-(.+)$", generation_name or "")
    if not match:
        return 0
    roman = match.group(1).upper()
    roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9}
    return roman_map.get(roman, 0)


def _gender_ratio_to_pct(gender_rate: int | float | None) -> tuple[float | None, float | None]:
    if gender_rate is None or gender_rate == -1:
        return None, None
    female_pct = float(gender_rate) / 8.0 * 100.0
    male_pct = 100.0 - female_pct
    return male_pct, female_pct


def _region_tag(canonical_slug: str) -> str:
    for token, label in REGION_TAGS.items():
        if f"-{token}" in canonical_slug or canonical_slug.endswith(token):
            return label
    return "None"


def _form_group(canonical_slug: str, is_mega: bool, is_battle_only: bool) -> str:
    if is_mega:
        return "Mega"
    if is_battle_only:
        return "Battle-Only"
    for token, label in FORM_GROUP_TOKENS.items():
        if f"-{token}" in canonical_slug or canonical_slug.endswith(token):
            return label
    return "Standard"


def _special_group(row: dict[str, Any]) -> str:
    if row.get("is_mythical"):
        return "Mythical"
    if row.get("is_legendary"):
        return "Legendary"
    if row.get("is_baby"):
        return "Baby"
    if row.get("is_mega"):
        return "Mega"
    if row.get("is_battle_only"):
        return "Battle-Only"
    if row.get("region_tag") != "None":
        return f"Regional-{row['region_tag']}"
    return "Ordinary"


def _official_species_pokedex_url(species_slug: str) -> str:
    return f"https://www.pokemon.com/us/pokedex/{species_slug}"


def _official_artwork_url(dexnum_int: int) -> str:
    return f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{int(dexnum_int):03d}.png"


def _requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "PokemonMLAnalysis/1.0"})
    return session


def _fetch_json(session: requests.Session, url: str, timeout: int = 30, retries: int = 5) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2.0 * attempt, 10.0))
    if last_error is None:  # pragma: no cover
        raise RuntimeError(f"Failed to fetch {url}")
    raise last_error


def _fetch_paginated_index(session: requests.Session, url: str, cache_path: Path, force_refresh: bool = False) -> list[dict[str, Any]]:
    cache = _safe_json_load(cache_path)
    if cache and not force_refresh:
        return list(cache.get("results", []))

    results: list[dict[str, Any]] = []
    next_url = url
    while next_url:
        payload = _fetch_json(session, next_url)
        results.extend(payload.get("results", []))
        next_url = payload.get("next")
    _safe_json_dump(cache_path, {"results": results})
    return results


def _fetch_resource_payloads(
    session: requests.Session,
    items: list[dict[str, Any]],
    cache_path: Path,
    force_refresh: bool = False,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    cache = _safe_json_load(cache_path)
    starting_count = len(cache)
    dump_every = 250 if len(items) >= 1000 else 100 if len(items) >= 300 else 25
    changed = False
    for i, item in enumerate(items, start=1):
        name = str(item["name"])
        if force_refresh or name not in cache:
            cache[name] = _fetch_json(session, str(item["url"]))
            changed = True
            if sleep_seconds:
                time.sleep(sleep_seconds)
        if changed and i % dump_every == 0:
            _safe_json_dump(cache_path, cache)
    if changed or starting_count != len(cache):
        _safe_json_dump(cache_path, cache)
    return cache


def _chain_nodes(chain: dict[str, Any], stage: int = 0) -> list[tuple[dict[str, Any], int]]:
    nodes = [(chain, stage)]
    for child in chain.get("evolves_to", []):
        nodes.extend(_chain_nodes(child, stage + 1))
    return nodes


def _build_evolution_edges(evolution_payloads: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    edge_rows: list[dict[str, Any]] = []
    stage_map: dict[str, int] = {}
    for chain_name, payload in evolution_payloads.items():
        chain_id = int(payload["id"])
        for node, stage in _chain_nodes(payload["chain"], stage=0):
            species_name = node["species"]["name"]
            stage_map[species_name] = min(stage_map.get(species_name, stage), stage)
            for child in node.get("evolves_to", []):
                detail = (child.get("evolution_details") or [{}])[0]
                edge_rows.append(
                    {
                        "evolution_chain_id": chain_id,
                        "from_species_slug": species_name,
                        "to_species_slug": child["species"]["name"],
                        "from_stage": stage,
                        "to_stage": stage + 1,
                        "trigger": (detail.get("trigger") or {}).get("name", "unknown"),
                        "min_level": detail.get("min_level"),
                        "item": (detail.get("item") or {}).get("name"),
                        "held_item": (detail.get("held_item") or {}).get("name"),
                        "known_move": (detail.get("known_move") or {}).get("name"),
                        "time_of_day": detail.get("time_of_day"),
                    }
                )
    return pd.DataFrame(edge_rows), stage_map


def sync_canonical_pokemon_data(
    root: Path | None = None,
    force_refresh: bool = False,
    validate_media: bool = True,
    sleep_seconds: float = 0.0,
) -> dict[str, pd.DataFrame]:
    base = root or ROOT
    _ = base
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = _requests_session()

    pokemon_index = _fetch_paginated_index(
        session,
        "https://pokeapi.co/api/v2/pokemon?limit=2000",
        RESOURCE_CACHE_PATHS["pokemon_index"],
        force_refresh=force_refresh,
    )
    print(f"[canonical] pokemon index rows: {len(pokemon_index)}")
    pokemon_payloads = _fetch_resource_payloads(
        session,
        pokemon_index,
        RESOURCE_CACHE_PATHS["pokemon"],
        force_refresh=force_refresh,
        sleep_seconds=sleep_seconds,
    )
    print(f"[canonical] pokemon payloads: {len(pokemon_payloads)}")

    species_items = []
    ability_items = {}
    move_items = {}
    for payload in pokemon_payloads.values():
        species_items.append(payload["species"])
        for ability in payload.get("abilities", []):
            ability_items[ability["ability"]["name"]] = ability["ability"]
        for move in payload.get("moves", []):
            move_items[move["move"]["name"]] = move["move"]

    species_items = sorted({item["name"]: item for item in species_items}.values(), key=lambda item: item["name"])
    ability_items_list = sorted(ability_items.values(), key=lambda item: item["name"])
    move_items_list = sorted(move_items.values(), key=lambda item: item["name"])

    species_payloads = _fetch_resource_payloads(
        session,
        species_items,
        RESOURCE_CACHE_PATHS["species"],
        force_refresh=force_refresh,
        sleep_seconds=sleep_seconds,
    )
    print(f"[canonical] species payloads: {len(species_payloads)}")
    form_payloads: dict[str, Any] = {}
    print("[canonical] form payloads: derived from pokemon endpoint")
    ability_payloads = _fetch_resource_payloads(
        session,
        ability_items_list,
        RESOURCE_CACHE_PATHS["abilities"],
        force_refresh=force_refresh,
        sleep_seconds=sleep_seconds,
    )
    print(f"[canonical] ability payloads: {len(ability_payloads)}")
    move_payloads = _fetch_resource_payloads(
        session,
        move_items_list,
        RESOURCE_CACHE_PATHS["moves"],
        force_refresh=force_refresh,
        sleep_seconds=sleep_seconds,
    )
    print(f"[canonical] move payloads: {len(move_payloads)}")

    evolution_items = []
    for payload in species_payloads.values():
        evolution_url = payload.get("evolution_chain", {}).get("url")
        if not evolution_url:
            continue
        chain_id = evolution_url.rstrip("/").split("/")[-1]
        evolution_items.append({"name": str(chain_id), "url": evolution_url})
    evolution_items = sorted({item["name"]: item for item in evolution_items}.values(), key=lambda item: int(item["name"]))
    evolution_payloads = _fetch_resource_payloads(
        session,
        evolution_items,
        RESOURCE_CACHE_PATHS["evolution"],
        force_refresh=force_refresh,
        sleep_seconds=sleep_seconds,
    )
    print(f"[canonical] evolution payloads: {len(evolution_payloads)}")

    evolution_edges_df, stage_map = _build_evolution_edges(evolution_payloads)

    ability_detail_rows = []
    for payload in ability_payloads.values():
        short_effect = ""
        for effect in payload.get("effect_entries", []):
            if effect.get("language", {}).get("name") == "en":
                short_effect = effect.get("short_effect") or effect.get("effect") or ""
                break
        ability_detail_rows.append(
            {
                "ability_name": payload["name"],
                "ability_display_name": _english_name(payload.get("names", []), _slug_to_title(payload["name"])),
                "generation": payload.get("generation", {}).get("name", "unknown"),
                "short_effect_en": _clean_flavor_text(short_effect),
            }
        )
    ability_details_df = pd.DataFrame(ability_detail_rows).sort_values("ability_name").reset_index(drop=True)

    move_detail_rows = []
    for payload in move_payloads.values():
        move_detail_rows.append(
            {
                "move_name": payload["name"],
                "move_display_name": _english_name(payload.get("names", []), _slug_to_title(payload["name"])),
                "move_type": _slug_to_title(payload.get("type", {}).get("name", "unknown")),
                "damage_class": payload.get("damage_class", {}).get("name", "unknown"),
                "power": payload.get("power"),
                "accuracy": payload.get("accuracy"),
                "pp": payload.get("pp"),
            }
        )
    move_details_df = pd.DataFrame(move_detail_rows).sort_values("move_name").reset_index(drop=True)

    pokemon_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    ability_rows: list[dict[str, Any]] = []
    move_rows: list[dict[str, Any]] = []
    form_rows: list[dict[str, Any]] = []
    species_rows: list[dict[str, Any]] = []
    flavor_rows: list[dict[str, Any]] = []
    media_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []

    species_seen: set[str] = set()
    validated_at = pd.Timestamp.now(tz="America/Los_Angeles").isoformat()

    unique_artwork_urls: dict[str, int] = {}
    artwork_status_cache: dict[str, int] = {}

    for pokemon_name, payload in pokemon_payloads.items():
        species_slug = payload["species"]["name"]
        species_payload = species_payloads[species_slug]
        form_payload = form_payloads.get(pokemon_name, {})
        dexnum_int = int(species_payload["id"])
        pokemon_id = int(payload["id"])
        canonical_slug = pokemon_name
        display_name = _slug_to_title(canonical_slug)
        form_suffix = canonical_slug.removeprefix(species_slug).strip("-")
        form_display_name = _slug_to_title(form_suffix) if form_suffix else "Default Form"
        region_tag = _region_tag(canonical_slug)
        is_mega = "-mega" in canonical_slug
        is_battle_only = int(any(token in canonical_slug for token in ["-totem", "-school", "-busted", "-disguised", "-hangry", "-gulping", "-gorging", "-eternamax"]))
        form_group = _form_group(canonical_slug, is_mega=is_mega, is_battle_only=is_battle_only)

        stats = {entry["stat"]["name"]: int(entry["base_stat"]) for entry in payload.get("stats", [])}
        type_names = {
            int(entry["slot"]): _slug_to_title(entry["type"]["name"])
            for entry in payload.get("types", [])
        }
        type1 = type_names.get(1, "Unknown")
        type2 = type_names.get(2, "None")

        growth_rate = (species_payload.get("growth_rate") or {}).get("name", "unknown")
        egg_groups = species_payload.get("egg_groups", [])
        egg_group1 = _slug_to_title(egg_groups[0]["name"]) if egg_groups else "Unknown"
        egg_group2 = _slug_to_title(egg_groups[1]["name"]) if len(egg_groups) > 1 else "Unknown"
        male_pct, female_pct = _gender_ratio_to_pct(species_payload.get("gender_rate"))
        evolution_chain_url = species_payload.get("evolution_chain", {}).get("url", "")
        evolution_chain_id = int(evolution_chain_url.rstrip("/").split("/")[-1]) if evolution_chain_url else 0
        flavor_texts = [
            _clean_flavor_text(entry.get("flavor_text", ""))
            for entry in species_payload.get("flavor_text_entries", [])
            if entry.get("language", {}).get("name") == "en"
        ]
        flavor_texts = list(dict.fromkeys(text for text in flavor_texts if text))
        flavor_text_primary = flavor_texts[0] if flavor_texts else ""
        flavor_text_corpus = " ".join(flavor_texts[:8])
        genus_en = ""
        for genus_entry in species_payload.get("genera", []):
            if genus_entry.get("language", {}).get("name") == "en":
                genus_en = genus_entry.get("genus", "")
                break

        ability_names: dict[str, str] = {"ability1": "Unknown", "ability2": "Unknown", "hidden_ability": "Unknown"}
        ability_summaries = []
        non_hidden_abilities = [
            entry for entry in sorted(payload.get("abilities", []), key=lambda item: item["slot"]) if not entry.get("is_hidden")
        ]
        if non_hidden_abilities:
            ability_names["ability1"] = _slug_to_title(non_hidden_abilities[0]["ability"]["name"])
        if len(non_hidden_abilities) > 1:
            ability_names["ability2"] = _slug_to_title(non_hidden_abilities[1]["ability"]["name"])
        hidden_entries = [entry for entry in payload.get("abilities", []) if entry.get("is_hidden")]
        if hidden_entries:
            ability_names["hidden_ability"] = _slug_to_title(hidden_entries[0]["ability"]["name"])
        for key in ["ability1", "ability2", "hidden_ability"]:
            raw_name = ability_names[key]
            if raw_name == "Unknown":
                continue
            ability_lookup = ability_details_df[ability_details_df["ability_display_name"] == raw_name]
            if ability_lookup.empty:
                ability_lookup = ability_details_df[
                    ability_details_df["ability_name"] == raw_name.lower().replace(" ", "-")
                ]
            if not ability_lookup.empty:
                ability_summaries.append(ability_lookup.iloc[0]["short_effect_en"])

        official_artwork_url = _official_artwork_url(dexnum_int)
        variant_artwork_url = (
            payload.get("sprites", {})
            .get("other", {})
            .get("official-artwork", {})
            .get("front_default")
            or ""
        )
        sprite_url = (
            payload.get("sprites", {})
            .get("other", {})
            .get("home", {})
            .get("front_default")
            or payload.get("sprites", {}).get("front_default")
            or variant_artwork_url
            or official_artwork_url
        )
        official_pokedex_url = _official_species_pokedex_url(species_slug)
        unique_artwork_urls.setdefault(official_artwork_url, dexnum_int)

        pokemon_rows.append(
            {
                "pokemon_id": pokemon_id,
                "pokemon_name": pokemon_name,
                "canonical_slug": canonical_slug,
                "display_name": display_name,
                "species_slug": species_slug,
                "dexnum_int": dexnum_int,
                "order": int(payload.get("order", pokemon_id)),
                "is_default": int(payload.get("is_default", False)),
                "base_experience": payload.get("base_experience"),
                "height": payload.get("height"),
                "weight": payload.get("weight"),
                "type1": type1,
                "type2": type2,
                "hp": stats.get("hp"),
                "attack": stats.get("attack"),
                "defense": stats.get("defense"),
                "sp_atk": stats.get("special-attack"),
                "sp_def": stats.get("special-defense"),
                "speed": stats.get("speed"),
                "total": sum(value for value in stats.values() if value is not None),
                "generation": _generation_number((species_payload.get("generation") or {}).get("name", "")),
                "capture_rate": species_payload.get("capture_rate"),
                "base_happiness": species_payload.get("base_happiness"),
                "base_exp": payload.get("base_experience"),
                "growth_rate": _slug_to_title(growth_rate),
                "egg_group1": egg_group1,
                "egg_group2": egg_group2,
                "percent_male": male_pct,
                "percent_female": female_pct,
                "egg_cycles": species_payload.get("hatch_counter"),
                "color": _slug_to_title((species_payload.get("color") or {}).get("name", "unknown")),
                "shape": _slug_to_title((species_payload.get("shape") or {}).get("name", "unknown")),
                "habitat": _slug_to_title((species_payload.get("habitat") or {}).get("name", "unknown")),
                "is_baby": int(species_payload.get("is_baby", False)),
                "is_legendary": int(species_payload.get("is_legendary", False)),
                "is_mythical": int(species_payload.get("is_mythical", False)),
                "region_tag": region_tag,
                "form_group": form_group,
                "is_mega": int(is_mega),
                "is_battle_only": int(is_battle_only),
                "evolution_chain_id": evolution_chain_id,
                "evolution_stage": int(stage_map.get(species_slug, 0)),
                "genus_en": genus_en,
                "flavor_text_en": flavor_text_primary,
                "flavor_text_corpus_en": flavor_text_corpus,
                "ability1": ability_names["ability1"],
                "ability2": ability_names["ability2"],
                "hidden_ability": ability_names["hidden_ability"],
                "ability_summary_en": " ".join(text for text in ability_summaries if text),
                "official_artwork_url": official_artwork_url,
                "artwork_variant_url": variant_artwork_url,
                "sprite_url": sprite_url,
                "image_url": variant_artwork_url or official_artwork_url or sprite_url,
                "official_pokedex_url": official_pokedex_url,
            }
        )

        for slot, type_name in sorted(type_names.items()):
            type_rows.append(
                {
                    "canonical_slug": canonical_slug,
                    "pokemon_id": pokemon_id,
                    "slot": int(slot),
                    "type_name": type_name,
                }
            )
        for ability in payload.get("abilities", []):
            ability_rows.append(
                {
                    "canonical_slug": canonical_slug,
                    "pokemon_id": pokemon_id,
                    "slot": int(ability["slot"]),
                    "ability_name": ability["ability"]["name"],
                    "ability_display_name": _slug_to_title(ability["ability"]["name"]),
                    "is_hidden": int(ability.get("is_hidden", False)),
                }
            )
        for move in payload.get("moves", []):
            details = move_payloads.get(move["move"]["name"], {})
            move_rows.append(
                {
                    "canonical_slug": canonical_slug,
                    "pokemon_id": pokemon_id,
                    "move_name": move["move"]["name"],
                    "move_display_name": _slug_to_title(move["move"]["name"]),
                    "move_type": _slug_to_title(details.get("type", {}).get("name", "unknown")),
                    "damage_class": details.get("damage_class", {}).get("name", "unknown"),
                    "power": details.get("power"),
                    "accuracy": details.get("accuracy"),
                    "pp": details.get("pp"),
                }
            )

        form_rows.append(
            {
                "canonical_slug": canonical_slug,
                "pokemon_id": pokemon_id,
                "species_slug": species_slug,
                "display_name": display_name,
                "form_display_name": form_display_name,
                "is_default": int(payload.get("is_default", False)),
                "is_mega": int(is_mega),
                "is_battle_only": int(is_battle_only),
                "form_order": payload.get("order"),
                "region_tag": region_tag,
                "form_group": form_group,
            }
        )

        if species_slug not in species_seen:
            species_seen.add(species_slug)
            species_rows.append(
                {
                    "species_slug": species_slug,
                    "species_display_name": _english_name(species_payload.get("names", []), _slug_to_title(species_slug)),
                    "dexnum_int": dexnum_int,
                    "generation": _generation_number((species_payload.get("generation") or {}).get("name", "")),
                    "gender_rate": species_payload.get("gender_rate"),
                    "capture_rate": species_payload.get("capture_rate"),
                    "base_happiness": species_payload.get("base_happiness"),
                    "hatch_counter": species_payload.get("hatch_counter"),
                    "is_baby": int(species_payload.get("is_baby", False)),
                    "is_legendary": int(species_payload.get("is_legendary", False)),
                    "is_mythical": int(species_payload.get("is_mythical", False)),
                    "color": _slug_to_title((species_payload.get("color") or {}).get("name", "unknown")),
                    "shape": _slug_to_title((species_payload.get("shape") or {}).get("name", "unknown")),
                    "habitat": _slug_to_title((species_payload.get("habitat") or {}).get("name", "unknown")),
                    "generation_name": (species_payload.get("generation") or {}).get("name", "unknown"),
                    "growth_rate": _slug_to_title((species_payload.get("growth_rate") or {}).get("name", "unknown")),
                    "egg_group1": egg_group1,
                    "egg_group2": egg_group2,
                    "percent_male": male_pct,
                    "percent_female": female_pct,
                    "evolution_chain_id": evolution_chain_id,
                    "genus_en": genus_en,
                }
            )
            for index, text in enumerate(flavor_texts, start=1):
                flavor_rows.append(
                    {
                        "species_slug": species_slug,
                        "dexnum_int": dexnum_int,
                        "entry_order": index,
                        "flavor_text_en": text,
                    }
                )

        text_rows.append(
            {
                "canonical_slug": canonical_slug,
                "display_name": display_name,
                "species_slug": species_slug,
                "text_corpus_en": " ".join(
                    part
                    for part in [
                        display_name,
                        _slug_to_title(species_slug),
                        genus_en,
                        flavor_text_corpus,
                        " ".join(text for text in ability_summaries if text),
                        form_display_name if form_display_name != "Default Form" else "",
                    ]
                    if part
                ),
            }
        )
        media_rows.append(
            {
                "canonical_slug": canonical_slug,
                "display_name": display_name,
                "dexnum_int": dexnum_int,
                "official_artwork_url": official_artwork_url,
                "artwork_variant_url": variant_artwork_url,
                "sprite_url": sprite_url,
                "image_url": variant_artwork_url or official_artwork_url or sprite_url,
                "official_pokedex_url": official_pokedex_url,
                "official_reference_scope": "species-page",
                "validated_at": validated_at,
            }
        )

    if validate_media:
        for official_url in unique_artwork_urls:
            try:
                response = session.head(official_url, timeout=30, allow_redirects=True)
                artwork_status_cache[official_url] = int(response.status_code)
            except requests.RequestException:
                artwork_status_cache[official_url] = 0

    pokemon_df = pd.DataFrame(pokemon_rows).sort_values(["dexnum_int", "order", "canonical_slug"]).reset_index(drop=True)
    species_df = pd.DataFrame(species_rows).sort_values(["dexnum_int", "species_slug"]).reset_index(drop=True)
    forms_df = pd.DataFrame(form_rows).sort_values(["pokemon_id"]).reset_index(drop=True)
    types_df = pd.DataFrame(type_rows).sort_values(["pokemon_id", "slot"]).reset_index(drop=True)
    abilities_df = pd.DataFrame(ability_rows).sort_values(["pokemon_id", "slot"]).reset_index(drop=True)
    moves_df = pd.DataFrame(move_rows).sort_values(["pokemon_id", "move_name"]).reset_index(drop=True)
    flavors_df = pd.DataFrame(flavor_rows).sort_values(["dexnum_int", "entry_order"]).reset_index(drop=True)
    media_df = pd.DataFrame(media_rows).sort_values(["dexnum_int", "canonical_slug"]).reset_index(drop=True)
    media_df["official_artwork_http_status"] = media_df["official_artwork_url"].map(artwork_status_cache).fillna(0).astype(int)
    media_df["validation_status"] = np.where(
        media_df["official_artwork_http_status"].eq(200),
        "official_media_verified_link_generated",
        "manual_review_required",
    )
    media_df["mismatch_fields"] = np.where(media_df["official_artwork_http_status"].eq(200), "", "official_artwork_url")
    media_df["validation_notes"] = (
        "pokemon.com page content is bot-protected; official page URL is attached for manual verification."
    )

    text_df = pd.DataFrame(text_rows).sort_values(["canonical_slug"]).reset_index(drop=True)
    validation_df = media_df[
        [
            "canonical_slug",
            "display_name",
            "dexnum_int",
            "official_pokedex_url",
            "official_artwork_url",
            "official_artwork_http_status",
            "validation_status",
            "validated_at",
            "mismatch_fields",
            "validation_notes",
        ]
    ].copy()

    move_type_counts = (
        moves_df.groupby(["canonical_slug", "move_type"]).size().unstack(fill_value=0).add_prefix("move_type_count_")
    )
    move_class_counts = (
        moves_df.groupby(["canonical_slug", "damage_class"]).size().unstack(fill_value=0).add_prefix("move_class_count_")
    )
    move_agg_df = pd.concat([move_type_counts, move_class_counts], axis=1).reset_index()
    move_agg_df["move_count"] = (
        moves_df.groupby("canonical_slug").size().reindex(move_agg_df["canonical_slug"]).fillna(0).to_numpy().astype(int)
    )

    master_df = (
        pokemon_df.merge(species_df[["species_slug", "species_display_name"]], how="left", on="species_slug")
        .merge(forms_df[["canonical_slug", "form_display_name"]], how="left", on="canonical_slug")
        .merge(text_df, how="left", on=["canonical_slug", "display_name", "species_slug"])
        .merge(media_df[["canonical_slug", "validation_status", "validated_at", "mismatch_fields"]], how="left", on="canonical_slug")
        .merge(move_agg_df, how="left", on="canonical_slug")
    )
    move_count_columns = [column for column in master_df.columns if column.startswith("move_type_count_") or column.startswith("move_class_count_")]
    for column in move_count_columns + ["move_count"]:
        if column in master_df.columns:
            master_df[column] = master_df[column].fillna(0).astype(float)
    master_df["single_type_flag"] = master_df["type2"].eq("None").astype(int)
    master_df["bulk_score"] = master_df["hp"] + master_df["defense"] + master_df["sp_def"]
    master_df["offense_score"] = master_df["attack"] + master_df["sp_atk"] + master_df["speed"]
    master_df["physical_bias"] = master_df["attack"] - master_df["sp_atk"]
    master_df["special_bias"] = master_df["sp_atk"] - master_df["attack"]
    master_df["speed_rank_pct"] = master_df["speed"].rank(pct=True)
    master_df["special_group"] = master_df.apply(lambda row: _special_group(row.to_dict()), axis=1)
    master_df["pokemon_api_ability_count"] = (
        abilities_df.groupby("canonical_slug").size().reindex(master_df["canonical_slug"]).fillna(0).to_numpy().astype(int)
    )
    master_df["base_stat_total_check"] = (
        master_df[["hp", "attack", "defense", "sp_atk", "sp_def", "speed"]].sum(axis=1).astype(int)
    )
    master_df["total_matches_sum"] = master_df["base_stat_total_check"].eq(master_df["total"]).astype(int)
    master_df = master_df.sort_values(["dexnum_int", "order", "canonical_slug"]).reset_index(drop=True)

    tables = {
        "pokemon": pokemon_df,
        "pokemon_species": species_df,
        "pokemon_forms": forms_df,
        "pokemon_types": types_df,
        "pokemon_abilities": abilities_df,
        "pokemon_moves": moves_df,
        "pokemon_flavor_texts": flavors_df,
        "pokemon_evolution_edges": evolution_edges_df.sort_values(["evolution_chain_id", "from_stage", "to_stage"]).reset_index(drop=True),
        "pokemon_media": media_df,
        "pokemon_text_corpus": text_df,
        "official_validation_report": validation_df,
        "move_details": move_details_df,
        "ability_details": ability_details_df,
        "pokemon_master": master_df,
    }
    for key, df in tables.items():
        CANONICAL_TABLE_PATHS[key].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CANONICAL_TABLE_PATHS[key], index=False)
    print(f"[canonical] normalized tables written to {CANONICAL_DIR}")
    return tables


def load_canonical_tables(root: Path | None = None) -> dict[str, pd.DataFrame]:
    base = root or ROOT
    _ = base
    if not all(path.exists() for path in CANONICAL_TABLE_PATHS.values()):
        return sync_canonical_pokemon_data(base, force_refresh=False, validate_media=True)
    return {name: pd.read_csv(path) for name, path in CANONICAL_TABLE_PATHS.items()}


def ensure_canonical_tables(root: Path | None = None, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    if force_refresh or not all(path.exists() for path in CANONICAL_TABLE_PATHS.values()):
        return sync_canonical_pokemon_data(root, force_refresh=force_refresh, validate_media=True)
    return load_canonical_tables(root)


def build_case_study_candidates(master_df: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "charizard",
        "gyarados",
        "exeggutor-alola",
        "lugia",
        "goodra-hisui",
    }
    selected = master_df[master_df["canonical_slug"].isin(targets)].copy()
    return selected[
        [
            "canonical_slug",
            "display_name",
            "type1",
            "type2",
            "species_display_name",
            "official_pokedex_url",
            "validation_status",
        ]
    ].reset_index(drop=True)


def _prepare_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=30, headers={"User-Agent": "PokemonMLAnalysis/1.0"})
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def ensure_image_embeddings(
    master_df: pd.DataFrame,
    path: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    target = path or IMAGE_EMBEDDINGS_PATH
    expected_slugs = list(master_df["canonical_slug"])
    if target.exists() and not force_refresh:
        payload = joblib.load(target)
        if isinstance(payload, pd.DataFrame) and set(expected_slugs).issubset(set(payload["canonical_slug"])):
            return payload.set_index("canonical_slug").loc[expected_slugs].reset_index()

    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Image embeddings require torch + torchvision.") from exc

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    model = resnet18(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    rows = []
    with torch.no_grad():
        for idx, row in enumerate(master_df.itertuples(index=False), start=1):
            url = getattr(row, "artwork_variant_url", "") or getattr(row, "official_artwork_url", "") or getattr(row, "sprite_url", "")
            vector = np.zeros(512, dtype=np.float32)
            if url:
                try:
                    image = _prepare_image(url)
                    tensor = preprocess(image).unsqueeze(0)
                    vector = feature_extractor(tensor).flatten().cpu().numpy().astype(np.float32)
                except Exception:
                    vector = np.zeros(512, dtype=np.float32)
            rows.append({"canonical_slug": row.canonical_slug, **{f"img_{i:03d}": float(v) for i, v in enumerate(vector)}})
            if idx % 100 == 0:
                print(f"[canonical] image embeddings processed: {idx}/{len(master_df)}")
    embedding_df = pd.DataFrame(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(embedding_df, target, compress=3)
    return embedding_df
