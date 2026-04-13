from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "Pokemon_Project_Rebuild_Streamlit.ipynb"


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Pokemon Canonical Type Analysis

This notebook rebuilds the type-prediction project around a canonical hybrid dataset:

- `PokeAPI` provides the structured backbone.
- `pokemon.com` official references are attached as URLs and validation metadata.
- Kaggle tables are retained only as historical baseline inputs.

The modeling workflow has three tracks:

1. Historical legacy baseline
2. Clean structured benchmark
3. Structured + text mainline and multimodal image experiment
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

from canonical_pokemon import CANONICAL_TABLE_PATHS, ensure_canonical_tables
from pokemon_project import notebook_ready_tables, train_project_bundle

ROOT = Path.cwd()
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 120)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Part 1. Why Replace Kaggle

The old project depended on third-party CSV snapshots. That created two problems:

- Some rows were incorrect or stale.
- Many rich fields were missing, especially text, form metadata, provenance, and media links.

This rebuild treats Kaggle as a historical baseline only. The canonical dataset is synchronized from `PokeAPI`, then enriched with official reference URLs and validation metadata.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """canonical_tables = ensure_canonical_tables(ROOT, force_refresh=False)
bundle = train_project_bundle(ROOT, include_external=False)

display(bundle["local_data_summary"])
display(bundle["matchup_audit"])
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Part 2. Canonical Dataset Inventory

The project now produces normalized entity tables and a denormalized `pokemon_master` table.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """inventory_rows = []
for name, path in CANONICAL_TABLE_PATHS.items():
    df = canonical_tables[name]
    inventory_rows.append({"table": name, "rows": len(df), "columns": len(df.columns), "path": str(path)})
inventory_df = pd.DataFrame(inventory_rows).sort_values("table").reset_index(drop=True)
display(inventory_df)

display(canonical_tables["pokemon_master"].head(8))
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Part 3. Validation and Provenance

Because `pokemon.com` blocks large-scale automated page scraping, this pipeline uses a conservative official-reference strategy:

- attach official Pokédex URLs
- verify official artwork URLs where possible
- record validation status and manual-review notes
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """validation_df = bundle["official_validation_report"].copy()
coverage = pd.DataFrame(
    [
        {
            "rows": len(validation_df),
            "official_artwork_verified_pct": validation_df["official_artwork_http_status"].eq(200).mean(),
            "manual_review_required_pct": validation_df["validation_status"].eq("manual_review_required").mean(),
        }
    ]
)
display(coverage)
display(validation_df.head(10))
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Part 4. Historical Baseline vs Canonical Benchmark

The next table compares the old CSV-style baseline against the new canonical pipeline.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """display(bundle["legacy_type_reports"])
display(bundle["type_reports"])
display(bundle["type_selection"])
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Part 5. Out-of-Fold Benchmark Summary

Formal reporting should come from evolution-group OOF, not from training-set predictions.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """display(bundle["type_oof_summary"])

evolution_oof = bundle["type_oof_summary"].query("split_mode == 'evolution_group'").copy()
display(evolution_oof)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Part 6. Random Split vs Species Group vs Evolution Group

- `random split` is historically comparable but optimistic.
- `species group` is stricter because it prevents same-species leakage.
- `evolution group OOF` is the main benchmark because it also blocks same-family leakage.

This is not LLM-style hallucination. When the model is wrong here, it is a generalization error under a stricter holdout definition.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """oof_summary = bundle["type_oof_summary"].copy()
comparison = oof_summary[[
    "split_mode",
    "all_types_correct_n",
    "all_types_correct_pct",
    "one_type_correct_n",
    "one_type_correct_pct",
    "zero_type_correct_n",
    "zero_type_correct_pct",
]]
display(comparison)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Part 7. Case Studies

These are examples that connect back to the original motivation: Pokemon that look like one type to humans, but are actually something else.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """display(bundle["case_study_results"])

detail_df = bundle["type_oof_details"]
final_model_name = bundle["type_bundle"]["final_model_name"]
mistakes = detail_df[(detail_df["model"] == final_model_name) & (detail_df["result_bucket"] != "all_types_correct")].copy()
display(mistakes.head(20))
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Part 8. Streamlit Deployment Notes

- The Streamlit app uses a prebuilt deploy artifact.
- It clearly separates `Ground Truth` from `Model Prediction`.
- It also shows provenance, validation status, and benchmark-vs-deployment framing.
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """tables = notebook_ready_tables(bundle)
display(tables["type_random"])
display(tables["type_grouped"])
display(bundle["battle_reports"])
"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    return nb


def main() -> None:
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
