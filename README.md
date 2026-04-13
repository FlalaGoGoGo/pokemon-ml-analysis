# Pokemon ML Analysis

This repository now uses a canonical hybrid dataset for type prediction:

- `PokeAPI` is the structured backbone.
- official `pokemon.com` references are attached as URLs and validation metadata.
- legacy Kaggle CSVs are kept only for historical comparison and the old battle demo workflow.

## Main Files

- `canonical_pokemon.py`: canonical data sync, normalized tables, media manifest, validation report, image embedding cache
- `pokemon_project.py`: training, evaluation, benchmarking, and inference wrappers
- `streamlit_app.py`: Streamlit demo
- `sync_canonical_data.py`: one-click canonical data sync
- `build_analysis_notebook.py`: regenerates the analysis notebook
- `build_deploy_bundle.py`: creates the prebuilt Streamlit deploy artifact

## Install

For app usage and notebook viewing:

```bash
pip install -r requirements.txt
```

For full canonical rebuild plus image experiment:

```bash
pip install -r requirements.txt
pip install -r requirements-analysis.txt
```

## Rebuild Canonical Data

```bash
python sync_canonical_data.py
```

This writes canonical tables under:

- `artifacts/canonical/pokemon.csv`
- `artifacts/canonical/pokemon_species.csv`
- `artifacts/canonical/pokemon_forms.csv`
- `artifacts/canonical/pokemon_types.csv`
- `artifacts/canonical/pokemon_abilities.csv`
- `artifacts/canonical/pokemon_moves.csv`
- `artifacts/canonical/pokemon_flavor_texts.csv`
- `artifacts/canonical/pokemon_evolution_edges.csv`
- `artifacts/canonical/pokemon_media_manifest.csv`
- `artifacts/canonical/pokemon_text_corpus.csv`
- `artifacts/canonical/official_validation_report.csv`
- `artifacts/canonical/pokemon_master.csv`

Raw API caches are stored in `artifacts/canonical/raw_cache/` and are intentionally ignored by git.

## Rebuild Notebook

Generate the notebook shell:

```bash
python build_analysis_notebook.py
```

Execute it:

```bash
jupyter nbconvert --execute --inplace Pokemon_Project_Rebuild_Streamlit.ipynb
```

Or open it interactively:

```bash
jupyter lab Pokemon_Project_Rebuild_Streamlit.ipynb
```

## Build Streamlit Deploy Artifact

```bash
python build_deploy_bundle.py
```

This writes:

- `artifacts/streamlit_cloud_bundle.joblib`

The Streamlit app prefers this deploy artifact so Community Cloud can start quickly without retraining on boot.

## Run the App Locally

```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud

Commit these files before deploying:

- `streamlit_app.py`
- `pokemon_project.py`
- `canonical_pokemon.py`
- `requirements.txt`
- `artifacts/streamlit_cloud_bundle.joblib`
- `artifacts/canonical/*.csv`
- `.streamlit/config.toml`

Use:

- Repo: `FlalaGoGoGo/pokemon-ml-analysis`
- Branch: `main`
- Main file: `streamlit_app.py`

## Notes

- `pokemon.com` page content is bot-protected, so official references are recorded conservatively as URLs plus validation metadata rather than full mirrored page text.
- The deploy model may differ from the experimental multimodal notebook model if the lighter structured/text model is within the configured selection margin.
- Formal reporting should come from evolution-group OOF, not from whole-dataset deployment predictions.
