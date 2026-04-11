# Pokemon ML Rebuild

This repository contains:

- `Pokemon_Project_Rebuild_Streamlit.ipynb`: the rebuilt analysis notebook
- `pokemon_project.py`: shared data, training, evaluation, and inference code
- `streamlit_app.py`: the Streamlit app
- `build_deploy_bundle.py`: creates a prebuilt deploy artifact for Streamlit Cloud

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter lab Pokemon_Project_Rebuild_Streamlit.ipynb
```

Run the app:

```bash
streamlit run streamlit_app.py
```

## Prepare Streamlit Cloud Deployment

Create the deploy artifact before pushing to GitHub:

```bash
python build_deploy_bundle.py
```

This writes:

- `artifacts/streamlit_cloud_bundle.joblib`

The Streamlit app prefers this artifact so Community Cloud can start quickly without retraining models on boot.

## One-Click Streamlit Cloud Deployment

1. Push this folder to a GitHub repository.
2. Make sure these files are committed:
   - `streamlit_app.py`
   - `requirements.txt`
   - `pokemon_project.py`
   - `artifacts/streamlit_cloud_bundle.joblib`
   - `.streamlit/config.toml`
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Choose `New app`.
5. Select the repository, branch, and set the main file path to:

```text
streamlit_app.py
```

6. Deploy. Streamlit Cloud will build the environment and expose a public `*.streamlit.app` URL.

## Notes

- If the deploy artifact is missing, the app can still fall back to training locally, but startup will be much slower.
- The notebook keeps the full training workflow; the Cloud deployment uses a deploy-optimized battle model artifact to keep repository size and boot time reasonable.
