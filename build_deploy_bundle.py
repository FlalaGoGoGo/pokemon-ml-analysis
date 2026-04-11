from __future__ import annotations

from pathlib import Path

from pokemon_project import DEPLOY_BUNDLE_PATH, save_deploy_bundle, train_deploy_bundle


def main() -> None:
    bundle = train_deploy_bundle(Path.cwd(), include_external=False)
    artifact_path = save_deploy_bundle(bundle, DEPLOY_BUNDLE_PATH, compress=3)
    artifact_size_mb = artifact_path.stat().st_size / 1024 / 1024
    print(f"Deploy bundle written to: {artifact_path}")
    print(f"Bundle size: {artifact_size_mb:.2f} MB")
    print(f"Type model: {bundle['type_bundle']['final_model_name']}")
    print(f"Battle model: {bundle['battle_bundle']['final_model_name']}")


if __name__ == "__main__":
    main()

