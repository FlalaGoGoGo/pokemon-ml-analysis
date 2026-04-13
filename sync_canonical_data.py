from __future__ import annotations

from canonical_pokemon import CANONICAL_TABLE_PATHS, sync_canonical_pokemon_data


def main() -> None:
    tables = sync_canonical_pokemon_data(force_refresh=False)
    print("Canonical tables written:")
    for name, path in CANONICAL_TABLE_PATHS.items():
        shape = tables[name].shape
        print(f"- {name}: {shape[0]} rows x {shape[1]} cols -> {path}")


if __name__ == "__main__":
    main()
