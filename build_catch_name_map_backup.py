import json
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"C:\PokemonDetector")
JSON_FILE = BASE_DIR / "pokemon_multilingual_names.json"
DATABASE_FILE = BASE_DIR / "poketwo_full_reference_features.pt"
OUTPUT_FILE = BASE_DIR / "pokemon_catch_name_map.json"
REPORT_FILE = BASE_DIR / "pokemon_catch_name_review.txt"

def normalize(value):
    value = str(value or "").casefold()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^\w\s]", "", value)
    return " ".join(value.split())

def main():
    import torch

    data = json.loads(JSON_FILE.read_text(encoding="utf-8"))
    database = torch.load(
        DATABASE_FILE, map_location="cpu", weights_only=False
    )

    records = data["records"]
    folders = sorted({
        Path(str(path).replace("\\", "/")).parent.name.lower()
        for path in database["image_paths"]
    })

    # Build aliases for locating the correct source record.
    aliases = defaultdict(set)

    for record in records:
        slug = record.get("slug", "")
        aliases[normalize(slug)].add(record["id"])

        for name in record.get("names", {}).values():
            if name.strip():
                aliases[normalize(name)].add(record["id"])

    # Names shared across different records are not safe catch names.
    name_owners = defaultdict(set)

    for record in records:
        for name in record.get("names", {}).values():
            if name.strip():
                name_owners[normalize(name)].add(record["id"])

    # Explicit folder-to-record exceptions.
    folder_overrides = {
        "nidoranf": "29",
        "nidoranm": "32",
        "blue-striped basculin": "10016",
        "white-striped basculin": "10247",
        "pom-pom oricorio": "10123",
        "three-segment dudunsparce": "10255",
    }

    by_id = {record["id"]: record for record in records}
    results = {}
    review = []
    unmatched = []

    for folder in folders:
        folder_key = normalize(folder)

        if folder == "mosslax":
            results[folder] = {
                "catch_name": "mosslax",
                "source": "custom_folder",
                "status": "custom",
            }
            continue

        record_ids = set()

        if folder in folder_overrides:
            record_ids.add(folder_overrides[folder])
        else:
            record_ids.update(aliases.get(folder_key, set()))

        # A folder should resolve to exactly one source record.
        if len(record_ids) != 1:
            unmatched.append(folder)
            results[folder] = {
                "catch_name": folder,
                "source": "folder_fallback",
                "status": "needs_review",
            }
            review.append(
                f"UNMATCHED OR AMBIGUOUS FOLDER: {folder} -> "
                f"{sorted(record_ids)}"
            )
            continue

        record = by_id[next(iter(record_ids))]
        slug = record.get("slug", "")
        names = record.get("names", {})

        # Preserve form-specific English names. For base Pokémon,
        # consider all allowed languages, but reject shared aliases.
        is_form = bool(record.get("is_form")) or (
            slug.lower() in {
                "nidoran-f", "nidoran-m"
            }
        )

        candidates = []

        if is_form:
            english = names.get("english", "").strip()
            alternate = names.get("english_alternate", "").strip()

            if english and normalize(english) != normalize("Toxtricity"):
                candidates.append(english)

            # Alternate English is safe only if unique and does not
            # erase a known form distinction.
            if alternate:
                key = normalize(alternate)
                if (
                    len(name_owners[key]) == 1
                    and not (
                        "toxtricity" in slug.lower()
                        and key == normalize("toxtricity")
                    )
                ):
                    candidates.append(alternate)
        else:
            for language in (
                "english", "english_alternate", "german", "french"
            ):
                name = names.get(language, "").strip()
                if not name:
                    continue

                key = normalize(name)

                if len(name_owners[key]) == 1:
                    candidates.append(name)

        # Folder identity is a safe fallback if no language candidate
        # is both unique and suitable.
        if not candidates:
            candidates = [folder]

        # Keep the shortest candidate; break ties deterministically.
        selected = sorted(
            set(candidates),
            key=lambda name: (len(name), name.casefold())
        )[0]

        # Nidoran is intentionally normalized to the same catch name.
        if folder in ("nidoranf", "nidoranm"):
            selected = "nidoran"

        results[folder] = {
            "catch_name": selected.lower(),
            "source": slug,
            "status": "matched",
        }

        review.append(
            f"{folder} -> {selected.lower()} "
            f"[record={slug}; candidates={sorted(set(candidates))}]"
        )

    OUTPUT_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    REPORT_FILE.write_text(
        "\n".join(review) + "\n",
        encoding="utf-8"
    )

    print("Mapping generation complete.")
    print(f"Reference folders: {len(folders):,}")
    print(f"Mapped folders: {sum(v['status'] == 'matched' for v in results.values()):,}")
    print(f"Custom folders: {sum(v['status'] == 'custom' for v in results.values()):,}")
    print(f"Folders needing review: {len(unmatched):,}")
    print(f"Mapping: {OUTPUT_FILE}")
    print(f"Review report: {REPORT_FILE}")
    print()
    print("No detector or feature-database files were modified.")

if __name__ == "__main__":
    main()

