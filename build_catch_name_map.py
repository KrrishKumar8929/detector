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
        DATABASE_FILE,
        map_location="cpu",
        weights_only=False
    )

    records = data["records"]

    folders = sorted({
        Path(str(path).replace("\\", "/")).parent.name.lower()
        for path in database["image_paths"]
    })

    # Index records by their official slugs.
    by_slug = {
        record["slug"]: record
        for record in records
    }

    # Build folder-name aliases from slugs and allowed languages.
    aliases = defaultdict(set)
    name_owners = defaultdict(set)

    allowed_languages = (
        "english",
        "english_alternate",
        "german",
        "french",
    )

    for record in records:
        pokemon_id = record["id"]
        slug = record.get("slug", "")

        aliases[normalize(slug)].add(pokemon_id)

        for language in allowed_languages:
            name = record.get("names", {}).get(language, "")
            if name and name.strip():
                key = normalize(name)
                aliases[key].add(pokemon_id)
                name_owners[key].add(pokemon_id)

    # Explicit folder-to-slug exceptions.
    # Slugs are safer than hardcoded numeric IDs.
    folder_overrides = {
        "nidoranf": "nidoran-f",
        "nidoranm": "nidoran-m",
        "blue-striped basculin": "basculin-blue-striped",
        "white-striped basculin": "basculin-white-striped",
        "pom-pom oricorio": "oricorio-pom-pom",
        "three-segment dudunsparce": "dudunsparce-three-segment",
    }

    results = {}
    review = []
    unmatched = []

    for folder in folders:

        # User-approved names for ambiguous folders.
        if folder in ("basculegion", "oinkologne"):
            results[folder] = {
                "catch_name": folder,
                "source": "manual_approval",
                "status": "manually_approved",
            }
            review.append(
                f"{folder} -> {folder} [manually approved by user]"
            )
            continue



        # Preserve the user's custom Pokémon.
        if folder == "mosslax":
            results[folder] = {
                "catch_name": "mosslax",
                "source": "custom_folder",
                "status": "custom",
            }
            review.append("mosslax -> mosslax [custom folder]")
            continue

        folder_key = normalize(folder)

        if folder in folder_overrides:
            slug = folder_overrides[folder]
            record = by_slug.get(slug)
            record_ids = {record["id"]} if record else set()
        else:
            record_ids = aliases.get(folder_key, set())
            record = None

            if len(record_ids) == 1:
                pokemon_id = next(iter(record_ids))
                record = next(
                    (r for r in records if r["id"] == pokemon_id),
                    None
                )

        # Never guess when a folder cannot be uniquely identified.
        if len(record_ids) != 1 or record is None:
            unmatched.append(folder)

            results[folder] = {
                "catch_name": folder,
                "source": "folder_fallback",
                "status": "needs_review",
            }

            review.append(
                f"NEEDS REVIEW: {folder} -> {sorted(record_ids)}"
            )
            continue

        slug = record.get("slug", "")
        names = record.get("names", {})

        # These Pokémon must retain their complete English identities.
        # Mega forms are identified by their slugs in this dataset.
        protected_form = (
            bool(record.get("is_form"))
            or "mega" in slug.casefold()
            or slug in {
                "nidoran-f",
                "nidoran-m",
                "toxtricity-amped",
            }
        )

        candidates = []

        if protected_form:
            # Do not replace a form's identity with a generic translated
            # name or a shorter name that might hide the form distinction.
            english = names.get("english", "").strip()

            if english:
                candidates.append(english)

        else:
            # For base Pokémon, choose the shortest unique name from the
            # allowed languages. Japanese names are never considered.
            for language in allowed_languages:
                name = names.get(language, "").strip()

                if not name:
                    continue

                key = normalize(name)

                if len(name_owners[key]) == 1:
                    candidates.append(name)

        if not candidates:
            selected = folder
            status = "needs_review"
            unmatched.append(folder)
        else:
            selected = sorted(
                set(candidates),
                key=lambda name: (len(name), name.casefold())
            )[0]
            status = "matched"

        # The user explicitly wants both Nidoran folders to use nidoran.
        if folder in ("nidoranf", "nidoranm"):
            selected = "nidoran"
            status = "matched"

        results[folder] = {
            "catch_name": selected.lower(),
            "source": slug,
            "status": status,
        }

        review.append(
            f"{folder} -> {selected.lower()} "
            f"[slug={slug}; protected_form={protected_form}; "
            f"candidates={sorted(set(candidates))}]"
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
    print(
        "Matched folders:",
        sum(v["status"] == "matched" for v in results.values())
    )
    print(
        "Custom folders:",
        sum(v["status"] == "custom" for v in results.values())
    )
    print(f"Folders needing review: {len(unmatched):,}")
    print(f"Mapping file: {OUTPUT_FILE}")
    print(f"Review report: {REPORT_FILE}")
    print()
    print("The detector, feature database, and reference images were not modified.")


if __name__ == "__main__":
    main()
