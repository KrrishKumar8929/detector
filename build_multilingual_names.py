import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "poketwo_official_data" / "csv" / "pokemon.csv"
OUTPUT_FILE = BASE_DIR / "pokemon_multilingual_names.json"

LANGUAGE_COLUMNS = {
    "english": "name.en",
    "english_alternate": "name.en2",
    "german": "name.de",
    "french": "name.fr",
}


def clean(value):
    if value is None:
        return ""
    return value.strip()


def main():
    if not CSV_FILE.is_file():
        raise FileNotFoundError(
            f"Pokémon CSV not found:\n{CSV_FILE}"
        )

    pokemon_records = []
    name_index = {}

    # utf-8-sig handles a possible UTF-8 byte-order mark.
    with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            names = {}

            for language, column in LANGUAGE_COLUMNS.items():
                name = clean(row.get(column, ""))

                if name:
                    names[language] = name

            record = {
                "id": clean(row.get("id", "")),
                "dex_number": clean(row.get("dex_number", "")),
                "slug": clean(row.get("slug", "")),
                "names": names,
                "region": clean(row.get("region", "")),
                "enabled": clean(row.get("enabled", "")),
                "catchable": clean(row.get("catchable", "")),
                "is_form": clean(row.get("is_form", "")),
                "form_item": clean(row.get("form_item", "")),
                "mega": clean(row.get("evo.mega", "")),
                "mega_x": clean(row.get("evo.mega_x", "")),
                "mega_y": clean(row.get("evo.mega_y", "")),
            }

            pokemon_records.append(record)

            # Index every available name and the Pokémon slug.
            possible_names = set(names.values())
            if record["slug"]:
                possible_names.add(record["slug"])

            for name in possible_names:
                key = name.casefold()

                if key not in name_index:
                    name_index[key] = []

                name_index[key].append(record["id"])

    output = {
        "source": "Pokétwo data repository CSV",
        "source_file": "csv/pokemon.csv",
        "record_count": len(pokemon_records),
        "languages": list(LANGUAGE_COLUMNS.keys()),
        "records": pokemon_records,
        "name_index": name_index,
    }

    # Write to a temporary file first, then replace the output.
    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8", newline="\n") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_file.replace(OUTPUT_FILE)

    print("Multilingual Pokémon database created successfully!")
    print(f"Records: {len(pokemon_records):,}")
    print(f"Distinct searchable names: {len(name_index):,}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("Languages included:")
    for language in LANGUAGE_COLUMNS:
        print(f"  - {language}")

    print()
    print("Note: This file preserves source names.")
    print("It does not yet determine the shortest valid catch name.")


if __name__ == "__main__":
    main()
