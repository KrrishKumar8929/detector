import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
import open_clip
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_DIR = BASE_DIR / "poketwo_reference_images"
DATABASE_FILE = BASE_DIR / "poketwo_full_reference_features.pt"
MAPPING_FILE = BASE_DIR / "pokemon_folder_mapping_report.txt"
NAMES_FILE = BASE_DIR / "pokemon_names.json"

BACKUP_FILE = BASE_DIR / "poketwo_full_reference_features_backup.pt"

BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize(name):
    name = str(name).lower().strip()

    name = name.replace("â™€", "f")
    name = name.replace("â™‚", "m")

    name = name.replace("farfetch'd", "farfetchd")
    name = name.replace("mr. mime", "mrmime")
    name = name.replace("mr. rime", "mrrime")

    return re.sub(r"[^a-z0-9]", "", name)


# ============================================================
# LOAD POKEMON NAMES
# ============================================================

def load_names():
    if not NAMES_FILE.is_file():
        return []

    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        entries = sorted(
            data.items(),
            key=lambda item: int(item[0])
        )
    else:
        entries = list(enumerate(data, start=1))

    return [
        (int(pokemon_id), str(name))
        for pokemon_id, name in entries
    ]


# ============================================================
# LOAD FOLDER MAPPING
# ============================================================

def load_folder_mapping():
    mapping = {}

    if not MAPPING_FILE.is_file():
        return mapping

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) >= 5 and parts[0] == "MATCH":
                try:
                    folder = parts[1]
                    pokemon_id = int(parts[2])
                    canonical_name = parts[3]

                    mapping[folder] = (
                        pokemon_id,
                        canonical_name
                    )

                except Exception:
                    pass

    return mapping


# ============================================================
# FIND POKEMON ID
# ============================================================

def find_pokemon_id(folder_name, folder_mapping, pokemon_names):

    # Custom Pokemon: use image-folder name for recognition.
    if folder_name.strip().lower() == "mosslax":
        return 0

    # Best option: existing mapping report
    if folder_name in folder_mapping:
        return folder_mapping[folder_name][0]

    normalized_folder = normalize(folder_name)

    # Try matching against pokemon_names.json
    matches = []

    for pokemon_id, pokemon_name in pokemon_names:
        normalized_name = normalize(pokemon_name)

        if not normalized_name:
            continue

        if normalized_name in normalized_folder:
            matches.append(
                (len(normalized_name), pokemon_id, pokemon_name)
            )

    if matches:
        # Longest matching Pokémon name wins.
        matches.sort(reverse=True)
        return matches[0][1]

    return 0


# ============================================================
# NORMALIZE PATH FOR DUPLICATE CHECKING
# ============================================================

def normalized_path(path):
    return str(path).replace("\\", "/").lower().strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("POKEMON REFERENCE DATABASE UPDATER")
    print("=" * 70)
    print()

    if not REFERENCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Reference folder not found:\n{REFERENCE_DIR}"
        )

    if not DATABASE_FILE.is_file():
        raise FileNotFoundError(
            f"Reference database not found:\n{DATABASE_FILE}"
        )

    # --------------------------------------------------------
    # Load existing database
    # --------------------------------------------------------

    print("Loading existing database...")

    database = torch.load(
        DATABASE_FILE,
        map_location="cpu",
        weights_only=False
    )

    old_features = database["features"].float()
    old_ids = database["image_ids"].long()
    old_paths = list(database["image_paths"])

    print(f"Existing reference images: {len(old_paths):,}")

    # --------------------------------------------------------
    # Build duplicate lookup
    # --------------------------------------------------------

    existing_paths = {
        normalized_path(path)
        for path in old_paths
    }

    # --------------------------------------------------------
    # Load mapping/name data
    # --------------------------------------------------------

    folder_mapping = load_folder_mapping()
    pokemon_names = load_names()

    # --------------------------------------------------------
    # Scan folders
    # --------------------------------------------------------

    print()
    print("Scanning Pokémon image folders...")

    new_records = []
    skipped = 0

    folders = sorted(
        p for p in REFERENCE_DIR.iterdir()
        if p.is_dir() and p.name != ".cache"
    )

    for folder in folders:

        pokemon_id = find_pokemon_id(
            folder.name,
            folder_mapping,
            pokemon_names
        )

        for image_path in sorted(folder.iterdir()):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            relative_path = image_path.relative_to(BASE_DIR)

            relative_string = str(relative_path)

            if normalized_path(relative_string) in existing_paths:
                skipped += 1
                continue

            new_records.append(
                (
                    image_path,
                    pokemon_id,
                    relative_string
                )
            )

    print()
    print(f"New images found:      {len(new_records):,}")
    print(f"Already in database:   {skipped:,}")

    if not new_records:
        print()
        print("No new images were found.")
        print("The database is already up to date.")
        return

    # --------------------------------------------------------
    # Show folders being added
    # --------------------------------------------------------

    new_folders = {}

    for image_path, pokemon_id, relative_path in new_records:
        folder_name = image_path.parent.name

        if folder_name not in new_folders:
            new_folders[folder_name] = 0

        new_folders[folder_name] += 1

    print()
    print("New images by Pokémon folder:")
    print("-" * 50)

    for folder_name, count in sorted(new_folders.items()):
        print(f"{folder_name}: {count:,}")

    # --------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------

    print()
    print("Loading CLIP model...")
    print(f"Device: {DEVICE}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    model = model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    print()
    print("Creating embeddings for new images...")
    print()

    new_features = []
    new_ids = []
    new_paths = []

    failed = 0
    total = len(new_records)

    for start in range(0, total, BATCH_SIZE):

        batch = new_records[
            start:start + BATCH_SIZE
        ]

        images = []
        ids = []
        paths = []

        for image_path, pokemon_id, relative_path in batch:

            try:

                with Image.open(image_path) as img:

                    image = preprocess(
                        img.convert("RGB")
                    )

                images.append(image)
                ids.append(pokemon_id)
                paths.append(relative_path)

            except Exception as exc:

                failed += 1

                print()
                print(
                    f"Could not read {image_path}: {exc}"
                )

        if images:

            tensor = torch.stack(images).to(DEVICE)

            with torch.inference_mode():

                features = model.encode_image(tensor)

                features = (
                    features
                    / features.norm(
                        dim=-1,
                        keepdim=True
                    ).clamp(min=1e-12)
                )

            new_features.append(
                features.cpu()
            )

            new_ids.extend(ids)
            new_paths.extend(paths)

        processed = min(
            start + len(batch),
            total
        )

        print(
            f"Processed: {processed:,}/{total:,}",
            end="\r",
            flush=True
        )

    print()
    print()

    if not new_features:

        print("No usable new images were processed.")
        return

    new_features = torch.cat(
        new_features,
        dim=0
    )

    new_ids = torch.tensor(
        new_ids,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Backup current database
    # --------------------------------------------------------

    print("Creating database backup...")

    shutil.copy2(
        DATABASE_FILE,
        BACKUP_FILE
    )

    print(f"Backup created:")
    print(BACKUP_FILE)

    # --------------------------------------------------------
    # Append new data
    # --------------------------------------------------------

    combined_features = torch.cat(
        [
            old_features,
            new_features
        ],
        dim=0
    )

    combined_ids = torch.cat(
        [
            old_ids,
            new_ids
        ],
        dim=0
    )

    combined_paths = (
        old_paths
        + new_paths
    )

    updated_database = {
        "features": combined_features,
        "image_ids": combined_ids,
        "image_paths": combined_paths
    }

    # --------------------------------------------------------
    # Save atomically
    # --------------------------------------------------------

    print()
    print("Saving updated database...")

    temp_file = DATABASE_FILE.with_suffix(
        ".tmp.pt"
    )

    torch.save(
        updated_database,
        temp_file
    )

    # Replace original only after successful save.
    temp_file.replace(DATABASE_FILE)

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATABASE UPDATE COMPLETE")
    print("=" * 70)

    print()
    print(f"Old images:       {len(old_paths):,}")
    print(f"Added images:     {len(new_paths):,}")
    print(f"Failed images:    {failed:,}")
    print(f"Total images:     {len(combined_paths):,}")

    print()
    print(f"Database:")
    print(DATABASE_FILE)

    print()
    print(f"Backup:")
    print(BACKUP_FILE)

    print()
    print("You can now restart Pokémon Detector.")


if __name__ == "__main__":
    main()