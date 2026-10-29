import json
import re
from pathlib import Path

import torch
import open_clip
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "poketwo_reference_images"
NAMES_FILE = BASE_DIR / "pokemon_names.json"
MAPPING_FILE = BASE_DIR / "pokemon_folder_mapping_report.txt"

REFERENCE_OUTPUT = BASE_DIR / "poketwo_full_reference_features.pt"
TEST_OUTPUT = BASE_DIR / "poketwo_full_test_features.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
TEST_PERCENT = 10

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def normalize(name):
    name = str(name).lower().strip()
    name = name.replace("♀", "f").replace("♂", "m")
    name = name.replace("farfetch'd", "farfetchd")
    name = name.replace("mr. mime", "mrmime")
    name = name.replace("mr. rime", "mrrime")
    return re.sub(r"[^a-z0-9]", "", name)


def load_names():
    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        entries = sorted(data.items(), key=lambda item: int(item[0]))
    else:
        entries = list(enumerate(data, start=1))

    # Preserve the exact IDs used by the existing project.
    return [
        (int(pokemon_id), str(name))
        for pokemon_id, name in entries
    ]


def load_folder_mapping():
    mapping = {}

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) >= 5 and parts[0] == "MATCH":
                folder = parts[1]
                pokemon_id = int(parts[2])
                canonical_name = parts[3]
                mapping[folder] = (pokemon_id, canonical_name)

    return mapping


def split_images(folder):
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    # Deterministic split: the same folder always gives the same test images.
    test_paths = paths[::10]
    test_set = set(test_paths)
    reference_paths = [p for p in paths if p not in test_set]

    return reference_paths, test_paths


def extract_features(model, preprocess, records, label):
    all_features = []
    all_ids = []
    all_paths = []
    failed = 0

    total = len(records)

    for start in range(0, total, BATCH_SIZE):
        batch_records = records[start:start + BATCH_SIZE]
        images = []
        ids = []
        paths = []

        for path, pokemon_id in batch_records:
            try:
                with Image.open(path) as img:
                    images.append(preprocess(img.convert("RGB")))
                    ids.append(pokemon_id)
                    paths.append(str(path.relative_to(BASE_DIR)))
            except Exception as exc:
                failed += 1
                print(f"\nCould not read {path}: {exc}")

        if images:
            tensor = torch.stack(images).to(DEVICE)

            with torch.inference_mode():
                features = model.encode_image(tensor)
                features = features / features.norm(
                    dim=-1, keepdim=True
                ).clamp(min=1e-12)

            all_features.append(features.cpu())
            all_ids.extend(ids)
            all_paths.extend(paths)

        processed = min(start + len(batch_records), total)
        print(
            f"{label}: {processed:,}/{total:,} files",
            end="\r",
            flush=True,
        )

    print()

    if all_features:
        features = torch.cat(all_features, dim=0)
    else:
        features = torch.empty((0, 512), dtype=torch.float32)

    result = {
        "features": features,
        "image_ids": torch.tensor(all_ids, dtype=torch.long),
        "image_paths": all_paths,
    }

    return result, failed


def main():
    if not REFERENCE_DIR.is_dir():
        raise FileNotFoundError(f"Dataset not found: {REFERENCE_DIR}")

    if not MAPPING_FILE.is_file():
        raise FileNotFoundError(f"Mapping report not found: {MAPPING_FILE}")

    names = load_names()
    folder_mapping = load_folder_mapping()

    folders = sorted(
        p for p in REFERENCE_DIR.iterdir()
        if p.is_dir() and p.name != ".cache"
    )

    missing = [folder.name for folder in folders
               if folder.name not in folder_mapping]

    print(f"Device: {DEVICE}")
    print(f"Pokémon names: {len(names)}")
    print(f"Dataset folders: {len(folders)}")
    print(f"Mapped folders: {len(folder_mapping)}")

    if missing:
        raise RuntimeError(
            f"{len(missing)} folders are unmapped: {missing[:20]}"
        )

    reference_records = []
    test_records = []
    folder_counts = {}

    for folder in folders:
        pokemon_id, canonical_name = folder_mapping[folder.name]
        reference_paths, test_paths = split_images(folder)

        reference_records.extend(
            (path, pokemon_id) for path in reference_paths
        )
        test_records.extend(
            (path, pokemon_id) for path in test_paths
        )

        folder_counts[folder.name] = {
            "pokemon_id": pokemon_id,
            "canonical_name": canonical_name,
            "reference_images": len(reference_paths),
            "test_images": len(test_paths),
        }

    print(f"Reference images planned: {len(reference_records):,}")
    print(f"Test images planned: {len(test_records):,}")

    print("\nLoading CLIP ViT-B-32...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )
    model = model.to(DEVICE)
    model.eval()

    print("\nExtracting reference features...")
    reference_data, reference_failed = extract_features(
        model, preprocess, reference_records, "Reference"
    )

    print("\nExtracting test features...")
    test_data, test_failed = extract_features(
        model, preprocess, test_records, "Test"
    )

    # Save to new files, never overwrite the old detector features.
    torch.save(reference_data, REFERENCE_OUTPUT)
    torch.save(test_data, TEST_OUTPUT)

    print("\n" + "=" * 55)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 55)
    print(f"Reference features: {len(reference_data['image_ids']):,}")
    print(f"Test features:      {len(test_data['image_ids']):,}")
    print(f"Failed reference:   {reference_failed:,}")
    print(f"Failed test:        {test_failed:,}")
    print(f"Reference shape:    {tuple(reference_data['features'].shape)}")
    print(f"Test shape:         {tuple(test_data['features'].shape)}")
    print(f"\nReference file: {REFERENCE_OUTPUT}")
    print(f"Test file:      {TEST_OUTPUT}")

    metadata_path = BASE_DIR / "poketwo_full_feature_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(folder_counts, f, indent=2, ensure_ascii=False)

    print(f"Metadata file:  {metadata_path}")


if __name__ == "__main__":
    main()