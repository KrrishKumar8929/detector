import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import open_clip
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "poketwo_reference_images"
PEAKYCHU_DIR = REFERENCE_DIR / "peakychu"
DATABASE_FILE = BASE_DIR / "poketwo_full_reference_features.pt"
CATCH_MAP_FILE = BASE_DIR / "pokemon_catch_name_map.json"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# These are the exact manual catch-name choices supplied by the user.
MAPPINGS = {
    "sandy wormadam": "wormadam",
    "trash wormadam": "wormadam",
    "heat rotom": "rotom",
    "wash rotom": "rotom",
    "frost rotom": "rotom",
    "fan rotom": "rotom",
    "mow rotom": "rotom",
    "dj rotom": "rotom",
    "rotom pokedex": "rotom",
    "rotom phone": "rotom",
    "rotom drone": "rotom",
    "droopy tatsugiri": "tatsugiri",
    "stretchy tatsugiri": "tatsugiri",
    "three-segment dudunsparce": "dudunsparce",
    "gimmighoul": "mordudor",
    "roaming gimmighoul": "mordudor",
    "gingerbread gimmighoul": "mordudor",
    "chi-yu": "yuyu",
    "gradient chi-yu": "yuyu",
    "chien-pao": "baojian",
    "clobbopus": "tatakko",
    "crabominable": "krawell",
    "farfetch'd": "porenta",
    "jigglypuff": "purin",
    "amped toxtricity": "toxtricity",
    "aqua breed tauros": "tauros",
    "attack deoxys": "deoxys",
    "blade aegislash": "aegislash",
    "blaze breed tauros": "tauros",
    "bloodmoon ursaluna": "ursaluna",
    "blue plumage squawkabilly": "squawkabilly",
    "blue-striped basculin": "basculin",
    "combat breed tauros": "tauros",
    "cornerstone mask ogerpon": "ogerpon",
    "cracked ditto": "ditto",
    "dada zarude": "zarude",
    "defense deoxys": "deoxys",
    "dusk lycanroc": "lycanroc",
    "easter bidoof": "bidoof",
    "egg nest lapras": "lapras",
    "family of three maushold": "maushold",
    "gorging cramorant": "cramorant",
    "gouging fire": "gouging fire",
    "great tusk": "great tusk",
    "gulping cramorant": "cramorant",
    "hangry morpeko": "morpeko",
    "hearthflame mask ogerpon": "ogerpon",
    "hero palafin": "palafin",
    "iron boulder": "iron boulder",
    "low key toxtricity": "toxtricity",
    "midnight lycanroc": "lycanroc",
    "noice face eiscue": "eiscue",
    "pa'u oricorio": "oricorio",
    "paldean wooper": "wooper",
    "partner eevee": "eevee",
    "partner pikachu": "pikachu",
    "pirouette meloetta": "meloetta",
    "pom-pom oricorio": "oricorio",
    "rainy castform": "castform",
    "rapid strike urshifu": "urshifu",
    "school wishiwashi": "wishiwashi",
    "sensu oricorio": "oricorio",
    "snowy castform": "castform",
    "speed deoxys": "deoxys",
    "sunny castform": "castform",
    "terastal terapagos": "terapagos",
    "wellspring mask ogerpon": "ogerpon",
    "white plumage squawkabilly": "squawkabilly",
    "white-striped basculin": "basculin",
    "yellow plumage squawkabilly": "squawkabilly",
    "zenith marshadow": "marshadow",
    "zen darmanitan": "darmanitan",
}

def normalized_path(path):
    return str(path).replace("\\", "/").lower().strip()

def main():
    print("=" * 68)
    print("POKEMON DETECTOR: MANUAL CATCH-NAME MAP + PEAKYCHU UPDATE")
    print("=" * 68)

    for required in (REFERENCE_DIR, PEAKYCHU_DIR, DATABASE_FILE, CATCH_MAP_FILE):
        if not required.exists():
            raise FileNotFoundError(f"Required file or folder not found:\n{required}")

    image_paths = sorted(
        p for p in PEAKYCHU_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in {PEAKYCHU_DIR}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    database_backup = BASE_DIR / f"poketwo_full_reference_features_backup_before_peakychu_{stamp}.pt"
    map_backup = BASE_DIR / f"pokemon_catch_name_map_backup_before_manual_changes_{stamp}.json"

    print(f"Peakychu images found: {len(image_paths)}")
    print(f"Manual name mappings to apply: {len(MAPPINGS)}")
    print("Creating fresh backups...")
    shutil.copy2(DATABASE_FILE, database_backup)
    shutil.copy2(CATCH_MAP_FILE, map_backup)
    print(f"Database backup: {database_backup.name}")
    print(f"Catch-map backup: {map_backup.name}")

    # Read and prepare catch map without discarding any existing entries or metadata.
    with open(CATCH_MAP_FILE, "r", encoding="utf-8") as f:
        catch_map = json.load(f)
    if not isinstance(catch_map, dict):
        raise TypeError("Catch-name map must be a JSON object/dictionary.")

    for folder_name, catch_name in MAPPINGS.items():
        existing = catch_map.get(folder_name)
        if isinstance(existing, dict):
            existing["catch_name"] = catch_name
            # Keep existing source/status and any other metadata intact.
        else:
            catch_map[folder_name] = {
                "catch_name": catch_name,
                "source": "user_manual",
                "status": "matched",
            }

    existing_peakychu = catch_map.get("peakychu")
    if isinstance(existing_peakychu, dict):
        existing_peakychu["catch_name"] = "peakychu"
    else:
        catch_map["peakychu"] = {
            "catch_name": "peakychu",
            "source": "user_manual",
            "status": "matched",
        }

    # Load existing feature database and verify all parallel arrays align.
    database = torch.load(DATABASE_FILE, map_location="cpu", weights_only=False)
    if not all(key in database for key in ("features", "image_ids", "image_paths")):
        raise KeyError("Feature database is missing features, image_ids, or image_paths.")

    old_features = database["features"].float().cpu()
    old_ids = database["image_ids"].long().cpu()
    old_paths = list(database["image_paths"])

    if old_features.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor, got {tuple(old_features.shape)}")
    if not (len(old_features) == len(old_ids) == len(old_paths)):
        raise ValueError("Existing feature, ID, and path counts do not match.")

    existing_paths = {normalized_path(p) for p in old_paths}
    new_records = []
    for image_path in image_paths:
        relative = str(image_path.relative_to(BASE_DIR))
        if normalized_path(relative) not in existing_paths:
            new_records.append((image_path, relative))

    print(f"Existing database images: {len(old_paths):,}")
    print(f"New Peakychu images to add: {len(new_records)}")

    # Load the same OpenCLIP model used by the project.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OpenCLIP model on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )
    model = model.to(device)
    model.eval()

    new_features_list = []
    new_ids_list = []
    new_paths = []

    with torch.inference_mode():
        for index, (image_path, relative_path) in enumerate(new_records, start=1):
            try:
                with Image.open(image_path) as image:
                    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
                features = model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                new_features_list.append(features.squeeze(0).float().cpu())
                # Custom folders such as Mosslax use ID 0 in this project.
                # Recognition is based on folder names in image_paths, not this ID.
                new_ids_list.append(0)
                new_paths.append(relative_path)
                print(f"[{index}/{len(new_records)}] Encoded {image_path.name}")
            except Exception as exc:
                raise RuntimeError(f"Failed to process image {image_path}:\n{exc}") from exc

    if new_features_list:
        added_features = torch.stack(new_features_list)
        if added_features.shape[1] != old_features.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: old={old_features.shape[1]}, "
                f"new={added_features.shape[1]}"
            )
        combined_features = torch.cat((old_features, added_features), dim=0)
        combined_ids = torch.cat((old_ids, torch.tensor(new_ids_list, dtype=torch.long)), dim=0)
        combined_paths = old_paths + new_paths
    else:
        combined_features = old_features
        combined_ids = old_ids
        combined_paths = old_paths

    updated_database = {
        "features": combined_features,
        "image_ids": combined_ids,
        "image_paths": combined_paths,
    }

    # Write temporary outputs first. Validate the database before replacing the original.
    temp_database = DATABASE_FILE.with_name(DATABASE_FILE.stem + ".peakychu.tmp.pt")
    temp_map = CATCH_MAP_FILE.with_name(CATCH_MAP_FILE.name + ".tmp")

    torch.save(updated_database, temp_database)
    check_db = torch.load(temp_database, map_location="cpu", weights_only=False)
    if not (
        len(check_db["features"]) == len(check_db["image_ids"]) == len(check_db["image_paths"])
    ):
        temp_database.unlink(missing_ok=True)
        raise RuntimeError("Temporary database validation failed; original database was not replaced.")

    with open(temp_map, "w", encoding="utf-8") as f:
        json.dump(catch_map, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Replace each target only after both temporary files were successfully written.
    # Backups above allow recovery if anything unexpected occurs.
    os.replace(temp_database, DATABASE_FILE)
    os.replace(temp_map, CATCH_MAP_FILE)

    print()
    print("=" * 68)
    print("UPDATE COMPLETED")
    print(f"Catch-map entries: {len(catch_map):,}")
    print(f"Manual mappings applied: {len(MAPPINGS)}")
    print(f"Peakychu images added: {len(new_paths)}")
    print(f"Database image count: {len(combined_paths):,}")
    print(f"Feature shape: {tuple(combined_features.shape)}")
    print("Existing image paths and features were retained.")
    print(f"Database backup: {database_backup}")
    print(f"Catch-map backup: {map_backup}")
    print("=" * 68)

if __name__ == "__main__":
    main()
