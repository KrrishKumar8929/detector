from pathlib import Path
import random

from PIL import Image

PROJECT_DIR = Path(__file__).parent
REFERENCE_DIR = PROJECT_DIR / "poketwo_reference_images"

pokemon_folders = ["porygon", "tarountula", "totodile"]

extensions = {".jpg", ".jpeg", ".png", ".webp"}

print("\nPokétwo Reference Image Check")
print("=" * 45)

for pokemon in pokemon_folders:
    folder = REFERENCE_DIR / pokemon

    if not folder.exists():
        print(f"\n{pokemon}: FOLDER NOT FOUND")
        continue

    images = [
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and ".cache" not in path.parts
    ]

    print(f"\n{pokemon.title()}: {len(images)} images")

    if images:
        sample = random.choice(images)

        try:
            with Image.open(sample) as image:
                print(f"  Sample: {sample.name}")
                print(f"  Dimensions: {image.size}")
        except Exception as error:
            print(f"  Could not open sample: {error}")

print("\nCheck complete. No files were modified.")