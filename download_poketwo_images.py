from huggingface_hub import snapshot_download
from pathlib import Path

destination = (
    Path.home()
    / "OneDrive"
    / "Desktop"
    / "PokemonDetector"
    / "poketwo_reference_images"
)

destination.mkdir(parents=True, exist_ok=True)

print("Downloading selected Pokétwo images...")
print("Destination:", destination)

snapshot_download(
    repo_id="SpreadSheets/Poketwo-Spawn-Images",
    repo_type="dataset",
    local_dir=str(destination),
    allow_patterns=[
        "tarountula/**",
        "porygon/**",
        "totodile/**",
    ],
)

print("\nDownload finished.")
print("Images saved in:", destination)