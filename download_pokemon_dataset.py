from huggingface_hub import snapshot_download

REPO_ID = "SpreadSheets/Poketwo-Spawn-Images"
DESTINATION = "poketwo_reference_images"

print("=" * 60)
print("POKETWO SPAWN IMAGE DOWNLOADER")
print("=" * 60)
print("Destination:", DESTINATION)
print("Existing Porygon, Tarountula, and Totodile folders will be preserved.")
print("Temporary uploads will be excluded.")
print("This may take a while.")
print()

try:
    result = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DESTINATION,
        allow_patterns=["*/*.jpg"],
        ignore_patterns=[
            "__temp_upload__/*",
            "porygon/*",
            "tarountula/*",
            "totodile/*",
        ],
    )
    print("\nDownload completed!")
    print("Dataset location:", result)

except KeyboardInterrupt:
    print("\nDownload interrupted. Run this script again to retry.")

except Exception as error:
    print("\nDownload failed:")
    print(type(error).__name__, error)
