from huggingface_hub import HfApi

REPO_ID = "SpreadSheets/Poketwo-Spawn-Images"

print("Connecting to Hugging Face...")
api = HfApi()

print("Reading dataset file list...")
files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

print(f"\nTotal files: {len(files)}")
print("\nFirst 40 files:")

for file in files[:40]:
    print(file)

image_extensions = (".jpg", ".jpeg", ".png", ".webp")

image_files = [
    file for file in files
    if file.lower().endswith(image_extensions)
]

print(f"\nTotal image files: {len(image_files)}")
print("\nFirst 20 image paths:")

for file in image_files[:20]:
    print(file)
