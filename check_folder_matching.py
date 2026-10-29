import json
from pathlib import Path

base = Path(".")
root = base / "poketwo_reference_images"

with open(base / "pokemon_names.json", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    try:
        data = dict(sorted(data.items(), key=lambda x: int(x[0])))
    except (ValueError, TypeError):
        pass
    names = list(data.values())
else:
    names = data

known = {
    str(name).replace("-", " ").strip().lower()
    for name in names
}

folders = sorted(
    p for p in root.iterdir()
    if p.is_dir() and p.name != ".cache"
)

matched = [p for p in folders if p.name.replace("-", " ").strip().lower() in known]
unmatched = [p for p in folders if p.name.replace("-", " ").strip().lower() not in known]

print("Pokédex names:", len(known))
print("Dataset folders:", len(folders))
print("Matched folders:", len(matched))
print("Unmatched folders:", len(unmatched))

print("\nExamples of unmatched folders:")
for folder in unmatched[:50]:
    print(folder.name)
