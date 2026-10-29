import json
from pathlib import Path

names = json.load(open("pokemon_names.json", encoding="utf-8"))

if isinstance(names, dict):
    names = list(names.values())

print("Checking important Pokémon names:\n")

checks = [
    "zygarde", "aegislash", "diglett", "toxtricity", "tauros",
    "deoxys", "basculegion", "basculin", "ditto", "zarude",
    "darmanitan", "farfetch'd", "giratina", "frillish",
    "eiscue", "enamorus", "tatsugiri", "dudunsparce",
    "lycanroc", "maushold", "oricorio", "palafin",
    "mr. mime", "mr. rime", "nidoran♀", "nidoran♂",
    "urshifu", "wormadam", "shaymin", "meloetta",
    "castform", "gourgeist", "pumpkaboo", "pyroar",
    "squawkabilly", "wishiwashi", "ogerpon", "terapagos",
    "gimmighoul", "marshadow"
]

normalized = {
    str(name).replace("-", " ").strip().lower(): name
    for name in names
}

for name in checks:
    key = name.replace("-", " ").strip().lower()
    result = normalized.get(key)
    print(f"{name:20} -> {result if result is not None else 'NOT FOUND'}")

print(f"\nTotal Pokémon names: {len(names)}")
