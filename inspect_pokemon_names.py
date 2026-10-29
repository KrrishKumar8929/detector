import json

with open("pokemon_names.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Data type:", type(data).__name__)

if isinstance(data, dict):
    print("First 10 entries:")
    for key, value in list(data.items())[:10]:
        print(repr(key), "->", repr(value))
    names = list(data.values())
else:
    names = data
    print("First 10 entries:", repr(names[:10]))

print("\nSearching for expected Pokémon:")
terms = [
    "zygarde", "aegislash", "toxtricity", "deoxys",
    "basculegion", "basculin", "darmanitan", "farfetch",
    "giratina", "frillish", "eiscue", "enamorus",
    "tatsugiri", "dudunsparce", "lycanroc", "maushold",
    "oricorio", "palafin", "mr. mime", "mr. rime",
    "urshifu", "wormadam", "shaymin", "meloetta",
    "gourgeist", "pumpkaboo", "pyroar", "squawkabilly",
    "wishiwashi"
]

for term in terms:
    matches = [repr(n) for n in names if term in str(n).lower()]
    print(f"{term:15} -> {', '.join(matches[:8]) if matches else 'NO MATCH'}")

print("\nLast 30 entries:")
for name in names[-30:]:
    print(repr(name))
