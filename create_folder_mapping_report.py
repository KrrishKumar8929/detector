import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATASET = BASE / "poketwo_reference_images"
NAMES_FILE = BASE / "pokemon_names.json"
OUTPUT = BASE / "pokemon_folder_mapping_report.txt"

with open(NAMES_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    entries = sorted(data.items(), key=lambda item: int(item[0]))
else:
    entries = list(enumerate(data, start=1))

def normalize(name):
    name = str(name).lower().replace("♀", "f").replace("♂", "m")
    name = name.replace("farfetch'd", "farfetchd")
    name = name.replace("mr. mime", "mrmime")
    name = name.replace("mr. rime", "mrrime")
    name = re.sub(r"[^a-z0-9]", "", name)
    return name

# Base species lookup: canonical base name -> existing Pokémon ID.
base_lookup = {}

for pokemon_id, raw_name in entries:
    name = str(raw_name).strip()
    # Remove common form suffixes from the canonical name.
    # More complex names will be handled by explicit aliases later.
    base_lookup.setdefault(normalize(name), (int(pokemon_id), name))

# Explicit aliases from dataset folder names to canonical names.
aliases = {
    "family of three maushold": "maushold family of four",
    "hangry morpeko": "morpeko",
    "rapid strike urshifu": "urshifu single strike",
    "10% zygarde": "zygarde 50",
    "complete zygarde": "zygarde 50",
    "zygarde cell": "zygarde 50",
    "zygarde core": "zygarde 50",
    "aegislash": "aegislash shield",
    "blade aegislash": "aegislash shield",
    "alolan diglett": "diglett",
    "alolan dugtrio": "dugtrio",
    "alolan exeggutor": "exeggutor",
    "alolan geodude": "geodude",
    "alolan golem": "golem",
    "alolan graveler": "graveler",
    "alolan grimer": "grimer",
    "alolan marowak": "marowak",
    "alolan meowth": "meowth",
    "alolan muk": "muk",
    "alolan ninetales": "ninetales",
    "alolan persian": "persian",
    "alolan raichu": "raichu",
    "alolan raticate": "raticate",
    "alolan rattata": "rattata",
    "alolan sandshrew": "sandshrew",
    "alolan sandslash": "sandslash",
    "alolan vulpix": "vulpix",
    "amped toxtricity": "toxtricity amped",
    "low key toxtricity": "toxtricity amped",
    "aqua breed tauros": "tauros",
    "combat breed tauros": "tauros",
    "blaze breed tauros": "tauros",
    "attack deoxys": "deoxys normal",
    "defense deoxys": "deoxys normal",
    "speed deoxys": "deoxys normal",
    "deoxys": "deoxys normal",
    "basculin": "basculin red striped",
    "blue-striped basculin": "basculin red striped",
    "white-striped basculin": "basculin red striped",
    "darmanitan": "darmanitan standard",
    "galarian darmanitan": "darmanitan standard",
    "galarian zen darmanitan": "darmanitan standard",
    "zen darmanitan": "darmanitan standard",
    "galarian darumaka": "darumaka",
    "farfetch'd": "farfetchd",
    "galarian farfetch'd": "farfetchd",
    "galarian corsola": "corsola",
    "galarian linoone": "linoone",
    "galarian meowth": "meowth",
    "galarian mr. mime": "mr. mime",
    "galarian ponyta": "ponyta",
    "galarian rapidash": "rapidash",
    "galarian slowbro": "slowbro",
    "galarian slowking": "slowking",
    "galarian slowpoke": "slowpoke",
    "galarian stunfisk": "stunfisk",
    "galarian weezing": "weezing",
    "galarian yamask": "yamask",
    "galarian zapdos": "zapdos",
    "galarian zigzagoon": "zigzagoon",
    "giratina": "giratina altered",
    "gorging cramorant": "cramorant",
    "gulping cramorant": "cramorant",
    "hangry morpeko": "morpeko full belly",
    "hearthflame mask ogerpon": "ogerpon",
    "wellspring mask ogerpon": "ogerpon",
    "cornerstone mask ogerpon": "ogerpon",
    "hero palafin": "palafin zero",
    "hisuian arcanine": "arcanine",
    "hisuian avalugg": "avalugg",
    "hisuian braviary": "braviary",
    "hisuian decidueye": "decidueye",
    "hisuian electrode": "electrode",
    "hisuian goodra": "goodra",
    "hisuian growlithe": "growlithe",
    "hisuian lilligant": "lilligant",
    "hisuian qwilfish": "qwilfish",
    "hisuian samurott": "samurott",
    "hisuian sliggoo": "sliggoo",
    "hisuian sneasel": "sneasel",
    "hisuian typhlosion": "typhlosion",
    "hisuian voltorb": "voltorb",
    "hisuian zoroark": "zoroark",
    "hisuian zorua": "zorua",
    "midnight lycanroc": "lycanroc midday",
    "dusk lycanroc": "lycanroc midday",
    "meowstic": "meowstic male",
    "mimikyu": "mimikyu busted",
    "minior": "minior red meteor",
    "mow rotom": "rotom",
    "fan rotom": "rotom",
    "frost rotom": "rotom",
    "heat rotom": "rotom",
    "wash rotom": "rotom",
    "nidoranf": "nidoran female",
    "nidoranm": "nidoran male",
    "noice face eiscue": "eiscue ice",
    "paldean wooper": "wooper",
    "partner eevee": "eevee",
    "partner pikachu": "pikachu",
    "pirouette meloetta": "meloetta aria",
    "pa'u oricorio": "oricorio baile",
    "pom-pom oricorio": "oricorio baile",
    "sensu oricorio": "oricorio baile",
    "sandy wormadam": "wormadam plant",
    "trash wormadam": "wormadam plant",
    "school wishiwashi": "wishiwashi solo",
    "snowy castform": "castform",
    "rainy castform": "castform",
    "sunny castform": "castform",
    "three-segment dudunsparce": "dudunsparce two segment",
    "stretchy tatsugiri": "tatsugiri curly",
    "droopy tatsugiri": "tatsugiri curly",
    "terastal terapagos": "terapagos",
    "roaming gimmighoul": "gimmighoul",
    "easter bidoof": "bidoof",
    "egg nest lapras": "lapras",
    "cracked ditto": "ditto",
    "dada zarude": "zarude",
    "bloodmoon ursaluna": "ursaluna",
    "white plumage squawkabilly": "squawkabilly green plumage",
    "blue plumage squawkabilly": "squawkabilly green plumage",
    "yellow plumage squawkabilly": "squawkabilly green plumage",
    "blue-striped basculin": "basculin red striped",
    "zenith marshadow": "marshadow",
}

def find_target(folder):
    alias_target = aliases.get(folder.lower())
    if alias_target:
        target_norm = normalize(alias_target)
        for pokemon_id, raw_name in entries:
            if normalize(raw_name) == target_norm:
                return int(pokemon_id), str(raw_name), "alias"

    folder_norm = normalize(folder)

    # Exact normalized match.
    for pokemon_id, raw_name in entries:
        if normalize(raw_name) == folder_norm:
            return int(pokemon_id), str(raw_name), "exact"

    # Try matching a canonical form name beginning with the dataset folder name.
    candidates = [
        (int(pokemon_id), str(raw_name))
        for pokemon_id, raw_name in entries
        if normalize(raw_name).startswith(folder_norm)
    ]

    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1], "prefix"

    return None

folders = sorted(
    p.name for p in DATASET.iterdir()
    if p.is_dir() and p.name != ".cache"
)

report = []
matched = 0

for folder in folders:
    result = find_target(folder)
    if result:
        pokemon_id, canonical_name, method = result
        report.append(
            f"MATCH\t{folder}\t{pokemon_id}\t{canonical_name}\t{method}"
        )
        matched += 1
    else:
        report.append(f"UNMATCHED\t{folder}")

OUTPUT.write_text("\n".join(report), encoding="utf-8")

print(f"Dataset folders: {len(folders)}")
print(f"Mapped folders: {matched}")
print(f"Still unmatched: {len(folders) - matched}")
print(f"Report saved to: {OUTPUT}")
print("\nFirst 40 unmatched folders:")

for line in report:
    if line.startswith("UNMATCHED"):
        print(line)
