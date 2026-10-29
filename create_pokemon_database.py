import json
import requests

API_URL = "https://pokeapi.co/api/v2/pokemon?limit=1025"

print("Downloading Pokémon names...")

try:
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()

    data = response.json()

    pokemon_database = {}

    for pokemon in data["results"]:
        pokemon_id = int(pokemon["url"].rstrip("/").split("/")[-1])
        pokemon_name = pokemon["name"].replace("-", " ").title()

        pokemon_database[str(pokemon_id)] = pokemon_name

    with open("pokemon_names.json", "w", encoding="utf-8") as file:
        json.dump(pokemon_database, file, indent=4, ensure_ascii=False)

    print(f"Successfully saved {len(pokemon_database)} Pokémon names.")
    print("Database file: pokemon_names.json")
    print("\nExample entries:")

    for pokemon_id in ["1", "4", "25", "150"]:
        print(f"{pokemon_id}: {pokemon_database[pokemon_id]}")

except requests.RequestException as error:
    print("Could not download Pokémon names.")
    print("Check your internet connection and try again.")
    print(f"Details: {error}")

except (KeyError, ValueError, json.JSONDecodeError) as error:
    print("The downloaded data could not be processed.")
    print(f"Details: {error}")

