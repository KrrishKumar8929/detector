import os
import requests
import time

# Folder where Pokémon images will be saved.
IMAGE_FOLDER = "pokemon_images"

# Create the folder if it doesn't exist.
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Official artwork image collection.
BASE_URL = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/"
    "master/sprites/pokemon/other/official-artwork/{}.png"
)

TOTAL_POKEMON = 1025

session = requests.Session()

print("Pokémon Image Downloader")
print("-------------------------")

downloaded = 0
skipped = 0
failed = []

for pokemon_id in range(1, TOTAL_POKEMON + 1):
    filename = os.path.join(
        IMAGE_FOLDER,
        f"{pokemon_id}.png"
    )

    # Don't download an image again if it already exists.
    if os.path.exists(filename):
        skipped += 1
        continue

    url = BASE_URL.format(pokemon_id)

    try:
        response = session.get(url, timeout=20)

        if response.status_code == 200:
            with open(filename, "wb") as image_file:
                image_file.write(response.content)

            downloaded += 1
        else:
            failed.append(pokemon_id)

    except requests.RequestException:
        failed.append(pokemon_id)

    # Show progress every 25 Pokémon.
    if pokemon_id % 25 == 0:
        print(
            f"Checked {pokemon_id}/{TOTAL_POKEMON} | "
            f"Downloaded: {downloaded} | "
            f"Already saved: {skipped} | "
            f"Failed: {len(failed)}"
        )

    # Avoid making requests too quickly.
    time.sleep(0.03)

print("\nDownload finished!")
print(f"New images downloaded: {downloaded}")
print(f"Already saved: {skipped}")
print(f"Failed downloads: {len(failed)}")

if failed:
    print("Some images failed to download.")
    print("You can run this program again to retry them.")
else:
    print("All requested images downloaded successfully.")

print(f"\nImages are stored in: {IMAGE_FOLDER}")