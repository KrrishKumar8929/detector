POKEMON DETECTOR FOR VIVALDI — INITIAL IMPLEMENTATION

This is a separate integration. It does not edit desktop_detector_v2_1.py or write to the existing database/map/reference images.

PACKAGE CONTENTS
- recognition_service.py: local HTTP service; reads the existing feature database and catch-name map.
- Start-PokemonDetectorService.ps1: launches the local service.
- extension\*: unpacked Manifest V3 Vivaldi extension.

INSTALL WITH POWERSHELL
1. Extract the ZIP to a temporary location, e.g. C:\Users\YOURNAME\Downloads\vivaldi_pokemon_detector.
2. Open PowerShell and run:
   New-Item -ItemType Directory -Force 'C:\PokemonDetector\vivaldi_pokemon_extension' | Out-Null
   Copy-Item -Recurse -Force 'C:\Users\YOURNAME\Downloads\vivaldi_pokemon_detector\*' 'C:\PokemonDetector\vivaldi_pokemon_extension\'
   Test-Path 'C:\PokemonDetector\poketwo_full_reference_features.pt'
   Test-Path 'C:\PokemonDetector\pokemon_catch_name_map.json'
   Test-Path 'C:\PokemonDetector\vivaldi_pokemon_extension\extension\manifest.json'
Replace YOURNAME and the extraction path with the real location. This copies only into the new subfolder; it does not overwrite the desktop detector or existing database/map.

START SERVICE
In PowerShell:
  Set-ExecutionPolicy -Scope Process Bypass
  & 'C:\PokemonDetector\vivaldi_pokemon_extension\Start-PokemonDetectorService.ps1'
Leave that window open. The service uses Python packages already used by the detector (torch, open_clip, Pillow). If the service reports a missing package, install it only in the Python environment used by your existing detector; do not rebuild the feature database.

LOAD EXTENSION IN VIVALDI
1. Open vivaldi://extensions
2. Enable Developer mode.
3. Click Load unpacked.
4. Select C:\PokemonDetector\vivaldi_pokemon_extension\extension
5. Open Discord Web and sign in normally. No Discord token is requested.
6. Click the extension icon and press Start monitoring. It starts with a baseline, so existing messages are ignored.
7. Select BOT 1 or BOT 2. When a spawn is recognized, inspect the generated command and click Copy command. Paste/send manually yourself.

BASIC SERVICE TEST
With the service running, open a second PowerShell window:
  Invoke-RestMethod 'http://127.0.0.1:8765/health'
The first recognition request may take time while the model/database load. The service binds only to 127.0.0.1.

IMPORTANT INITIAL-LIMITATION NOTES
- This is an initial implementation, not yet validated against the user's live Discord DOM or installed Windows Python environment.
- Spawn filtering uses conservative text heuristics plus Discord image URLs. Discord changes its DOM, so selectors/filter wording may need tuning after a live test.
- Browser fetch permissions/CORS and clipboard behavior must be verified in Vivaldi.
- A similarity score is not a calibrated probability.
- The service does not send messages to Discord.
- If an image arrives after the message insertion, DOM mutations retry detection; it does not intentionally add a fixed 1–2 second delay.
