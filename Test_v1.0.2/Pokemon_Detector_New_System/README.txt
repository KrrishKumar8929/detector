POKEMON DETECTOR — NEW DETECTION SYSTEM (v1.0.0)

WHAT THIS BUILD DOES
- Replaces the Discord DOM detector with a resilient mutation observer that watches message text, new elements, image URLs, and late-loaded embed images.
- Uses the existing selected-channel / selected-tab settings, floating per-tile panel, global Stop, name/language options, user alt names, blacklist, BOT1/BOT2 command formatting, and copy-to-clipboard workflow.
- Keeps the queue sequential with NO additional 1.5-second cooldown.
- Local recognition compares three reference types when available:
  1) full spawn images with backgrounds: C:\PokemonDetector\poketwo_full_reference_features.pt
  2) transparent/no-background official images: C:\PokemonDetector\poketwo_official_data\images
  3) official silhouettes: C:\PokemonDetector\poketwo_official_data\silhouettes
- The source reference images and existing full-spawn feature database are read-only. Separate cache files are used for transparent and silhouette references.
- It never sends a catch message automatically. It can copy the command to the clipboard, and you decide whether to paste/send it.

INSTALL
1. Extract this ZIP to Downloads. Keep the extracted folder in place.
2. Close the currently running recognition-service PowerShell window with Ctrl+C.
3. Open PowerShell, then run (assuming you extracted into Downloads with the default folder name):
   Set-ExecutionPolicy -Scope Process Bypass
   & "$env:USERPROFILE\Downloads\Pokemon_Detector_New_System\Install-Recognition-Service.ps1"
   If you extracted somewhere else, replace that path with the actual extracted folder path.
   This script backs up the current recognition_service.py before replacing it. It does not edit any database or image files.
4. Start the service:
   & 'C:\PokemonDetector\vivaldi_pokemon_extension\Start-PokemonDetectorService.ps1'
   Keep this window open.
5. In Vivaldi open vivaldi://extensions. Disable the old Pokémon Detector while testing, so two versions don't monitor the same messages.
6. Enable Developer mode, click Load unpacked, and select the package's 'extension' folder. It contains manifest.json.
7. Refresh every Discord tile/tab with Ctrl+R. Then open the extension popup, select your channels, and start monitoring.

REFERENCE FOLDERS
- Background spawn features: C:\PokemonDetector\poketwo_full_reference_features.pt (existing database; not rebuilt)
- Transparent/no-background art: C:\PokemonDetector\poketwo_official_data\images (existing 1,770 files)
- Silhouettes: C:\PokemonDetector\poketwo_official_data\silhouettes (optional; if absent, recognition still works with the other two sources)

FIRST RUN
The model initializes on the first recognition request. The first run may take longer if it has to build the silhouette feature cache. After that, the silhouette cache is reused unless the folder contents change.

CHECK HEALTH
In a second PowerShell window:
  Invoke-RestMethod 'http://127.0.0.1:8765/health' | Format-List
The response includes cropped_indexed and silhouette_indexed. silhouette_indexed is 0 if the silhouettes folder is missing or contains unsupported/unnamed numeric files.

TROUBLESHOOTING
- If the panel stays on 'Monitoring active — waiting for a new spawn', open Discord DevTools (F12 > Console). Look for '[Pokémon Detector] Spawn candidate detected'. If no message appears, verify the channel is selected in the extension popup and refresh that Discord tile after loading the extension.
- If the extension was reloaded while Discord stayed open, refresh every Discord tile to replace stale content scripts.
- If the service says loaded=False immediately after starting, that is expected until the first /recognize request initializes it. Do not delete/rebuild the full database.
- This is a new integration build and should be tested on a new spawn before relying on it. Scores are similarity scores, not calibrated probabilities.


Version 1.0.2 performance change:
- Recognition jobs are now processed with bounded concurrency (2 at a time) instead of one global sequential queue.
- The pending queue is capped at 8; if it fills, the oldest not-yet-started job is discarded so new spawns do not wait behind a large backlog.
- Recognition databases and matching logic are unchanged. All three reference sources remain enabled.
- This change is not a guarantee of a specific recognition time; test with your normal number of Discord tiles.
