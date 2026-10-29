Pokemon Detector v0.7.1 - Panel Fix

This build restores floating-panel visibility on extension install/update and retries panel creation briefly during Discord startup. It keeps Add All Selected Tabs and the global Stop button.

INSTALL
1. Extract this ZIP to a NEW folder.
2. Open vivaldi://extensions.
3. Remove/disable the old unpacked detector entry (do not delete your reference-image database or service folder).
4. Click Load unpacked and select the extracted folder.
5. Refresh every Discord tab/tile with Ctrl+R.
6. Open the extension popup and click Start monitoring. Adding channels does not automatically start monitoring.

IF THE PANEL APPEARS BUT RECOGNITION FAILS
The Windows recognition service must also be running. In a browser tab open http://127.0.0.1:8765/health . It should return an OK/healthy response and loaded=True. If it says connection refused, start your existing Pokemon Detector service using your usual PowerShell script. This extension ZIP does not contain the model/service files.

If the panel remains missing: open the extension popup, click Show panels on all tiles if that button is available, then refresh Discord again.
