from pathlib import Path
from datetime import datetime
import re

path = Path("desktop_detector.py")
source = path.read_text(encoding="utf-8")

pattern = r"(?ms)^def wait_for_clear_spawn\(initial_crop\):\n.*?(?=^# ============================================================\n# IMAGE RECOGNITION)"

match = re.search(pattern, source)
if not match:
    raise SystemExit("ERROR: Function boundaries not found. No changes made.")

old_function = match.group(0)

new_function = '''def wait_for_clear_spawn(initial_crop, box=None):
    """
    Wait briefly for Discord's Pokémon artwork to load, then
    recapture the known crop without performing OCR a second time.
    """

    time.sleep(0.15)

    if not monitoring or box is None:
        return initial_crop

    if selected_area is not None:
        origin_x, origin_y = selected_area[0], selected_area[1]
    else:
        window = get_discord_window()
        if window is None:
            return initial_crop
        origin_x, origin_y = max(0, window.left), max(0, window.top)

    left, top, right, bottom = box
    region = (
        origin_x + left,
        origin_y + top,
        right - left,
        bottom - top
    )

    try:
        screenshot = pyautogui.screenshot(region=region).convert("RGB")
        if screenshot.width > 0 and screenshot.height > 0:
            return screenshot
    except Exception as exc:
        print(f"Updated spawn capture failed: {exc}")

    return initial_crop


'''

source = source[:match.start()] + new_function + source[match.end():]

old_call = "crop = wait_for_clear_spawn(crop)"
if source.count(old_call) != 1:
    raise SystemExit("ERROR: Expected call site not found exactly once. No changes made.")

source = source.replace(
    old_call,
    "crop = wait_for_clear_spawn(crop, box)",
    1
)

compile(source, str(path), "exec")

backup = path.with_name(
    f"desktop_detector_backup_before_speedup_{datetime.now():%Y%m%d_%H%M%S}.py"
)
backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
path.write_text(source, encoding="utf-8")

print(f"Backup created: {backup}")
print("Optimization applied and Python syntax verified.")
print("Changed only desktop_detector.py.")
