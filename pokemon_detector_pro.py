import os
import json
import time
import threading
import hashlib
import tkinter as tk

from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps, ImageEnhance

import torch
import open_clip
import mss

from pynput import keyboard
import pyautogui


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NAMES_FILE = os.path.join(BASE_DIR, "pokemon_names.json")
FEATURES_FILE = os.path.join(
    BASE_DIR, "pokemon_image_features.pt"
)

BOT_1_COMMAND = "@Pokétwo#8236 c "
BOT_2_COMMAND = "p!c "

MODEL_NAME = "ViT-B-32"
PRETRAINED_NAME = "laion2b_s34b_b79k"

MONITOR_INTERVAL = 1.0
MONITOR_MIN_ANALYSIS_INTERVAL = 3.0

TOP_RESULTS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimize CPU processing.
if DEVICE == "cpu":
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)


# =========================================================
# GLOBAL VARIABLES
# =========================================================

root = None

status_label = None
results_container = None
monitor_button = None

model = None
preprocess = None
tokenizer = None

pokemon_names = []
reference_features = None
reference_ids = []

monitor_region = None
monitoring = False

selection_overlay = None
selection_canvas = None

selection_start_x = 0
selection_start_y = 0
selection_rectangle = None

selection_callback = None

monitor_thread = None

analysis_lock = threading.Lock()

last_monitor_hash = None
last_analysis_time = 0

hotkey_listener = None


# =========================================================
# LOAD POKEMON NAMES
# =========================================================

def load_pokemon_names():

    global pokemon_names

    if not os.path.exists(NAMES_FILE):
        raise FileNotFoundError(
            "pokemon_names.json was not found.\n\n"
            "Keep this file in the same folder as "
            "pokemon_detector_pro.py."
        )

    with open(NAMES_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):

        pokemon_names = data

    elif isinstance(data, dict):

        # Format: {"1": "Bulbasaur", "2": "Ivysaur"}
        if all(str(key).isdigit() for key in data.keys()):

            pokemon_names = [
                data[key]
                for key in sorted(data.keys(), key=lambda x: int(x))
            ]

        # Format: {"Bulbasaur": 1, "Ivysaur": 2}
        elif all(
            isinstance(value, int)
            for value in data.values()
        ):

            pokemon_names = [
                name
                for name, number in sorted(
                    data.items(),
                    key=lambda item: item[1]
                )
            ]

        else:

            # Fall back to the original file's value order.
            pokemon_names = list(data.values())

    else:

        raise ValueError(
            "Unsupported format in pokemon_names.json."
        )

    pokemon_names = [
        str(name).replace("-", " ").replace("_", " ").strip()
        for name in pokemon_names
    ]

    print(f"Loaded {len(pokemon_names)} Pokémon names.")


# =========================================================
# LOAD AI MODEL
# =========================================================

def load_ai():

    global model, preprocess, tokenizer
    global reference_features, reference_ids

    print("Loading Pokémon database...")

    load_pokemon_names()

    print("Loading AI model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED_NAME
    )

    model = model.to(DEVICE)
    model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    print("Loading reference-image features...")

    if not os.path.exists(FEATURES_FILE):

        raise FileNotFoundError(
            "pokemon_image_features.pt was not found.\n\n"
            "Keep your existing reference-image feature file "
            "in the PokemonDetector folder."
        )

    saved_data = torch.load(
        FEATURES_FILE,
        map_location="cpu",
        weights_only=True
    )

    # New cache format preserves the real Pokémon IDs.
    if isinstance(saved_data, dict):

        if (
            "features" not in saved_data
            or "image_ids" not in saved_data
        ):

            raise ValueError(
                "The reference feature file has an "
                "unexpected format."
            )

        reference_features = saved_data["features"]
        reference_ids = saved_data["image_ids"].tolist()

    else:

        # Old cache format.
        # This assumes images are ordered by Pokédex ID
        # and that no reference images are missing.
        reference_features = saved_data

        reference_ids = list(
            range(1, len(reference_features) + 1)
        )

        print(
            "WARNING: Using an older feature cache. "
            "If reference images are missing, IDs may "
            "not match correctly."
        )

    reference_features = reference_features.float()

    reference_features = torch.nn.functional.normalize(
        reference_features,
        dim=-1
    )

    # Move reference features to the AI device once at startup.
    reference_features = reference_features.to(DEVICE)

    if len(reference_features) != len(reference_ids):

        raise ValueError(
            "The number of reference features does not "
            "match the number of reference IDs."
        )

    print(
        f"Loaded {len(reference_features)} reference images."
    )

    print(f"Device: {DEVICE}")

    print("AI is ready!")


# =========================================================
# IMAGE PREPARATION
# =========================================================

def make_image_variants(image):

    image = image.convert("RGB")

    width, height = image.size

    # First version: original image.
    variants = [image]

    # Second version: center crop to reduce text/background.
    crop_width = int(width * 0.75)
    crop_height = int(height * 0.75)

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    cropped = image.crop(
        (
            left,
            top,
            left + crop_width,
            top + crop_height
        )
    )

    variants.append(cropped)

    return variants



# =========================================================
# AI IMAGE MATCHING
# =========================================================

def analyze_pokemon_image(image):

    """
    Compare the selected image against the existing
    reference-image feature database.

    Returns a list of:
        (Pokémon name, similarity score)
    """

    if model is None or reference_features is None:

        raise RuntimeError("The AI model is not ready.")

    if not analysis_lock.acquire(blocking=False):

        print("An analysis is already running.")
        return []

    try:

        image = image.convert("RGB")

        variants = make_image_variants(image)

        image_tensors = []

        for variant in variants:

            tensor = preprocess(variant)
            image_tensors.append(tensor)

        batch = torch.stack(image_tensors).to(DEVICE)

        with torch.inference_mode():

            image_embeddings = model.encode_image(batch)

            image_embeddings = torch.nn.functional.normalize(
                image_embeddings,
                dim=-1
            )

        similarities = image_embeddings @ reference_features.T

        # Keep the highest similarity for each reference image.
        best_reference_scores = similarities.max(dim=0).values

        # Combine scores for different reference images
        # belonging to the same Pokémon.
        best_pokemon_scores = {}

        for index, pokemon_id in enumerate(reference_ids):

            if pokemon_id < 1 or pokemon_id > len(pokemon_names):
                continue

            name = pokemon_names[pokemon_id - 1]

            score = float(best_reference_scores[index].item())

            if (
                name not in best_pokemon_scores
                or score > best_pokemon_scores[name]
            ):

                best_pokemon_scores[name] = score

        results = sorted(
            best_pokemon_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return results[:TOP_RESULTS]

    finally:

        analysis_lock.release()


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(results):

    if results_container is None:
        return

    for widget in results_container.winfo_children():
        widget.destroy()

    if not results:

        status_label.config(
            text="No predictions were generated.",
            fg="#f87171"
        )

        return

    for rank, (name, score) in enumerate(results, start=1):

        row = tk.Frame(
            results_container,
            bg="#303030"
        )

        row.pack(
            fill="x",
            padx=8,
            pady=4
        )

        name_label = tk.Label(
            row,
            text=f"{rank}. {name}",
            font=("Segoe UI", 11, "bold"),
            bg="#303030",
            fg="white",
            anchor="w"
        )

        name_label.pack(
            side="left",
            fill="x",
            expand=True,
            padx=5,
            pady=7
        )

        score_label = tk.Label(
            row,
            text=f"{score * 100:.1f}%",
            font=("Segoe UI", 9),
            bg="#303030",
            fg="#93c5fd"
        )

        score_label.pack(
            side="left",
            padx=5
        )

        bot1_button = tk.Button(
            row,
            text="BOT 1",
            command=lambda n=name: copy_name(n, 1),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief="flat"
        )

        bot1_button.pack(
            side="right",
            padx=3,
            pady=4
        )

        bot2_button = tk.Button(
            row,
            text="BOT 2",
            command=lambda n=name: copy_name(n, 2),
            bg="#059669",
            fg="white",
            activebackground="#047857",
            activeforeground="white",
            relief="flat"
        )

        bot2_button.pack(
            side="right",
            padx=3,
            pady=4
        )

    status_label.config(
        text="Analysis complete. Check the top predictions.",
        fg="#4ade80"
    )


# =========================================================
# COPY POKEMON COMMAND
# =========================================================

def copy_name(name, bot_number):

    # Make the first letter lowercase.
    name = name[0].lower() + name[1:] if name else name

    if bot_number == 1:
        command = BOT_1_COMMAND + name
    else:
        command = BOT_2_COMMAND + name

    try:
        # Copy the command to the clipboard.
        root.clipboard_clear()
        root.clipboard_append(command)
        root.update()

        status_label.config(
            text=f"Copied: {command}",
            fg="#4ade80"
        )

        # Find the Discord window.
        import pygetwindow as gw

        discord_windows = gw.getWindowsWithTitle("Discord")

        if discord_windows:
            discord_window = discord_windows[0]

            # Restore Discord if it is minimized.
            if discord_window.isMinimized:
                discord_window.restore()

            # Bring Discord to the foreground.
            discord_window.activate()

        else:
            status_label.config(
                text="Command copied, but Discord was not found.",
                fg="#facc15"
            )

    except Exception as error:
        print("Copy/focus error:", error)

        # The command may already be copied even if
        # activating Discord fails.
        status_label.config(
            text="Command copied. Click Discord and press Ctrl+V.",
            fg="#facc15"
        )




# =========================================================
# ANALYZE AN IMAGE
# =========================================================

def analyze_image_async(image, source="Manual selection"):

    def worker():

        try:

            root.after(
                0,
                lambda: status_label.config(
                    text="Analyzing image...",
                    fg="#facc15"
                )
            )

            results = analyze_pokemon_image(image)

            root.after(
                0,
                lambda r=results: display_results(r)
            )

        except Exception as error:

            error_text = str(error)

            print("Analysis error:", error_text)

            root.after(
                0,
                lambda e=error_text: status_label.config(
                    text=f"Analysis failed: {e[:70]}",
                    fg="#f87171"
                )
            )

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


# =========================================================
# SCREENSHOT SELECTION
# =========================================================

def start_selection():

    begin_screen_selection(
        callback=manual_selection_finished,
        title="Select the Pokémon image"
    )


def start_monitor_selection():

    begin_screen_selection(
        callback=monitor_selection_finished,
        title="Select the area to monitor"
    )


def begin_screen_selection(callback, title):

    global selection_overlay
    global selection_canvas
    global selection_callback
    global selection_rectangle
    global selection_start_x, selection_start_y

    if selection_overlay is not None:

        return

    selection_callback = callback

    selection_rectangle = None

    selection_start_x = 0
    selection_start_y = 0

    try:

        with mss.mss() as screenshotter:

            monitor = screenshotter.monitors[1]

            screen_width = monitor["width"]
            screen_height = monitor["height"]

    except Exception as error:

        messagebox.showerror(
            "Screenshot error",
            str(error)
        )

        return

    selection_overlay = tk.Toplevel(root)

    selection_overlay.title(title)

    selection_overlay.attributes("-topmost", True)
    selection_overlay.attributes("-fullscreen", True)
    selection_overlay.attributes("-alpha", 0.30)

    selection_overlay.configure(bg="black")

    selection_canvas = tk.Canvas(
        selection_overlay,
        bg="black",
        cursor="cross",
        highlightthickness=0
    )

    selection_canvas.pack(
        fill="both",
        expand=True
    )

    selection_canvas.create_text(
        screen_width // 2,
        35,
        text=(
            title
            + " — drag to select; press ESC to cancel"
        ),
        fill="white",
        font=("Segoe UI", 15, "bold")
    )

    selection_canvas.bind(
        "<ButtonPress-1>",
        selection_mouse_down
    )

    selection_canvas.bind(
        "<B1-Motion>",
        selection_mouse_move
    )

    selection_canvas.bind(
        "<ButtonRelease-1>",
        selection_mouse_up
    )

    selection_overlay.bind(
        "<Escape>",
        cancel_selection
    )

    selection_overlay.focus_force()


def selection_mouse_down(event):

    global selection_start_x, selection_start_y
    global selection_rectangle

    selection_start_x = event.x
    selection_start_y = event.y

    if selection_rectangle is not None:

        selection_canvas.delete(selection_rectangle)

    selection_rectangle = selection_canvas.create_rectangle(
        event.x,
        event.y,
        event.x,
        event.y,
        outline="#22c55e",
        width=3
    )


def selection_mouse_move(event):

    if selection_rectangle is None:
        return

    selection_canvas.coords(
        selection_rectangle,
        selection_start_x,
        selection_start_y,
        event.x,
        event.y
    )


def selection_mouse_up(event):

    global selection_overlay
    global selection_canvas
    global selection_callback

    x1 = min(selection_start_x, event.x)
    y1 = min(selection_start_y, event.y)

    x2 = max(selection_start_x, event.x)
    y2 = max(selection_start_y, event.y)

    callback = selection_callback

    selection_overlay.destroy()

    selection_overlay = None
    selection_canvas = None
    selection_callback = None

    if x2 - x1 < 20 or y2 - y1 < 20:

        status_label.config(
            text="Selection too small. Try again.",
            fg="#f87171"
        )

        return

    try:

        with mss.mss() as screenshotter:

            monitor = screenshotter.monitors[1]

            screenshot = screenshotter.grab(
                {
                    "left": monitor["left"] + x1,
                    "top": monitor["top"] + y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            )

            image = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.rgb
            )

        if callback:

            callback(
                image,
                {
                    "left": monitor["left"] + x1,
                    "top": monitor["top"] + y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            )

    except Exception as error:

        messagebox.showerror(
            "Screenshot error",
            str(error)
        )


def cancel_selection(event=None):

    global selection_overlay
    global selection_canvas
    global selection_callback

    if selection_overlay is not None:

        selection_overlay.destroy()

    selection_overlay = None
    selection_canvas = None
    selection_callback = None

    status_label.config(
        text="Selection cancelled.",
        fg="#facc15"
    )


# =========================================================
# MANUAL SELECTION
# =========================================================

def manual_selection_finished(image, region):

    status_label.config(
        text="Image selected. Starting recognition...",
        fg="#facc15"
    )

    analyze_image_async(
        image,
        source="Manual selection"
    )


# =========================================================
# AUTOMATIC MONITORING
# =========================================================

def monitor_selection_finished(image, region):

    global monitor_region
    global monitoring
    global last_monitor_hash
    global last_analysis_time

    monitor_region = region

    last_monitor_hash = None
    last_analysis_time = 0

    monitoring = True

    monitor_button.config(
        text="STOP MONITORING",
        bg="#dc2626"
    )

    status_label.config(
        text="Monitoring selected area...",
        fg="#4ade80"
    )

    start_monitor_thread()


def start_monitor_thread():

    global monitor_thread

    if (
        monitor_thread is not None
        and monitor_thread.is_alive()
    ):

        return

    monitor_thread = threading.Thread(
        target=monitor_loop,
        daemon=True
    )

    monitor_thread.start()


def image_hash(image):

    small = image.convert("L").resize((32, 32))

    return hashlib.md5(
        small.tobytes()
    ).hexdigest()


def monitor_loop():

    global last_monitor_hash
    global last_analysis_time

    while monitoring:

        try:

            if monitor_region is None:

                time.sleep(MONITOR_INTERVAL)
                continue

            with mss.mss() as screenshotter:

                screenshot = screenshotter.grab(
                    monitor_region
                )

                image = Image.frombytes(
                    "RGB",
                    screenshot.size,
                    screenshot.rgb
                )

            current_hash = image_hash(image)

            current_time = time.time()

            image_changed = (
                current_hash != last_monitor_hash
            )

            enough_time_passed = (
                current_time - last_analysis_time
                >= MONITOR_MIN_ANALYSIS_INTERVAL
            )

            if image_changed and enough_time_passed:

                last_monitor_hash = current_hash
                last_analysis_time = current_time

                # Skip extremely small or blank images.
                if image.width >= 40 and image.height >= 40:

                    root.after(
                        0,
                        lambda: status_label.config(
                            text="New image detected. Analyzing...",
                            fg="#facc15"
                        )
                    )

                    results = analyze_pokemon_image(image)

                    root.after(
                        0,
                        lambda r=results: display_results(r)
                    )

            time.sleep(MONITOR_INTERVAL)

        except Exception as error:

            print("Monitoring error:", error)

            time.sleep(2)


def stop_monitoring():

    global monitoring
    global monitor_region
    global last_monitor_hash

    monitoring = False

    monitor_region = None
    last_monitor_hash = None

    monitor_button.config(
        text="START MONITORING (F9)",
        bg="#2563eb"
    )

    status_label.config(
        text="Monitoring stopped.",
        fg="#facc15"
    )


def toggle_monitoring():

    if monitoring:

        stop_monitoring()

    else:

        start_monitor_selection()


# =========================================================
# HOTKEYS
# =========================================================

def start_hotkey_listener():

    def on_press(key):

        try:

            if key == keyboard.Key.f8:

                if root:

                    root.after(0, start_selection)

            elif key == keyboard.Key.f9:

                if root:

                    root.after(0, toggle_monitoring)

        except Exception as error:

            print("Hotkey error:", error)

    global hotkey_listener

    hotkey_listener = keyboard.Listener(
        on_press=on_press
    )

    hotkey_listener.daemon = True
    hotkey_listener.start()


# =========================================================
# MAIN WINDOW
# =========================================================

def create_main_window():

    global root
    global status_label
    global results_container
    global monitor_button

    root = tk.Tk()

    root.title("Pokémon Detector Pro")

    root.geometry("480x600")
    root.minsize(420, 450)

    root.configure(bg="#202020")

    title_label = tk.Label(
        root,
        text="POKÉMON DETECTOR PRO",
        font=("Segoe UI", 16, "bold"),
        bg="#202020",
        fg="white"
    )

    title_label.pack(
        pady=(12, 5)
    )

    subtitle = tk.Label(
        root,
        text="AI-powered image recognition",
        font=("Segoe UI", 9),
        bg="#202020",
        fg="#9ca3af"
    )

    subtitle.pack(
        pady=(0, 10)
    )

    button_frame = tk.Frame(
        root,
        bg="#202020"
    )

    button_frame.pack(
        fill="x",
        padx=12,
        pady=5
    )

    manual_button = tk.Button(
        button_frame,
        text="SELECT POKÉMON (F8)",
        command=start_selection,
        font=("Segoe UI", 10, "bold"),
        bg="#2563eb",
        fg="white",
        activebackground="#1d4ed8",
        activeforeground="white",
        relief="flat",
        height=2
    )

    manual_button.pack(
        fill="x",
        pady=3
    )

    monitor_button = tk.Button(
        button_frame,
        text="START MONITORING (F9)",
        command=toggle_monitoring,
        font=("Segoe UI", 10, "bold"),
        bg="#2563eb",
        fg="white",
        activebackground="#1d4ed8",
        activeforeground="white",
        relief="flat",
        height=2
    )

    monitor_button.pack(
        fill="x",
        pady=3
    )

    status_label = tk.Label(
        root,
        text="Loading AI model...",
        font=("Segoe UI", 9),
        bg="#202020",
        fg="#facc15",
        wraplength=440
    )

    status_label.pack(
        fill="x",
        padx=12,
        pady=10
    )

    separator = tk.Frame(
        root,
        height=1,
        bg="#444444"
    )

    separator.pack(
        fill="x",
        padx=12,
        pady=3
    )

    results_title = tk.Label(
        root,
        text="TOP 5 PREDICTIONS",
        font=("Segoe UI", 11, "bold"),
        bg="#202020",
        fg="white"
    )

    results_title.pack(
        anchor="w",
        padx=12,
        pady=(8, 4)
    )

    results_container = tk.Frame(
        root,
        bg="#202020"
    )

    results_container.pack(
        fill="both",
        expand=True,
        padx=8,
        pady=4
    )

    instructions = tk.Label(
        root,
        text=(
            "F8: Select an image    "
            "F9: Monitor an area\n"
            "BOT 1 / BOT 2: Copy Pokémon commands"
        ),
        font=("Segoe UI", 8),
        bg="#202020",
        fg="#9ca3af",
        justify="center"
    )

    instructions.pack(
        fill="x",
        pady=(5, 10)
    )

    root.protocol(
        "WM_DELETE_WINDOW",
        close_program
    )


# =========================================================
# CLOSE PROGRAM
# =========================================================

def close_program():

    global monitoring

    monitoring = False

    try:

        if hotkey_listener:

            hotkey_listener.stop()

    except Exception:

        pass

    try:

        root.destroy()

    except Exception:

        pass


# =========================================================
# START PROGRAM
# =========================================================

def main():

    create_main_window()

    def initialize():

        try:

            load_ai()

            root.after(
                0,
                lambda: status_label.config(
                    text="AI is ready! Press F8 to begin.",
                    fg="#4ade80"
                )
            )

        except Exception as error:

            error_text = str(error)

            print("Startup error:", error_text)

            root.after(
                0,
                lambda e=error_text: status_label.config(
                    text=f"Startup error: {e[:100]}",
                    fg="#f87171"
                )
            )

            root.after(
                0,
                lambda e=error_text: messagebox.showerror(
                    "Pokémon Detector Error",
                    e
                )
            )

    threading.Thread(
        target=initialize,
        daemon=True
    ).start()

    start_hotkey_listener()

    root.mainloop()


if __name__ == "__main__":

    main()