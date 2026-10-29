import os
import json
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import torch
import open_clip
import mss

from PIL import Image, ImageTk, ImageDraw
from pynput import keyboard


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NAMES_FILE = os.path.join(BASE_DIR, "pokemon_names.json")
IMAGES_DIR = os.path.join(BASE_DIR, "pokemon_images")
FEATURES_FILE = os.path.join(
    BASE_DIR, "pokemon_image_features.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

TOP_RESULTS = 5


# =========================================================
# GLOBAL VARIABLES
# =========================================================

model = None
preprocess = None
tokenizer = None

pokemon_names = []
text_features = None
image_features = None
image_ids = []

root = None
status_label = None
results_frame = None
results_container = None

overlay = None
canvas = None
screen_image = None
canvas_image = None
selection_rectangle = None

start_x = 0
start_y = 0

hotkey_listener = None
busy = False


# =========================================================
# LOAD DATABASE AND AI MODEL
# =========================================================

def load_database():
    global pokemon_names

    if not os.path.exists(NAMES_FILE):
        raise FileNotFoundError(
            "pokemon_names.json was not found."
        )

    with open(NAMES_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        pokemon_names = list(data.values())
    else:
        pokemon_names = data

    pokemon_names = [
        str(name).replace("-", " ").strip()
        for name in pokemon_names
    ]


def load_ai():
    global model, preprocess, tokenizer
    global text_features, image_features, image_ids

    print("Loading Pokémon database...")
    load_database()

    print("Loading AI model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    # -----------------------------------------------------
    # Prepare text features
    # -----------------------------------------------------

    print("Preparing Pokémon name recognition...")

    prompts = [
        f"a picture of the Pokémon {name}"
        for name in pokemon_names
    ]

    feature_batches = []

    with torch.no_grad():
        for start in range(0, len(prompts), 64):
            batch = prompts[start:start + 64]

            tokens = tokenizer(batch).to(DEVICE)

            features = model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

            feature_batches.append(features.cpu())

    text_features = torch.cat(feature_batches, dim=0)

    # -----------------------------------------------------
    # Load saved reference-image features
    # -----------------------------------------------------

    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(
            "pokemon_image_features.pt was not found.\n\n"
            "Run pokemon_image_matcher.py first."
        )

    print("Loading reference-image features...")

    image_features = torch.load(
        FEATURES_FILE,
        map_location="cpu",
        weights_only=True
    )

    # The existing image matcher created this cache using
    # numbered files from 1.png through 1025.png.
    image_ids = list(range(1, len(image_features) + 1))

    print("AI is ready!")


# =========================================================
# CREATE MAIN WINDOW
# =========================================================

def create_main_window():
    global root, status_label, results_frame, results_container

    root = tk.Tk()

    root.title("Pokémon Detector Pro")
    root.geometry("410x510+100+100")
    root.attributes("-topmost", True)

    root.configure(bg="#20242c")

    title = tk.Label(
        root,
        text="POKÉMON DETECTOR",
        font=("Segoe UI", 17, "bold"),
        fg="white",
        bg="#20242c"
    )
    title.pack(pady=(15, 5))

    subtitle = tk.Label(
        root,
        text="AI recognition + reference-image matching",
        font=("Segoe UI", 9),
        fg="#c8d0dc",
        bg="#20242c"
    )
    subtitle.pack(pady=(0, 12))

    select_button = tk.Button(
        root,
        text="SELECT POKÉMON IMAGE  (F8)",
        font=("Segoe UI", 11, "bold"),
        command=start_selection,
        bg="#3b82f6",
        fg="white",
        activebackground="#2563eb",
        activeforeground="white",
        relief="flat",
        padx=12,
        pady=10
    )
    select_button.pack(fill="x", padx=18, pady=5)

    status_label = tk.Label(
        root,
        text="Loading AI...",
        font=("Segoe UI", 9),
        fg="#fbbf24",
        bg="#20242c",
        wraplength=370
    )
    status_label.pack(pady=10)

    separator = ttk.Separator(root, orient="horizontal")
    separator.pack(fill="x", padx=15, pady=5)

    results_title = tk.Label(
        root,
        text="TOP PREDICTIONS",
        font=("Segoe UI", 11, "bold"),
        fg="white",
        bg="#20242c"
    )
    results_title.pack(pady=5)

    results_frame = tk.Frame(root, bg="#20242c")
    results_frame.pack(fill="both", expand=True, padx=15, pady=5)

    results_container = results_frame

    instructions = tk.Label(
        root,
        text="F8 = Select image   |   Esc = Cancel selection",
        font=("Segoe UI", 8),
        fg="#aeb8c7",
        bg="#20242c"
    )
    instructions.pack(pady=(3, 10))

    root.protocol("WM_DELETE_WINDOW", close_program)

    # Start global keyboard listener
    start_hotkey_listener()

    root.after(1000, finish_loading)

    return root


# =========================================================
# LOAD MODEL WITHOUT FREEZING THE WINDOW
# =========================================================

def finish_loading():
    status_label.config(
        text="AI is loading in the background...",
        fg="#fbbf24"
    )

    def worker():
        try:
            load_ai()

            root.after(
                0,
                lambda: status_label.config(
                    text="Ready! Press F8 or click the button.",
                    fg="#4ade80"
                )
            )

        except Exception as error:
            message = str(error)

            root.after(
                0,
                lambda: status_label.config(
                    text="Loading failed. Check the terminal.",
                    fg="#f87171"
                )
            )

            print("\nERROR WHILE LOADING AI:")
            print(message)

            root.after(
                0,
                lambda: messagebox.showerror(
                    "AI Loading Error",
                    message
                )
            )

    threading.Thread(target=worker, daemon=True).start()


# =========================================================
# GLOBAL F8 SHORTCUT
# =========================================================

def start_hotkey_listener():
    global hotkey_listener

    def on_press(key):
        try:
            if key == keyboard.Key.f8:
                if root:
                    root.after(0, start_selection)
        except Exception:
            pass

    hotkey_listener = keyboard.Listener(on_press=on_press)
    hotkey_listener.daemon = True
    hotkey_listener.start()


# =========================================================
# SCREENSHOT SELECTION
# =========================================================

def start_selection():
    global overlay, canvas, screen_image, canvas_image
    global selection_rectangle, start_x, start_y

    if busy:
        return

    if model is None:
        status_label.config(
            text="Please wait for the AI to finish loading.",
            fg="#fbbf24"
        )
        return

    try:
        root.withdraw()
        root.update()

        with mss.mss() as screenshot_tool:
            monitor = screenshot_tool.monitors[1]
            screenshot = screenshot_tool.grab(monitor)

            screen_image = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.rgb
            )

        # Darken the screenshot so the selection area is visible.
        display_image = screen_image.copy()

        dark_overlay = Image.new(
            "RGB",
            display_image.size,
            (0, 0, 0)
        )

        display_image = Image.blend(
            display_image,
            dark_overlay,
            0.30
        )

        overlay = tk.Toplevel(root)
        overlay.attributes("-fullscreen", True)
        overlay.attributes("-topmost", True)
        overlay.overrideredirect(True)

        canvas = tk.Canvas(
            overlay,
            cursor="crosshair",
            highlightthickness=0
        )
        canvas.pack(fill="both", expand=True)

        canvas_image = ImageTk.PhotoImage(display_image)

        canvas.create_image(
            0,
            0,
            image=canvas_image,
            anchor="nw"
        )

        canvas.create_text(
            25,
            25,
            text="Drag around the Pokémon image. Press ESC to cancel.",
            anchor="nw",
            fill="white",
            font=("Segoe UI", 16, "bold")
        )

        selection_rectangle = None

        canvas.bind("<ButtonPress-1>", selection_start)
        canvas.bind("<B1-Motion>", selection_move)
        canvas.bind("<ButtonRelease-1>", selection_finish)

        overlay.bind("<Escape>", cancel_selection)
        overlay.focus_force()

    except Exception as error:
        if overlay:
            overlay.destroy()

        overlay = None

        root.deiconify()

        messagebox.showerror(
            "Selection Error",
            str(error)
        )


def selection_start(event):
    global start_x, start_y, selection_rectangle

    start_x = event.x
    start_y = event.y

    if selection_rectangle is not None:
        canvas.delete(selection_rectangle)

    selection_rectangle = canvas.create_rectangle(
        start_x,
        start_y,
        start_x,
        start_y,
        outline="#00ff66",
        width=3
    )


def selection_move(event):
    if selection_rectangle is not None:
        canvas.coords(
            selection_rectangle,
            start_x,
            start_y,
            event.x,
            event.y
        )


def selection_finish(event):
    global overlay, screen_image

    x1 = min(start_x, event.x)
    y1 = min(start_y, event.y)
    x2 = max(start_x, event.x)
    y2 = max(start_y, event.y)

    if x2 - x1 < 20 or y2 - y1 < 20:
        cancel_selection()
        return

    selected_image = screen_image.crop((x1, y1, x2, y2))

    image_path = os.path.join(
        BASE_DIR,
        "selected_pokemon.png"
    )

    selected_image.save(image_path)

    overlay.destroy()
    overlay = None

    root.deiconify()
    root.lift()

    status_label.config(
        text="Image selected! Analyzing...",
        fg="#fbbf24"
    )

    analyze_selected_image(image_path)


def cancel_selection(event=None):
    global overlay

    if overlay is not None:
        overlay.destroy()
        overlay = None

    root.deiconify()
    root.lift()

    status_label.config(
        text="Selection cancelled. Press F8 to try again.",
        fg="#c8d0dc"
    )


# =========================================================
# ANALYZE IMAGE USING BOTH METHODS
# =========================================================

def analyze_selected_image(image_path):
    global busy

    if busy:
        return

    busy = True

    for widget in results_container.winfo_children():
        widget.destroy()

    status_label.config(
        text="Analyzing image with two AI methods...",
        fg="#fbbf24"
    )

    def worker():
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(DEVICE)

            # ---------------------------------------------
            # Method 1: Compare with Pokémon names
            # ---------------------------------------------

            with torch.no_grad():
                query_features = model.encode_image(image_input)
                query_features /= query_features.norm(
                    dim=-1,
                    keepdim=True
                )

                text_scores = (
                    query_features.cpu() @ text_features.T
                ).squeeze(0)

                # -----------------------------------------
                # Method 2: Compare with reference artwork
                # -----------------------------------------

                image_scores = (
                    query_features.cpu() @ image_features.T
                ).squeeze(0)

            # ---------------------------------------------
            # Use image matching as the primary method
            # ---------------------------------------------

            combined_scores = {}

            # Rank Pokémon by reference-image similarity.
            image_order = torch.argsort(
                image_scores,
                descending=True
            ).tolist()

            # Add the top image matches first.
            for rank, position in enumerate(image_order, start=1):
                pokemon_id = image_ids[position]

                if 1 <= pokemon_id <= len(pokemon_names):
                    name = pokemon_names[pokemon_id - 1]

                    combined_scores[name] = (
                        1.0 / (60 + rank)
                    )

            # Add text-recognition results only as a small bonus.
            text_order = torch.argsort(
                text_scores,
                descending=True
            ).tolist()

            for rank, index in enumerate(text_order, start=1):
                name = pokemon_names[index]

                combined_scores[name] = (
                    combined_scores.get(name, 0)
                    + 0.05 / (60 + rank)
                )

            ranked_results = sorted(
                combined_scores.items(),
                key=lambda item: item[1],
                reverse=True
            )[:TOP_RESULTS]

            root.after(
                0,
                lambda: display_results(ranked_results)
            )

        except Exception as error:
            message = str(error)

            root.after(
                0,
                lambda: status_label.config(
                    text="Recognition failed. Check the terminal.",
                    fg="#f87171"
                )
            )

            print("\nRECOGNITION ERROR:")
            print(message)

        finally:
            root.after(0, finish_analysis)

    threading.Thread(target=worker, daemon=True).start()


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(results):
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
            bg="#303743",
            padx=8,
            pady=7
        )

        row.pack(fill="x", pady=3)

        name_label = tk.Label(
            row,
            text=f"{rank}. {name}",
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg="#303743",
            anchor="w"
        )

        name_label.pack(
            side="left",
            fill="x",
            expand=True
        )

        copy_button = tk.Button(
            row,
            text="COPY",
            font=("Segoe UI", 8, "bold"),
            command=lambda pokemon=name: copy_name(pokemon),
            bg="#475569",
            fg="white",
            relief="flat"
        )

        copy_button.pack(side="right")

    status_label.config(
        text="Analysis complete! Choose a name to copy it.",
        fg="#4ade80"
    )


def copy_name(name):
    root.clipboard_clear()
    root.clipboard_append(name)
    root.update()

    status_label.config(
        text=f"Copied: {name}",
        fg="#4ade80"
    )


def finish_analysis():
    global busy
    busy = False


# =========================================================
# CLOSE PROGRAM
# =========================================================

def close_program():
    global hotkey_listener

    try:
        if hotkey_listener:
            hotkey_listener.stop()
    except Exception:
        pass

    try:
        if overlay:
            overlay.destroy()
    except Exception:
        pass

    root.destroy()


# =========================================================
# START PROGRAM
# =========================================================

if __name__ == "__main__":
    create_main_window()
    root.mainloop()