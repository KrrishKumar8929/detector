import tkinter as tk
from tkinter import ttk
import mss
from PIL import Image, ImageTk


class PokemonDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokémon Detector")
        self.root.geometry("450x400+100+100")
        self.root.attributes("-topmost", True)

        self.selected_image = None
        self.photo = None

        title = ttk.Label(
            root,
            text="Pokémon Detector",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        instructions = ttk.Label(
            root,
            text="Select a Pokémon image from your screen."
        )
        instructions.pack(pady=5)

        self.image_label = ttk.Label(
            root,
            text="No Pokémon image selected yet"
        )
        self.image_label.pack(pady=10)

        self.status_label = ttk.Label(
            root,
            text="Ready"
        )
        self.status_label.pack(pady=5)

        self.select_button = ttk.Button(
            root,
            text="Select Pokémon Image",
            command=self.start_selection
        )
        self.select_button.pack(pady=10)

        self.save_button = ttk.Button(
            root,
            text="Save Selected Image",
            command=self.save_image,
            state="disabled"
        )
        self.save_button.pack(pady=5)

    def start_selection(self):
        try:
            # Capture the screen before opening the selection overlay.
            with mss.mss() as screen:
                self.monitor = screen.monitors[1]
                shot = screen.grab(self.monitor)

                self.screen_image = Image.frombytes(
                    "RGB",
                    shot.size,
                    shot.rgb
                )

            self.root.withdraw()

            self.overlay = tk.Toplevel()
            self.overlay.attributes("-fullscreen", True)
            self.overlay.attributes("-topmost", True)
            self.overlay.attributes("-alpha", 0.25)
            self.overlay.configure(bg="black")
            self.overlay.config(cursor="cross")

            self.canvas = tk.Canvas(
                self.overlay,
                bg="black",
                cursor="cross",
                highlightthickness=0
            )
            self.canvas.pack(fill="both", expand=True)

            self.start_x = None
            self.start_y = None
            self.rectangle = None

            self.canvas.bind(
                "<ButtonPress-1>",
                self.begin_drag
            )
            self.canvas.bind(
                "<B1-Motion>",
                self.drag
            )
            self.canvas.bind(
                "<ButtonRelease-1>",
                self.finish_selection
            )

            self.overlay.bind(
                "<Escape>",
                self.cancel_selection
            )

            self.status_label.configure(
                text="Drag around the Pokémon. Press Esc to cancel."
            )

        except Exception as error:
            self.root.deiconify()
            self.status_label.configure(
                text=f"Error: {error}"
            )

    def begin_drag(self, event):
        self.start_x = event.x
        self.start_y = event.y

        if self.rectangle:
            self.canvas.delete(self.rectangle)

        self.rectangle = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            event.x,
            event.y,
            outline="red",
            width=3
        )

    def drag(self, event):
        if self.rectangle:
            self.canvas.coords(
                self.rectangle,
                self.start_x,
                self.start_y,
                event.x,
                event.y
            )

    def finish_selection(self, event):
        if self.start_x is None or self.start_y is None:
            self.cancel_selection()
            return

        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)

        # Convert screen coordinates to screenshot coordinates.
        left = int(x1)
        top = int(y1)
        right = int(x2)
        bottom = int(y2)

        self.overlay.destroy()
        self.root.deiconify()
        self.root.attributes("-topmost", True)

        if right - left < 10 or bottom - top < 10:
            self.status_label.configure(
                text="Selection too small. Please try again."
            )
            return

        self.selected_image = self.screen_image.crop(
            (left, top, right, bottom)
        )

        preview = self.selected_image.copy()
        preview.thumbnail((400, 240))

        self.photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(
            image=self.photo,
            text=""
        )

        self.save_button.configure(state="normal")

        self.status_label.configure(
            text="Image selected successfully!"
        )

    def cancel_selection(self, event=None):
        if hasattr(self, "overlay") and self.overlay.winfo_exists():
            self.overlay.destroy()

        self.root.deiconify()
        self.root.attributes("-topmost", True)

        self.status_label.configure(
            text="Selection cancelled."
        )

    def save_image(self):
        if self.selected_image is None:
            self.status_label.configure(
                text="Select an image first."
            )
            return

        try:
            self.selected_image.save("selected_pokemon.png")

            self.status_label.configure(
                text="Saved as selected_pokemon.png"
            )

        except Exception as error:
            self.status_label.configure(
                text=f"Save error: {error}"
            )


if __name__ == "__main__":
    window = tk.Tk()
    app = PokemonDetector(window)
    window.mainloop()