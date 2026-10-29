from pathlib import Path

p = Path("desktop_detector.py")
s = p.read_text(encoding="utf-8")

old = '''def start_monitoring():
    global monitoring, worker, last_spawn_key

    if monitoring:
        return

    if model is None:
        messagebox.showerror(
            "AI not ready",
            "Wait until the AI finishes loading."
        )
        return

    last_spawn_key = None
    monitoring = True

    start_button.config(state="disabled")
    stop_button.config(state="normal")

    worker = threading.Thread(
        target=monitor_loop,
        daemon=True
    )

    worker.start()


def stop_monitoring():
    global monitoring

    monitoring = False

    start_button.config(state="normal")
    stop_button.config(state="disabled")
'''

new = '''def start_monitoring():
    global monitoring, worker, last_spawn_key

    if monitoring:
        return

    if worker is not None and worker.is_alive():
        update_status("Waiting for the previous monitoring thread to stop...")
        start_button.config(state="disabled")
        return

    if model is None:
        messagebox.showerror(
            "AI not ready",
            "Wait until the AI finishes loading."
        )
        return

    last_spawn_key = None
    monitoring = True

    start_button.config(state="disabled")
    stop_button.config(state="normal")

    worker = threading.Thread(
        target=monitor_loop,
        daemon=True
    )

    worker.start()


def stop_monitoring():
    global monitoring

    monitoring = False

    start_button.config(state="disabled")
    stop_button.config(state="disabled")
    update_status("Stopping monitoring...")

    root.after(100, finish_stopping)


def finish_stopping():
    if worker is not None and worker.is_alive():
        root.after(100, finish_stopping)
        return

    start_button.config(state="normal")
    stop_button.config(state="disabled")
    update_status("Monitoring stopped.")
'''

if old not in s:
    raise SystemExit("STOP: Expected functions not found. No changes made.")

p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("Thread restart fix applied.")
