"""Local Pokémon recognition service for the Vivaldi extension.
Uses the existing C:\\PokemonDetector database and catch-name map read-only.
"""
from __future__ import annotations
import io, json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(r"C:\PokemonDetector")
DATABASE_FILE = BASE_DIR / "poketwo_full_reference_features.pt"
METADATA_FILE = BASE_DIR / "poketwo_full_feature_metadata.json"
CATCH_NAME_MAP_FILE = BASE_DIR / "pokemon_catch_name_map.json"
HOST = "127.0.0.1"
PORT = 8765
MAX_IMAGE_BYTES = 12 * 1024 * 1024

_model_lock = threading.Lock()
_loaded = False
_load_error = None
try:
    import torch
    import open_clip
    from PIL import Image
except Exception as exc:
    torch = None
    open_clip = None
    Image = None
    _load_error = f"Python dependency import failed: {exc}"

model = preprocess = reference_features = reference_names = catch_name_map = None
device = "cpu"


def load_model():
    global _loaded, _load_error, model, preprocess, reference_features, reference_names, catch_name_map, device
    with _model_lock:
        if _loaded:
            return
        if _load_error:
            raise RuntimeError(_load_error)
        for path in (DATABASE_FILE, CATCH_NAME_MAP_FILE):
            if not path.is_file():
                raise RuntimeError(f"Required existing file not found: {path}")
        try:
            with CATCH_NAME_MAP_FILE.open("r", encoding="utf-8") as f:
                raw_map = json.load(f)
            catch_name_map = {
                str(folder).strip().lower(): str(entry.get("catch_name", folder)).strip().lower()
                for folder, entry in raw_map.items() if isinstance(entry, dict)
            }
            database = torch.load(str(DATABASE_FILE), map_location="cpu", weights_only=False)
            reference_features = database["features"].float()
            reference_ids = database["image_ids"].long()
            image_paths = database["image_paths"]
            reference_names = [
                os.path.basename(os.path.dirname(str(path))).replace("_", " ").strip().lower()
                for path in image_paths
            ]
            if len(reference_names) != reference_features.shape[0]:
                raise RuntimeError(
                    "Database mismatch: image_paths count does not match features count "
                    f"({len(reference_names)} vs {reference_features.shape[0]})."
                )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model = model.to(device)
            model.eval()
            reference_features = reference_features / reference_features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)
            _loaded = True
            print(f"Loaded {len(reference_names):,} reference features on {device}.")
        except Exception as exc:
            _load_error = f"Model initialization failed: {type(exc).__name__}: {exc}"
            raise RuntimeError(_load_error) from exc


def recognize_image(image):
    load_model()
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        query = model.encode_image(image_tensor)
        query = query / query.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        scores = (query.cpu() @ reference_features.T).squeeze(0)
    count = min(100, scores.numel())
    top_scores, top_indices = scores.topk(count)
    best_by_name = {}
    for score, index in zip(top_scores.tolist(), top_indices.tolist()):
        name = reference_names[index]
        if name in ("nidoranf", "nidoranm"):
            name = "nidoran"
        if name not in best_by_name or score > best_by_name[name]:
            best_by_name[name] = score
    ranked = sorted(best_by_name.items(), key=lambda item: item[1], reverse=True)[:5]
    return [
        {"reference_name": name, "catch_name": catch_name_map.get(name.strip().lower(), name.strip().lower()), "score": float(score)}
        for name, score in ranked
    ]


class Handler(BaseHTTPRequestHandler):
    server_version = "PokemonDetectorLocal/1.0"
    def _send(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)
    def do_OPTIONS(self):
        self._send(204, {})
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "loaded": _loaded, "device": device, "error": _load_error})
        else:
            self._send(404, {"error": "Not found"})
    def do_POST(self):
        if self.path != "/recognize":
            return self._send(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_IMAGE_BYTES:
                return self._send(413, {"error": f"Image must be between 1 byte and {MAX_IMAGE_BYTES} bytes."})
            content_type = self.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/") and "application/octet-stream" not in content_type:
                return self._send(415, {"error": "Expected image/* or application/octet-stream content type."})
            body = self.rfile.read(length)
            if len(body) != length:
                return self._send(400, {"error": "Incomplete request body."})
            load_model()
            with Image.open(io.BytesIO(body)) as img:
                img.verify()
            with Image.open(io.BytesIO(body)) as img:
                results = recognize_image(img)
            self._send(200, {"ok": True, "results": results})
        except Exception as exc:
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    def log_message(self, fmt, *args):
        print("[HTTP] " + (fmt % args))

if __name__ == "__main__":
    print(f"Pokémon Detector service listening only on http://{HOST}:{PORT}")
    print("Existing reference database and catch-name map are read-only inputs.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
