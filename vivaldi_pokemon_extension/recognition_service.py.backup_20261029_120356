"""Dual-database local Pokémon recognition service for the Vivaldi extension.

Reads, but never changes, the user's full-spawn references and official PNG art.
Feature caches are written as separate generated files under C:\\PokemonDetector.
"""
from __future__ import annotations
import io, json, os, re, threading, hashlib, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(r"C:\PokemonDetector")
FULL_DB_FILE = BASE_DIR / "poketwo_full_reference_features.pt"
CATCH_NAME_MAP_FILE = BASE_DIR / "pokemon_catch_name_map.json"
POKEMON_NAMES_FILE = BASE_DIR / "pokemon_names.json"
CROPPED_DIR = BASE_DIR / "poketwo_official_data" / "images"
CROPPED_CACHE_FILE = BASE_DIR / "poketwo_official_crop_features.pt"
CROPPED_CACHE_META = BASE_DIR / "poketwo_official_crop_features_meta.json"
HOST, PORT = "127.0.0.1", 8765
MAX_IMAGE_BYTES = 12 * 1024 * 1024
SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}

_lock = threading.RLock()
_loaded = False
_load_error = None
_segmenter = None
_segmenter_checked = False
segmentation_available = False
segmentation_error = None
model = preprocess = full_features = full_names = catch_name_map = None
crop_features = crop_names = None
device = "cpu"

try:
    import torch
    import open_clip
    from PIL import Image, ImageFile, ImageOps
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception as exc:
    torch = open_clip = Image = ImageOps = None
    _load_error = f"Python dependency import failed: {type(exc).__name__}: {exc}"


def _norm_name(value: str) -> str:
    value = str(value).lower().strip().replace("♀", "f").replace("♂", "m")
    value = value.replace("farfetch'd", "farfetchd").replace("mr. mime", "mr mime")
    return re.sub(r"[^a-z0-9]+", "", value)


def _open_rgb(path_or_image):
    if isinstance(path_or_image, Image.Image):
        im = path_or_image.copy()
    else:
        im = Image.open(path_or_image)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return im.convert("RGB")


def _prepare_subject(image):
    """Return an isolated subject crop when rembg is available; otherwise None."""
    global _segmenter, _segmenter_checked, segmentation_available, segmentation_error
    if not _segmenter_checked:
        _segmenter_checked = True
        try:
            from rembg import new_session, remove
            _segmenter = (new_session("u2net"), remove)
            segmentation_available = True
            print("Background segmentation enabled (rembg/u2net).")
        except Exception as exc:
            segmentation_available = False
            segmentation_error = f"rembg unavailable: {type(exc).__name__}: {exc}"
            print("Background segmentation is not installed; using full-image fallback.")
    if not _segmenter:
        return None
    try:
        session, remove = _segmenter
        rgba = image.convert("RGBA")
        result = remove(rgba, session=session)
        if isinstance(result, bytes):
            result = Image.open(io.BytesIO(result)).convert("RGBA")
        else:
            result = result.convert("RGBA")
        alpha = result.getchannel("A")
        bbox = alpha.getbbox()
        if not bbox:
            return None
        # Ignore tiny alpha noise around the subject and crop with a small safety margin.
        result = result.crop(bbox)
        w, h = result.size
        pad = max(8, int(max(w, h) * 0.08))
        canvas = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (255, 255, 255, 255))
        canvas.alpha_composite(result, (pad, pad))
        return canvas.convert("RGB")
    except Exception as exc:
        print(f"[SEGMENT] Could not isolate subject: {type(exc).__name__}: {exc}")
        return None


def _encode_images(images):
    if not images:
        return torch.empty((0, 512), dtype=torch.float32)
    batch = torch.stack([preprocess(im).float() for im in images]).to(device)
    with torch.inference_mode():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return features.cpu().float()


def _load_names():
    with POKEMON_NAMES_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {str(k): str(v).strip().lower() for k, v in data.items()}
    return {str(i): str(v).strip().lower() for i, v in enumerate(data, 1)}


def _crop_manifest():
    records = []
    names_by_id = _load_names()
    if not CROPPED_DIR.is_dir():
        print(f"Official cropped-image folder not found: {CROPPED_DIR}")
        return records, "missing"
    for p in sorted(CROPPED_DIR.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED:
            continue
        # Official-artwork downloads are normally named 1.png, 2.png, ...
        match = re.match(r"^(\d+)$", p.stem.strip())
        if match and match.group(1) in names_by_id:
            name = names_by_id[match.group(1)]
        else:
            # Support named files too; numeric names are preferred for the official set.
            name = p.stem.replace("_", " ").replace("-", " ").strip().lower()
            if not name:
                continue
        try:
            stat = p.stat()
            records.append((str(p), name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            continue
    signature = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    return records, signature


def _load_or_build_crop_features():
    global crop_features, crop_names
    records, signature = _crop_manifest()
    if not records:
        crop_features, crop_names = None, []
        print("No official cropped PNG images were found; cropped comparison disabled.")
        return
    if CROPPED_CACHE_FILE.is_file() and CROPPED_CACHE_META.is_file():
        try:
            meta = json.loads(CROPPED_CACHE_META.read_text(encoding="utf-8"))
            if meta.get("signature") == signature:
                cached = torch.load(str(CROPPED_CACHE_FILE), map_location="cpu", weights_only=False)
                if len(cached.get("names", [])) == len(records) and cached["features"].shape[0] == len(records):
                    crop_features = cached["features"].float()
                    crop_names = list(cached["names"])
                    crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    print(f"Loaded cached official cropped-art features: {len(crop_names):,} images.")
                    return
        except Exception as exc:
            print(f"Crop feature cache will be rebuilt: {exc}")

    print(f"Building official cropped-art feature index for {len(records):,} images (first startup only)...")
    all_features, all_names = [], []
    batch_images, batch_names = [], []
    failed = 0
    for idx, (path, name, _, _) in enumerate(records, 1):
        try:
            with Image.open(path) as raw:
                batch_images.append(_open_rgb(raw))
                batch_names.append(name)
            if len(batch_images) >= 32 or idx == len(records):
                all_features.append(_encode_images(batch_images))
                all_names.extend(batch_names)
                batch_images, batch_names = [], []
        except Exception as exc:
            failed += 1
            print(f"[CROP INDEX] Skipping {path}: {exc}")
        if idx % 100 == 0 or idx == len(records):
            print(f"Cropped image indexing: {idx:,}/{len(records):,}", end="\r", flush=True)
    print()
    if not all_features:
        crop_features, crop_names = None, []
        return
    crop_features = torch.cat(all_features, dim=0).float()
    crop_names = all_names
    torch.save({"features": crop_features, "names": crop_names}, CROPPED_CACHE_FILE)
    CROPPED_CACHE_META.write_text(json.dumps({"signature": signature, "images": len(records), "failed": failed}, indent=2), encoding="utf-8")
    crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    print(f"Saved separate cropped-art cache: {len(crop_names):,} images ({failed} failed).")


def load_model():
    global _loaded, _load_error, model, preprocess, full_features, full_names, catch_name_map, device
    with _lock:
        if _loaded:
            return
        if _load_error:
            raise RuntimeError(_load_error)
        for path in (FULL_DB_FILE, CATCH_NAME_MAP_FILE, POKEMON_NAMES_FILE):
            if not path.is_file():
                raise RuntimeError(f"Required existing file not found: {path}")
        try:
            with CATCH_NAME_MAP_FILE.open("r", encoding="utf-8") as f:
                raw_map = json.load(f)
            catch_name_map = {
                _norm_name(folder): str(entry.get("catch_name", folder)).strip().lower()
                for folder, entry in raw_map.items() if isinstance(entry, dict)
            }
            database = torch.load(str(FULL_DB_FILE), map_location="cpu", weights_only=False)
            full_features = database["features"].float()
            image_paths = database["image_paths"]
            full_names = [os.path.basename(os.path.dirname(str(p))).replace("_", " ").strip().lower() for p in image_paths]
            if len(full_names) != full_features.shape[0]:
                raise RuntimeError(f"Full reference database mismatch: {len(full_names)} labels vs {full_features.shape[0]} features.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            model = model.to(device).eval()
            full_features = full_features / full_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            _load_or_build_crop_features()
            _loaded = True
            print(f"Loaded {len(full_names):,} full-spawn references and {len(crop_names or []):,} cropped references on {device}.")
        except Exception as exc:
            _load_error = f"Model initialization failed: {type(exc).__name__}: {exc}"
            raise RuntimeError(_load_error) from exc


def _scores_by_name(query, features, names, top=100):
    if features is None or not names or features.shape[0] == 0:
        return {}
    scores = (query.cpu() @ features.T).squeeze(0)
    count = min(top, scores.numel())
    top_scores, top_indices = scores.topk(count)
    best = {}
    for score, index in zip(top_scores.tolist(), top_indices.tolist()):
        name = str(names[index]).strip().lower()
        if name in ("nidoran f", "nidoranf", "nidoran female", "nidoran-f"):
            name = "nidoran"
        elif name in ("nidoran m", "nidoranm", "nidoran male", "nidoran-m"):
            name = "nidoran"
        key = _norm_name(name)
        if not key:
            continue
        if key not in best or score > best[key][0]:
            best[key] = (float(score), name)
    return {key: (score, name) for key, (score, name) in best.items()}


def recognize_image(image):
    load_model()
    original = ImageOps.exif_transpose(image).convert("RGB")
    original_tensor = preprocess(original).unsqueeze(0).to(device)
    with torch.inference_mode():
        full_query = model.encode_image(original_tensor)
        full_query = (full_query / full_query.norm(dim=-1, keepdim=True).clamp(min=1e-12)).cpu()
    full_rank = _scores_by_name(full_query, full_features, full_names)

    subject = _prepare_subject(image)
    crop_rank = {}
    if crop_features is not None:
        # When segmentation is unavailable, still compare the complete spawn to the
        # transparent-art index. The score is usually less reliable, but remains useful.
        crop_source = subject if subject is not None else original
        crop_tensor = preprocess(crop_source).unsqueeze(0).to(device)
        with torch.inference_mode():
            crop_query = model.encode_image(crop_tensor)
            crop_query = (crop_query / crop_query.norm(dim=-1, keepdim=True).clamp(min=1e-12)).cpu()
        crop_rank = _scores_by_name(crop_query, crop_features, crop_names)

    # Merge by canonicalized label. Crop match gets a slight preference because its
    # background is removed; the full-spawn score remains useful when segmentation fails.
    combined = {}
    all_keys = set(full_rank) | set(crop_rank)
    for key in all_keys:
        f = full_rank.get(key)
        c = crop_rank.get(key)
        if f and c:
            # Prefer the cropped-art score only when an actual foreground mask was made.
            full_weight, crop_weight = (0.42, 0.58) if subject is not None else (0.60, 0.40)
            score = full_weight * f[0] + crop_weight * c[0]
            display = c[1] if c[0] >= f[0] - 0.03 else f[1]
        elif c:
            score, display = c[0], c[1]
        else:
            score, display = f[0], f[1]
        combined[key] = {"score": float(score), "name": display,
                         "full_score": f[0] if f else None,
                         "cropped_score": c[0] if c else None}
    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:5]
    results = []
    for item in ranked:
        name = item["name"].strip().lower()
        catch = catch_name_map.get(_norm_name(name), name)
        results.append({"reference_name": name, "catch_name": catch, "score": item["score"],
                        "full_score": item["full_score"], "cropped_score": item["cropped_score"],
                        "segmented": subject is not None})
    return results


class Handler(BaseHTTPRequestHandler):
    server_version = "PokemonDetectorLocal/1.1"
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
    def do_OPTIONS(self): self._send(204, {})
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "loaded": _loaded, "device": device,
                             "segmentation_available": segmentation_available,
                             "segmentation_error": segmentation_error,
                             "full_database": str(FULL_DB_FILE),
                             "cropped_database": str(CROPPED_DIR),
                             "cropped_indexed": len(crop_names or []), "error": _load_error})
        else: self._send(404, {"error": "Not found"})
    def do_POST(self):
        if self.path != "/recognize": return self._send(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_IMAGE_BYTES:
                return self._send(413, {"error": f"Image must be between 1 byte and {MAX_IMAGE_BYTES} bytes."})
            content_type = self.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/") and "application/octet-stream" not in content_type:
                return self._send(415, {"error": "Expected image/* or application/octet-stream content type."})
            body = self.rfile.read(length)
            if len(body) != length: return self._send(400, {"error": "Incomplete request body."})
            with Image.open(io.BytesIO(body)) as img: img.verify()
            with Image.open(io.BytesIO(body)) as img: results = recognize_image(img)
            self._send(200, {"ok": True, "results": results, "segmentation_available": segmentation_available})
        except Exception as exc:
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    def log_message(self, fmt, *args): print("[HTTP] " + (fmt % args))

if __name__ == "__main__":
    print(f"Dual-database Pokémon Detector listening on http://{HOST}:{PORT}")
    print(f"Full-spawn refs: {FULL_DB_FILE}")
    print(f"Cropped PNG refs: {CROPPED_DIR}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
