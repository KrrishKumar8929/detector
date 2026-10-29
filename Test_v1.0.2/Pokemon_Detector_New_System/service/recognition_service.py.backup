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
SILHOUETTE_DIR = BASE_DIR / "poketwo_official_data" / "silhouettes"
CROPPED_CACHE_FILE = BASE_DIR / "poketwo_official_crop_features.pt"
CROPPED_CACHE_META = BASE_DIR / "poketwo_official_crop_features_meta.json"
SILHOUETTE_CACHE_FILE = BASE_DIR / "poketwo_official_silhouette_features.pt"
SILHOUETTE_CACHE_META = BASE_DIR / "poketwo_official_silhouette_features_meta.json"
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
silhouette_features = silhouette_names = None
device = "cpu"

try:
    import torch
    import open_clip
    from PIL import Image, ImageFile, ImageOps, ImageChops
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception as exc:
    torch = open_clip = Image = ImageOps = ImageChops = None
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


def _image_manifest(directory: Path):
    records = []
    names_by_id = _load_names()
    if not directory.is_dir():
        print(f"Reference-image folder not found (optional): {directory}")
        return records, "missing"
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED:
            continue
        match = re.match(r"^(\d+)$", p.stem.strip())
        if match:
            if match.group(1) in names_by_id:
                name = names_by_id[match.group(1)]
            elif p.parent != directory:
                name = p.parent.name.replace("_", " ").replace("-", " ").strip().lower()
            else:
                # Don't incorrectly label unknown numeric IDs with the directory name.
                continue
        else:
            # Support name-based files and per-Pokémon subfolders.
            name = p.stem.replace("_", " ").replace("-", " ").strip().lower()
            if not name or name.isdigit():
                name = p.parent.name.replace("_", " ").replace("-", " ").strip().lower()
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
    records, signature = _image_manifest(CROPPED_DIR)
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


def _load_or_build_silhouette_features():
    """Index official Pokétwo silhouettes separately; never edits the source images."""
    global silhouette_features, silhouette_names
    records, signature = _image_manifest(SILHOUETTE_DIR)
    if not records:
        silhouette_features, silhouette_names = None, []
        print("No official silhouette images found; silhouette comparison disabled.")
        return
    if SILHOUETTE_CACHE_FILE.is_file() and SILHOUETTE_CACHE_META.is_file():
        try:
            meta = json.loads(SILHOUETTE_CACHE_META.read_text(encoding="utf-8"))
            if meta.get("signature") == signature:
                cached = torch.load(str(SILHOUETTE_CACHE_FILE), map_location="cpu", weights_only=False)
                if len(cached.get("names", [])) == len(records) and cached["features"].shape[0] == len(records):
                    silhouette_features = cached["features"].float()
                    silhouette_names = list(cached["names"])
                    silhouette_features = silhouette_features / silhouette_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    print(f"Loaded cached official silhouette features: {len(silhouette_names):,} images.")
                    return
        except Exception as exc:
            print(f"Silhouette feature cache will be rebuilt: {exc}")
    print(f"Building official silhouette feature index for {len(records):,} images (first startup only)...")
    all_features, all_names, batch_images, batch_names = [], [], [], []
    failed = 0
    for idx, (path, name, _, _) in enumerate(records, 1):
        try:
            with Image.open(path) as raw:
                im = _open_rgb(raw)
                # Normalize black/white silhouettes to black foreground on a white canvas.
                gray = im.convert("L")
                gray = ImageOps.autocontrast(gray)
                im = gray.convert("RGB")
                batch_images.append(im)
                batch_names.append(name)
            if len(batch_images) >= 32 or idx == len(records):
                all_features.append(_encode_images(batch_images))
                all_names.extend(batch_names)
                batch_images, batch_names = [], []
        except Exception as exc:
            failed += 1
            print(f"[SILHOUETTE INDEX] Skipping {path}: {exc}")
        if idx % 100 == 0 or idx == len(records):
            print(f"Silhouette image indexing: {idx:,}/{len(records):,}", end="\r", flush=True)
    print()
    if not all_features:
        silhouette_features, silhouette_names = None, []
        return
    silhouette_features = torch.cat(all_features, dim=0).float()
    silhouette_names = all_names
    torch.save({"features": silhouette_features, "names": silhouette_names}, SILHOUETTE_CACHE_FILE)
    SILHOUETTE_CACHE_META.write_text(json.dumps({"signature": signature, "images": len(records), "failed": failed}, indent=2), encoding="utf-8")
    silhouette_features = silhouette_features / silhouette_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    print(f"Saved separate silhouette cache: {len(silhouette_names):,} images ({failed} failed).")


def _make_silhouette(subject):
    """Convert a foreground crop to a black-on-white silhouette query."""
    if subject is None:
        return None
    try:
        rgb = subject.convert("RGB")
        white = Image.new("RGB", rgb.size, "white")
        difference = ImageChops.difference(rgb, white).convert("L")
        foreground = difference.point(lambda value: 255 if value > 24 else 0)
        hist = foreground.histogram()
        foreground_count = sum(hist[128:])
        background_count = sum(hist[:128])
        if foreground_count < 20 or background_count < 20:
            return None
        canvas = Image.new("RGB", rgb.size, "white")
        canvas.paste("black", (0, 0, rgb.width, rgb.height), foreground)
        return canvas
    except Exception as exc:
        print(f"[SILHOUETTE QUERY] Could not create silhouette: {exc}")
        return None


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
            _load_or_build_silhouette_features()
            _loaded = True
            print(f"Loaded {len(full_names):,} full-spawn references, {len(crop_names or []):,} transparent-art references, and {len(silhouette_names or []):,} silhouette references on {device}.")
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
    subject = _prepare_subject(image)
    silhouette_query = _make_silhouette(subject) if subject is not None and silhouette_features is not None else None

    # Encode the full spawn, segmented subject, and silhouette together to reduce model-call overhead.
    query_images = [original]
    subject_index = None
    silhouette_index = None
    if subject is not None:
        subject_index = len(query_images)
        query_images.append(subject)
    if silhouette_query is not None:
        silhouette_index = len(query_images)
        query_images.append(silhouette_query)
    query_features = _encode_images(query_images)
    full_query = query_features[0:1]
    full_rank = _scores_by_name(full_query, full_features, full_names)

    crop_rank = {}
    if crop_features is not None:
        crop_query = query_features[subject_index:subject_index+1] if subject_index is not None else full_query
        crop_rank = _scores_by_name(crop_query, crop_features, crop_names)

    silhouette_rank = {}
    if silhouette_features is not None and silhouette_index is not None:
        silhouette_rank = _scores_by_name(query_features[silhouette_index:silhouette_index+1], silhouette_features, silhouette_names)

    combined = {}
    all_keys = set(full_rank) | set(crop_rank) | set(silhouette_rank)
    for key in all_keys:
        f = full_rank.get(key)
        c = crop_rank.get(key)
        s = silhouette_rank.get(key)
        if f and c and s:
            score = 0.34 * f[0] + 0.46 * c[0] + 0.20 * s[0]
            display = max([(f[0], f[1]), (c[0], c[1]), (s[0], s[1])], key=lambda item: item[0])[1]
        elif f and c:
            full_weight, crop_weight = (0.42, 0.58) if subject is not None else (0.60, 0.40)
            score = full_weight * f[0] + crop_weight * c[0]
            display = c[1] if c[0] >= f[0] - 0.03 else f[1]
        elif f and s:
            score = 0.68 * f[0] + 0.32 * s[0]
            display = s[1] if s[0] > f[0] else f[1]
        elif c and s:
            score = 0.70 * c[0] + 0.30 * s[0]
            display = c[1] if c[0] >= s[0] else s[1]
        elif c:
            score, display = c[0], c[1]
        elif s:
            score, display = s[0], s[1]
        else:
            score, display = f[0], f[1]
        combined[key] = {"score": float(score), "name": display,
                         "full_score": f[0] if f else None,
                         "cropped_score": c[0] if c else None,
                         "silhouette_score": s[0] if s else None}
    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:5]
    results = []
    for item in ranked:
        name = item["name"].strip().lower()
        catch = catch_name_map.get(_norm_name(name), name)
        results.append({"reference_name": name, "catch_name": catch, "score": item["score"],
                        "full_score": item["full_score"], "cropped_score": item["cropped_score"],
                        "silhouette_score": item["silhouette_score"],
                        "segmented": subject is not None,
                        "matched_sources": [source for source, value in (("background", item["full_score"]), ("transparent", item["cropped_score"]), ("silhouette", item["silhouette_score"])) if value is not None]})
    return results


class Handler(BaseHTTPRequestHandler):
    server_version = "PokemonDetectorLocal/2.0"
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
                             "cropped_indexed": len(crop_names or []),
                             "silhouette_database": str(SILHOUETTE_DIR),
                             "silhouette_indexed": len(silhouette_names or []), "error": _load_error})
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
    print(f"Transparent PNG refs: {CROPPED_DIR}")
    print(f"Silhouette refs (optional): {SILHOUETTE_DIR}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
