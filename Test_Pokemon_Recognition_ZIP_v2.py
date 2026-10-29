"""Batch-test the local Pokémon recognition service with every image in a ZIP.

Run from PowerShell:
    py Test_Pokemon_Recognition_ZIP.py

The script reads ZIP entries in memory, sends raw image bytes to the local service,
prints the top five results, and writes a CSV next to the ZIP. It never modifies
the service, reference databases, caches, or browser extension.
"""
import ast
import csv
import json
import mimetypes
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ENDPOINT = "http://127.0.0.1:8765/recognize"
SOURCE = Path(
    r"C:\PokemonDetector\Test_v1.0.2\Pokemon_Detector_New_System"
    r"\service\recognition_service.py"
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
TIMEOUT_SECONDS = 120


def inspect_service_source():
    """Inspect the local handler and report the response contract when available."""
    if not SOURCE.is_file():
        print(f"Service source not found at {SOURCE}; using the known raw-image API contract.")
        return
    try:
        source_text = SOURCE.read_text(encoding="utf-8-sig")
        ast.parse(source_text)  # Confirm that the file is readable Python source.
        if '"results": results' in source_text or "'results': results" in source_text:
            print("Inspected service response: JSON object with a 'results' list.")
            print("Each result contains 'catch_name', 'reference_name', and 'score'.")
        else:
            print("Service source is readable; response fields will be parsed defensively.")
        if "Expected image/* or application/octet-stream content type." in source_text:
            print("Detected request format: raw image bytes with an image/* Content-Type.")
    except Exception as exc:
        print(f"Source inspection warning: {type(exc).__name__}: {exc}")
        print("Continuing with the known raw-image API contract.")


def request_recognition(image_bytes, filename):
    """POST raw image bytes and return parsed JSON."""
    suffix = Path(filename).suffix.lower()
    content_type = mimetypes.types_map.get(suffix)
    if not content_type or not content_type.startswith("image/"):
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(suffix, "application/octet-stream")

    request = urllib.request.Request(
        ENDPOINT,
        data=image_bytes,
        headers={
            "Content-Type": content_type,
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        response_text = response.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Service returned invalid JSON: {response_text[:500]}") from exc
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"HTTP {response.status}: {response_text[:500]}")
        return payload


def extract_predictions(payload):
    """Parse the actual recognition_service.py response structure.

    Expected shape:
      {"ok": true, "results": [
        {"catch_name": "...", "reference_name": "...", "score": 0.8, ...}
      ]}
    """
    if not isinstance(payload, dict):
        raise ValueError("Response JSON is not an object.")
    if payload.get("ok") is False:
        raise ValueError(str(payload.get("error", "Service returned ok=false.")))

    results = payload.get("results")
    if not isinstance(results, list):
        # Defensive support for alternate response shapes without guessing silently.
        for key in ("predictions", "matches", "top5"):
            if isinstance(payload.get(key), list):
                results = payload[key]
                break
        else:
            raise ValueError(
                "Response contains no 'results' list. Keys received: "
                + ", ".join(map(str, payload.keys()))
            )

    predictions = []
    for item in results:
        if not isinstance(item, dict):
            continue
        # The supplied service defines catch_name as the user-facing command name.
        name = item.get("catch_name") or item.get("reference_name") or item.get("name")
        score = item.get("score", item.get("similarity", item.get("confidence")))
        if name is None:
            continue
        try:
            numeric_score = float(score) if score is not None else None
        except (TypeError, ValueError):
            numeric_score = None
        predictions.append((str(name), numeric_score))

    return predictions[:5]


def main():
    print("Pokémon Recognition ZIP Test")
    print(f"Endpoint: {ENDPOINT}")
    inspect_service_source()

    zip_text = input("\nEnter full path to test ZIP: ").strip().strip('"').strip()
    zip_path = Path(zip_text).expanduser()
    if not zip_path.is_file():
        print(f"ERROR: ZIP file not found: {zip_path}")
        input("Press Enter to exit...")
        return

    csv_path = zip_path.with_name(zip_path.stem + "_recognition_results.csv")
    fieldnames = [
        "image_filename",
        "zip_entry",
        "processing_seconds",
        "test_status",
        "error",
    ]
    for rank in range(1, 6):
        fieldnames.extend([f"top_{rank}_name", f"top_{rank}_similarity"])

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            image_entries = [
                info for info in archive.infolist()
                if not info.is_dir()
                and Path(info.filename).suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            if not image_entries:
                print("No supported images found in the ZIP.")
                input("Press Enter to exit...")
                return

            print(f"Found {len(image_entries)} supported image(s) in ZIP.")
            rows = []
            succeeded = 0
            failed = 0

            for index, info in enumerate(image_entries, 1):
                short_name = Path(info.filename).name
                print(f"\n[{index}/{len(image_entries)}] {info.filename}")
                started = time.perf_counter()
                row = {
                    "image_filename": short_name,
                    "zip_entry": info.filename,
                    "processing_seconds": "",
                    "test_status": "ERROR",
                    "error": "",
                }
                for rank in range(1, 6):
                    row[f"top_{rank}_name"] = ""
                    row[f"top_{rank}_similarity"] = ""

                try:
                    if info.file_size > MAX_IMAGE_BYTES:
                        raise ValueError(
                            f"Image is {info.file_size} bytes; service limit is "
                            f"{MAX_IMAGE_BYTES} bytes."
                        )
                    image_bytes = archive.read(info)
                    payload = request_recognition(image_bytes, info.filename)
                    predictions = extract_predictions(payload)
                    if not predictions:
                        raise ValueError("Service returned no usable named predictions.")

                    row["test_status"] = "OK"
                    succeeded += 1
                    for rank, (name, score) in enumerate(predictions, 1):
                        row[f"top_{rank}_name"] = name
                        row[f"top_{rank}_similarity"] = (
                            "" if score is None else f"{score:.8f}"
                        )
                    print("  Top predictions:")
                    for rank, (name, score) in enumerate(predictions, 1):
                        score_text = "score unavailable" if score is None else f"{score:.6f}"
                        print(f"    {rank}. {name} — {score_text}")
                    if len(predictions) < 5:
                        print(f"  Note: service returned only {len(predictions)} named result(s).")

                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")
                    row["error"] = f"HTTP {exc.code}: {detail[:800]}"
                    failed += 1
                    print(f"  ERROR: {row['error']}")
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    failed += 1
                    print(f"  ERROR: {row['error']}")
                finally:
                    elapsed = time.perf_counter() - started
                    row["processing_seconds"] = f"{elapsed:.4f}"
                    print(f"  Time: {elapsed:.4f} seconds")
                    rows.append(row)

            with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"\nCompleted: {succeeded} succeeded, {failed} failed.")
            print(f"CSV: {csv_path}")
            print("CSV prediction names use catch_name (the service's catch-command name).")
    except zipfile.BadZipFile:
        print("ERROR: The selected file is not a valid ZIP archive.")
    except Exception as exc:
        print(f"FATAL ERROR: {type(exc).__name__}: {exc}")

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
