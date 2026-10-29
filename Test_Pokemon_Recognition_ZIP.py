"""Read-only ZIP batch tester for a locally running Pokémon recognition service.

Run: py Test_Pokemon_Recognition_ZIP.py
No third-party packages required. Does not extract ZIP entries to disk.
"""
import ast
import base64
import csv
import io
import json
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

ENDPOINT = 'http://127.0.0.1:8765/recognize'
SOURCE = Path(r'C:\PokemonDetector\Test_v1.0.2\Pokemon_Detector_New_System\service\recognition_service.py')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
TIMEOUT = 120


def inspect_source():
    """Inspect the local service source when present; never change it."""
    if not SOURCE.is_file():
        print('Service source not accessible here; using multipart field "image" and flexible response parsing.')
        print('If the service rejects the request, inspect its /recognize handler and adjust make_request().')
        return 'image', 'multipart'
    try:
        code = SOURCE.read_text(encoding='utf-8-sig')
        tree = ast.parse(code)
        field, mode = 'image', 'multipart'
        handlers = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decos = [ast.get_source_segment(code, d) or '' for d in node.decorator_list]
            if any('recognize' in d for d in decos) or node.name in ('recognize', 'recognize_image'):
                handlers.append(node)
        if handlers:
            handler = handlers[0]
            snippet = ast.get_source_segment(code, handler) or ''
            print('Inspected local /recognize handler:', handler.name)
            for arg in handler.args.args + handler.args.kwonlyargs:
                ann = ast.get_source_segment(code, arg.annotation) if arg.annotation else ''
                if ann and ('UploadFile' in ann or 'File' in ann):
                    field = arg.arg
                    break
            match = re.search(r'(?:request\.)?files\s*\[\s*[\'\"]([^\'\"]+)', snippet)
            match = match or re.search(r'(?:request\.)?files\.get\(\s*[\'\"]([^\'\"]+)', snippet)
            if match:
                field = match.group(1)
            if re.search(r'request\.get_json\(|await\s+request\.json\(|request\.json\b', snippet):
                mode = 'json'
            if re.search(r'request\.(?:files|form)\b|UploadFile|\bFile\s*\(', snippet):
                mode = 'multipart'
            print(f'Detected request style: {mode}; image field: {field!r}')
        else:
            print('Could not identify /recognize handler statically; defaulting to multipart image.')
        return field, mode
    except Exception as exc:
        print(f'Source inspection warning: {exc}; defaulting to multipart image.')
        return 'image', 'multipart'


def make_request(name, data, field, mode):
    mime = mimetypes.guess_type(name)[0] or 'application/octet-stream'
    if mode == 'json':
        # Adjust this if source inspection shows a different JSON contract.
        payload = json.dumps({field: base64.b64encode(data).decode('ascii'), 'filename': Path(name).name}).encode()
        content_type = 'application/json'
    else:
        boundary = '----PokemonZipTest' + uuid.uuid4().hex
        safe_name = Path(name).name.replace('"', '_').replace('\r', '_').replace('\n', '_')
        payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{safe_name}"\r\n'
                   f'Content-Type: {mime}\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
        content_type = f'multipart/form-data; boundary={boundary}'
    request = urllib.request.Request(ENDPOINT, data=payload, headers={'Content-Type': content_type, 'Accept': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode('utf-8-sig'))


def predictions_from_response(result):
    """Accept common response layouts, retaining unknown payloads for troubleshooting."""
    if isinstance(result, list):
        candidates = result
    elif isinstance(result, dict):
        candidates = None
        for holder in (result, result.get('result'), result.get('data')):
            if not isinstance(holder, dict):
                continue
            for key in ('top5', 'top_5', 'predictions', 'top_predictions', 'results', 'matches', 'candidates'):
                if isinstance(holder.get(key), list):
                    candidates = holder[key]
                    break
            if candidates is not None:
                break
        if candidates is None:
            if any(k in result for k in ('name', 'pokemon', 'pokemon_name', 'label', 'prediction')):
                candidates = [result]
            else:
                raise ValueError('Unknown response structure: ' + json.dumps(result, ensure_ascii=False)[:700])
    else:
        raise ValueError('Unexpected response type: ' + type(result).__name__)
    parsed = []
    for item in candidates[:5]:
        if isinstance(item, dict):
            name = next((item[k] for k in ('name', 'pokemon', 'pokemon_name', 'label', 'prediction', 'class_name') if item.get(k) is not None), None)
            score = next((item[k] for k in ('similarity', 'similarity_score', 'score', 'confidence', 'cosine_similarity') if item.get(k) is not None), None)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name, score = item[0], item[1]
        else:
            name, score = item, None
        parsed.append((str(name) if name is not None else '', score))
    if not parsed:
        raise ValueError('Service returned an empty prediction list.')
    return parsed


def main():
    print('Pokémon recognition ZIP test (read-only)')
    raw = input('Enter full path to test ZIP: ').strip().strip('"').strip("'")
    path = Path(raw).expanduser()
    if not path.is_file() or not zipfile.is_zipfile(path):
        print('ERROR: That path is not an existing valid ZIP file.')
        return 1
    field, mode = inspect_source()
    output = path.with_name(path.stem + '_recognition_results.csv')
    columns = ['image_filename', 'zip_entry', 'processing_seconds', 'test_status', 'error']
    for n in range(1, 6):
        columns += [f'top_{n}_name', f'top_{n}_similarity']
    rows = []
    with zipfile.ZipFile(path) as archive:
        images = [info for info in archive.infolist() if not info.is_dir() and Path(info.filename).suffix.lower() in EXTENSIONS]
        print(f'Found {len(images)} supported image(s) in ZIP.\n')
        for index, info in enumerate(images, 1):
            print(f'[{index}/{len(images)}] {info.filename}')
            row = dict.fromkeys(columns, '')
            row['image_filename'] = Path(info.filename).name
            row['zip_entry'] = info.filename
            start = time.perf_counter()
            try:
                data = archive.read(info)
                result = make_request(info.filename, data, field, mode)
                predictions = predictions_from_response(result)
                for n, (name, score) in enumerate(predictions, 1):
                    row[f'top_{n}_name'] = name
                    row[f'top_{n}_similarity'] = score if score is not None else ''
                    print(f'  {n}. {name}: {score if score is not None else "N/A"}')
                row['test_status'] = 'OK'
            except urllib.error.HTTPError as exc:
                body = exc.read().decode('utf-8', errors='replace')[:700]
                row['test_status'] = 'ERROR'
                row['error'] = f'HTTP {exc.code}: {body}'
                print('  ERROR:', row['error'])
            except Exception as exc:
                row['test_status'] = 'ERROR'
                row['error'] = f'{type(exc).__name__}: {exc}'
                print('  ERROR:', row['error'])
            finally:
                row['processing_seconds'] = f'{time.perf_counter() - start:.4f}'
                print('  Time:', row['processing_seconds'], 'seconds\n')
                rows.append(row)
    with output.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(r['test_status'] == 'OK' for r in rows)
    print(f'Completed: {ok} succeeded, {len(rows) - ok} failed. CSV: {output}')
    return 0 if ok == len(rows) and rows else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.')
        sys.exit(130)
