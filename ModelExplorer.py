#!/usr/bin/env python3
"""Cluster-aware Hugging Face model explorer (GGUF / MLX).

Public launch build:
  - Authenticated HF crawl via HF_TOKEN (Actions only; never written to output)
  - Writes _site/index.html + _site/data/{gguf,mlx}/*.json + meta.json
  - Browser Refresh loads CDN snapshots only (no huggingface.co from client)
  - Hardware estimator: effective-bits quants, 4-term memory, tok/s, interconnect
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone

from huggingface_hub import HfApi

DEFAULT_QUANT = "Q4_K_M"
DEFAULT_OVERHEAD = 0.20
MIN_PARAMS_B = 0
DEFAULT_HOSTS = [("My machine", 16, 400)]  # name, GB, bandwidth GB/s
SEED_MAX_B = 2000
SEED_LIMIT_PER_BAND = 250
SIZE_BAND_EDGES = [0, 4, 8, 15, 35, 70, 120, 250, 500, 1000, 2000]

# Effective bytes/param (GGUF names are floors — Q4_K_M ≈ 4.9 bits, not 4.0)
BYTES_PER_PARAM = {
    "IQ4_XS": 0.53,
    "Q4_K_M": 0.61,
    "Q5_K_M": 0.71,
    "Q6_K": 0.82,
    "Q8_0": 1.06,
    "MXFP4": 0.53,
    "FP8": 1.00,
    "BF16": 2.00,
    "MLX4": 0.56,
    "MLX5": 0.69,
    "MLX6": 0.81,
    "MLX8": 1.06,
}

QUANT_BITS = {
    "IQ4_XS": 4,
    "Q4_K_M": 4,
    "Q5_K_M": 5,
    "Q6_K": 6,
    "Q8_0": 8,
    "MXFP4": 4,
    "FP8": 8,
    "BF16": 16,
    "MLX4": 4,
    "MLX5": 5,
    "MLX6": 6,
    "MLX8": 8,
}


def auto_map_range(total_gb: float, quant: str = DEFAULT_QUANT, overhead: float = DEFAULT_OVERHEAD):
    bpp = BYTES_PER_PARAM.get(quant, BYTES_PER_PARAM[DEFAULT_QUANT])
    weight_budget = total_gb * (1.0 - overhead)
    max_b = max(1, round(weight_budget / bpp))
    return MIN_PARAMS_B, int(max_b)


def size_bands(max_b: int):
    bands = []
    for i in range(len(SIZE_BAND_EDGES) - 1):
        lo = SIZE_BAND_EDGES[i]
        if lo >= max_b:
            break
        bands.append((lo, min(SIZE_BAND_EDGES[i + 1], max_b)))
    if max_b > SIZE_BAND_EDGES[-1]:
        bands.append((SIZE_BAND_EDGES[-1], max_b))
    return bands


def _extract_bits(model_id: str, tags=None):
    id_ = model_id.lower()
    for t in tags or []:
        m = re.match(r"^([2-8])-?bit$", str(t).lower())
        if m:
            return int(m.group(1))
    m = re.search(r"(?:^|[-_/])mlx-?([2-8])bit(?:$|[-_/])", id_)
    if m:
        return int(m.group(1))
    m = re.search(r"[-_]([2-8])bit(?:$|[-_/])", id_)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[-_/])(?:q|iq)([4568])(?:[_-]|$)", id_)
    if m:
        return int(m.group(1))
    if re.search(r"(?:^|[-_/])(?:bf16|bfloat16)(?:$|[-_/])", id_) or re.search(
        r"[-_]f16(?:$|[-_/])", id_
    ):
        return 16
    if re.search(r"(?:^|[-_/])(?:fp8|f8)(?:$|[-_/])", id_):
        return 8
    if re.search(r"(?:^|[-_/])mxfp4(?:$|[-_/])", id_):
        return 4
    return None


def _size_from_meta(gguf=None, safetensors=None):
    def total_of(obj):
        if obj is None:
            return None
        total = obj.get("total") if isinstance(obj, dict) else getattr(obj, "total", None)
        if isinstance(total, (int, float)) and total > 0:
            size_b = float(total) / 1e9
            return round(size_b, 3) if size_b < 1 else round(size_b, 1)
        return None

    return total_of(gguf) or total_of(safetensors)


def _parse_size_from_text(text: str):
    if not text:
        return None, None
    m = re.search(r"(?i)(\d+(?:\.\d+)?)B-?A(\d+(?:\.\d+)?)B?", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(?i)(\d+)x(\d+(?:\.\d+)?)B", text)
    if m:
        total = float(m.group(1)) * float(m.group(2))
        return total, float(m.group(2))
    candidates = []
    for m in re.finditer(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)B(?![A-Za-z])", text):
        prefix = text[max(0, m.start() - 3) : m.start()]
        if re.search(r"(?i)(?:q|iq)$", prefix):
            continue
        val = float(m.group(1))
        if 0.05 <= val <= 3000:
            candidates.append(val)
    if candidates:
        total = min(candidates)
        return total, total
    m_candidates = []
    for m in re.finditer(r"(?i)(?<![A-Za-z0-9])(\d{2,3})M(?![A-Za-z])", text):
        val = float(m.group(1))
        if 20 <= val <= 999:
            m_candidates.append(val / 1000.0)
    if m_candidates:
        total = min(m_candidates)
        return total, total
    return None, None


def resolve_size(model_id: str, tags=None, gguf=None, safetensors=None):
    meta = _size_from_meta(gguf, safetensors)
    total, active = _parse_size_from_text(model_id)
    if meta is not None:
        if active is not None and total is not None and active < total:
            return meta, active
        return meta, meta
    if total is not None:
        return total, active if active is not None else total
    for t in tags or []:
        t = str(t)
        if t.startswith("base_model:"):
            name = re.sub(r"^base_model:(?:quantized:)?", "", t)
            total, active = _parse_size_from_text(name)
            if total is not None:
                return total, active if active is not None else total
    return None, None


def format_size_label(size_b):
    if size_b is None:
        return "—"
    size_b = float(size_b)
    if size_b < 1:
        return f"{int(round(size_b * 1000))}M"
    rounded = round(size_b, 1)
    if rounded == int(rounded):
        return f"{int(rounded)}B"
    return f"{rounded}B"


def to_record(model):
    tags = getattr(model, "tags", None) or []
    gguf = getattr(model, "gguf", None)
    safetensors = getattr(model, "safetensors", None)
    last_modified = getattr(model, "last_modified", None) or getattr(model, "lastModified", None)
    if isinstance(last_modified, str):
        model_date = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    else:
        model_date = last_modified
        if model_date is not None and model_date.tzinfo is None:
            model_date = model_date.replace(tzinfo=timezone.utc)
    size_b, active_b = resolve_size(model.id, tags, gguf=gguf, safetensors=safetensors)
    bits = _extract_bits(model.id, tags)
    return {
        "id": model.id,
        "downloads": getattr(model, "downloads", 0) or 0,
        "date": model_date.strftime("%Y-%m-%d") if model_date else "",
        "sizeB": size_b,
        "activeB": active_b,
        "bits": bits,
    }


def fetch_band(api: HfApi, library: str, lo: int, hi: int, cutoff: datetime):
    print(f"  {library} {lo}B–{hi}B (top {SEED_LIMIT_PER_BAND})...")
    models = api.list_models(
        filter=library,
        num_parameters=f"min:{lo}B,max:{hi}B",
        sort="downloads",
        limit=SEED_LIMIT_PER_BAND,
        expand=["gguf", "safetensors", "tags", "lastModified", "downloads"],
    )
    records = []
    for model in models:
        last_modified = getattr(model, "last_modified", None) or getattr(model, "lastModified", None)
        if not last_modified:
            continue
        if isinstance(last_modified, str):
            model_date = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
        else:
            model_date = last_modified
            if model_date.tzinfo is None:
                model_date = model_date.replace(tzinfo=timezone.utc)
        if model_date < cutoff:
            continue
        records.append(to_record(model))
    return records


def fetch_all_snapshots(api: HfApi, max_b: int):
    cutoff = datetime.now(timezone.utc) - timedelta(days=365)
    bands = size_bands(max_b)
    out = {"gguf": {}, "mlx": {}}
    for library in ("gguf", "mlx"):
        print(f"Fetching {library.upper()} bands…")
        for lo, hi in bands:
            key = f"{lo}-{hi}"
            out[library][key] = fetch_band(api, library, lo, hi, cutoff)
    return out, bands


def build_rows(records):
    rows = []
    for rec in records:
        high = "high" if rec["downloads"] > 1000 else ""
        bits_attr = f' data-bits="{rec["bits"]}"' if rec["bits"] is not None else ""
        size_attr = f' data-size-b="{rec["sizeB"]}"' if rec["sizeB"] is not None else ""
        active_attr = (
            f' data-active-b="{rec["activeB"]}"' if rec.get("activeB") is not None else ""
        )
        size_label = format_size_label(rec["sizeB"])
        rows.append(
            f"""                    <tr data-model-id="{rec['id'].lower()}"{bits_attr}{size_attr}{active_attr}>
                        <td><a class="model-link" href="https://huggingface.co/{rec['id']}" target="_blank" rel="noopener">{rec['id']}</a></td>
                        <td style="text-align: right;"><span class="size-text">{size_label}</span></td>
                        <td style="text-align: right;"><span class="fit-text">—</span></td>
                        <td style="text-align: right;"><span class="tps-text">—</span></td>
                        <td style="text-align: right;"><span class="badge-downloads {high}">{rec['downloads']:,}</span></td>
                        <td style="text-align: right;"><span class="date-text">{rec['date']}</span></td>
                    </tr>"""
        )
    return "\n".join(rows)


def write_site(snapshots, bands, seed_records, min_b, max_b):
    root = os.path.dirname(os.path.abspath(__file__))
    site = os.path.join(root, "_site")
    data_root = os.path.join(site, "data")
    if os.path.isdir(site):
        shutil.rmtree(site)
    os.makedirs(data_root, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "generated_at": generated_at,
        "seed_max_b": SEED_MAX_B,
        "formats": ["gguf", "mlx"],
        "bands": [{"lo": a, "hi": b} for a, b in bands],
        "quant_default": DEFAULT_QUANT,
    }
    with open(os.path.join(data_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for library, band_map in snapshots.items():
        lib_dir = os.path.join(data_root, library)
        os.makedirs(lib_dir, exist_ok=True)
        for key, records in band_map.items():
            path = os.path.join(lib_dir, f"{key}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, separators=(",", ":"))

    html = (
        HTML_TEMPLATE.replace("__MIN_B__", str(min_b))
        .replace("__MAX_B__", str(max_b))
        .replace("__TOTAL__", str(len(seed_records)))
        .replace("__DATE__", datetime.now().strftime("%Y-%m-%d %H:%M"))
        .replace("__GENERATED_AT__", generated_at)
        .replace("__TABLE_ROWS__", build_rows(seed_records))
        .replace("__BYTES_PER_PARAM_JSON__", json.dumps(BYTES_PER_PARAM))
        .replace("__QUANT_BITS_JSON__", json.dumps(QUANT_BITS))
        .replace("__BAND_EDGES_JSON__", json.dumps(SIZE_BAND_EDGES))
        .replace("__SEED_MAX_B__", str(SEED_MAX_B))
    )
    with open(os.path.join(site, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    with open(os.path.join(root, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    root_data = os.path.join(root, "data")
    if os.path.isdir(root_data):
        shutil.rmtree(root_data)
    shutil.copytree(data_root, root_data)
    return site


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Explorer — Cluster-aware GGUF / MLX</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --panel: rgba(20, 26, 43, 0.72);
            --border: rgba(255, 255, 255, 0.08);
            --text: #f3f4f6;
            --muted: #9ca3af;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.15);
            --success: #10b981;
            --success-glow: rgba(16, 185, 129, 0.15);
            --warn: #fbbf24;
            --danger: #f87171;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'DM Sans', sans-serif;
            background-color: var(--bg);
            background-image:
                radial-gradient(circle at 12% 18%, rgba(59, 130, 246, 0.1) 0%, transparent 42%),
                radial-gradient(circle at 88% 78%, rgba(16, 185, 129, 0.06) 0%, transparent 40%);
            background-attachment: fixed;
            color: var(--text);
            min-height: 100vh;
            padding: 2rem 1.5rem 3rem;
        }
        .container { max-width: 1280px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 1.75rem; }
        h1 {
            font-size: 2.35rem; font-weight: 700; letter-spacing: -0.03em;
            background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.4rem;
        }
        .subtitle { color: var(--muted); font-size: 1.05rem; }
        .panel {
            background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
            padding: 1.1rem 1.25rem; backdrop-filter: blur(12px); margin-bottom: 1.1rem;
        }
        .panel-title {
            font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.06em; color: var(--muted); margin-bottom: 0.85rem;
        }
        .format-toggle { display: inline-flex; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
        .format-toggle button {
            background: transparent; border: none; color: var(--muted); font: inherit; font-weight: 600;
            padding: 0.55rem 1.15rem; cursor: pointer;
        }
        .format-toggle button.active { background: var(--accent-glow); color: #93c5fd; }
        .format-toggle button:hover:not(.active) { color: var(--text); }
        .controls-row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; justify-content: space-between; }
        .hosts-list { display: flex; flex-direction: column; gap: 0.55rem; }
        .host-row {
            display: grid; grid-template-columns: 1fr 88px 96px 168px 40px; gap: 0.5rem; align-items: center;
        }
        .hosts-header {
            display: grid; grid-template-columns: 1fr 88px 96px 168px 40px; gap: 0.5rem;
            font-size: 0.72rem; color: var(--muted); text-transform: uppercase;
            letter-spacing: 0.05em; padding: 0 0.2rem; margin-bottom: 0.4rem;
        }
        .hosts-header span:not(:first-child) { text-align: right; }
        .host-row input, .host-row select, .field select, .field input[type="number"] {
            width: 100%; padding: 0.55rem 0.75rem; background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border); border-radius: 8px; color: var(--text); font: inherit; font-size: 0.95rem;
        }
        .host-row input:focus, .field select:focus, .field input:focus {
            outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-glow);
        }
        .host-row input[type="number"] { font-family: 'JetBrains Mono', monospace; text-align: right; }
        .icon-btn, .text-btn {
            background: transparent; border: 1px solid var(--border); color: var(--muted);
            border-radius: 8px; cursor: pointer; font: inherit;
        }
        .icon-btn { height: 38px; width: 38px; font-size: 1.1rem; line-height: 1; }
        .text-btn { padding: 0.45rem 0.85rem; font-size: 0.88rem; font-weight: 500; }
        .icon-btn:hover, .text-btn:hover { color: var(--text); border-color: var(--accent); background: var(--accent-glow); }
        .icon-btn.danger:hover { border-color: var(--danger); color: var(--danger); background: rgba(248, 113, 113, 0.1); }
        .host-actions { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.85rem; }
        .derive-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.85rem; align-items: end;
        }
        .field label { display: block; font-size: 0.75rem; color: var(--muted); margin-bottom: 0.35rem; font-weight: 500; }
        .readout {
            font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 500;
            color: #93c5fd; padding: 0.55rem 0;
        }
        .formula-hint { margin-top: 0.75rem; font-size: 0.8rem; color: var(--muted); line-height: 1.45; }
        .bound-box {
            margin-top: 0.85rem; padding: 0.75rem 0.9rem; border-radius: 10px;
            border: 1px solid var(--border); background: rgba(15, 23, 42, 0.45);
            font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #93c5fd;
        }
        .search-container { position: relative; margin-bottom: 1rem; }
        .search-input {
            width: 100%; padding: 1rem 1.5rem 1rem 3rem; background: var(--panel);
            border: 1px solid var(--border); border-radius: 12px; color: var(--text);
            font-size: 1rem; backdrop-filter: blur(12px);
        }
        .search-input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
        .search-icon { position: absolute; left: 1rem; top: 50%; transform: translateY(-50%); color: var(--muted); pointer-events: none; }
        .stats {
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
            gap: 0.75rem; margin-bottom: 0.85rem; color: var(--muted); font-size: 0.9rem; font-weight: 500;
        }
        .stats-right { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
        .refresh-btn {
            display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.4rem 0.85rem;
            background: transparent; border: 1px solid var(--border); border-radius: 8px;
            color: var(--muted); font: inherit; font-size: 0.85rem; font-weight: 500; cursor: pointer;
        }
        .refresh-btn:hover:not(:disabled) { color: var(--text); border-color: var(--accent); background: var(--accent-glow); }
        .refresh-btn:disabled { opacity: 0.6; cursor: wait; }
        .refresh-btn.spinning svg { animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .refresh-status { font-size: 0.85rem; color: var(--danger); }
        .table-wrapper {
            background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
            overflow-x: auto; backdrop-filter: blur(12px);
        }
        table { width: 100%; border-collapse: collapse; text-align: left; min-width: 900px; }
        th {
            background: rgba(15, 23, 42, 0.85); padding: 1rem 1rem; font-size: 0.8rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted);
            border-bottom: 1px solid var(--border); cursor: pointer; user-select: none; white-space: nowrap;
        }
        th:hover { color: var(--text); }
        .sort-arrow { font-size: 0.75rem; color: var(--accent); margin-left: 6px; }
        td { padding: 0.95rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.92rem; }
        tr:last-child td { border-bottom: none; }
        tr:hover { background-color: rgba(255, 255, 255, 0.02); }
        .model-link { font-family: 'JetBrains Mono', monospace; color: #60a5fa; text-decoration: none; font-weight: 500; font-size: 0.88rem; }
        .model-link:hover { color: #93c5fd; text-decoration: underline; }
        .badge-downloads {
            display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem;
            font-weight: 600; background: var(--accent-glow); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3);
        }
        .badge-downloads.high { background: var(--success-glow); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
        .date-text, .size-text, .tps-text { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; }
        .size-text { color: #93c5fd; font-weight: 500; }
        .tps-text { color: #a5b4fc; }
        .fit-text { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 600; }
        .fit-ok { color: #6ee7b7; }
        .fit-tight { color: var(--warn); }
        .fit-no { color: var(--danger); }
        .no-results { display: none; text-align: center; padding: 3rem; color: var(--muted); font-size: 1.05rem; }
        .pager {
            display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
            gap: 0.75rem; margin-top: 0.85rem; color: var(--muted); font-size: 0.88rem;
        }
        .pager-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
        .pager-btn {
            display: inline-flex; align-items: center; justify-content: center;
            min-width: 2.1rem; height: 2.1rem; padding: 0 0.65rem;
            background: transparent; border: 1px solid var(--border); border-radius: 8px;
            color: var(--muted); font: inherit; font-size: 0.85rem; font-weight: 500; cursor: pointer;
        }
        .pager-btn:hover:not(:disabled) { color: var(--text); border-color: var(--accent); background: var(--accent-glow); }
        .pager-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .pager-btn.active { color: #93c5fd; border-color: var(--accent); background: var(--accent-glow); }
        .pager-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
        .pager-size {
            display: inline-flex; align-items: center; gap: 0.4rem;
        }
        .pager-size select {
            padding: 0.35rem 0.55rem; background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border); border-radius: 8px; color: var(--text);
            font: inherit; font-size: 0.85rem;
        }
        @media (max-width: 720px) {
            .host-row, .hosts-header { grid-template-columns: 1fr 72px 78px 120px 36px; }
            h1 { font-size: 1.8rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Model Explorer</h1>
            <div class="subtitle">Cluster-aware GGUF / MLX · memory fit · theoretical tok/s · every-2h snapshots</div>
        </header>

        <div class="panel">
            <div class="panel-title">Format</div>
            <div class="controls-row">
                <div class="format-toggle" role="group" aria-label="Model format">
                    <button type="button" id="btnGguf" class="active" data-format="gguf">GGUF</button>
                    <button type="button" id="btnMlx" data-format="mlx">MLX</button>
                </div>
                <span id="dateWindowHint" class="formula-hint" style="margin:0">Snapshot: __GENERATED_AT__ · last 365 days of models</span>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Hosts</div>
            <div class="hosts-header"><span>Name</span><span>RAM GB</span><span>Usable GB</span><span>Mem bandwidth</span><span></span></div>
            <div id="hostsList" class="hosts-list"></div>
            <div class="host-actions">
                <button type="button" class="text-btn" id="addHostBtn">+ Add host</button>
                <button type="button" class="text-btn" id="loadClusterBtn">Load example cluster</button>
                <button type="button" class="text-btn" id="loadNvl72Btn">Load GB200 NVL72</button>
                <button type="button" class="text-btn" id="resetHostsBtn">Reset to single host</button>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Auto-map + estimator</div>
            <div class="derive-grid">
                <div class="field">
                    <label>Total usable GB</label>
                    <div class="readout" id="totalGbReadout">16 GB</div>
                </div>
                <div class="field">
                    <label>Quant</label>
                    <select id="quantSelect">
                        <option value="IQ4_XS">IQ4_XS (0.53 B/p)</option>
                        <option value="Q4_K_M" selected>Q4_K_M (0.61 B/p)</option>
                        <option value="Q5_K_M">Q5_K_M (0.71 B/p)</option>
                        <option value="Q6_K">Q6_K (0.82 B/p)</option>
                        <option value="Q8_0">Q8_0 (1.06 B/p)</option>
                        <option value="MXFP4">MXFP4 (0.53 B/p)</option>
                        <option value="FP8">FP8 (1.00 B/p)</option>
                        <option value="BF16">BF16 (2.00 B/p, GGUF)</option>
                        <option value="MLX4">MLX 4-bit (0.56)</option>
                        <option value="MLX5">MLX 5-bit (0.69)</option>
                        <option value="MLX6">MLX 6-bit (0.81)</option>
                        <option value="MLX8">MLX 8-bit (1.06)</option>
                    </select>
                </div>
                <div class="field">
                    <label>Context</label>
                    <select id="contextSelect">
                        <option value="4096" selected>4k</option>
                        <option value="8192">8k</option>
                        <option value="32768">32k</option>
                        <option value="131072">128k</option>
                        <option value="262144">256k</option>
                        <option value="524288">512k</option>
                        <option value="1048576">1M</option>
                    </select>
                </div>
                <div class="field">
                    <label>KV dtype</label>
                    <select id="kvSelect">
                        <option value="2" selected>FP16</option>
                        <option value="1">FP8</option>
                    </select>
                </div>
                <div class="field">
                    <label>Interconnect</label>
                    <select id="linkSelect">
                        <option value="tb4">USB4/TB4 40G — ~32 Gbps data</option>
                        <option value="tb5">USB4v2/TB5 80G — ~64 Gbps data</option>
                        <option value="tb5a">USB4v2 asym 120G — ~96 Gbps data</option>
                        <option value="1gbe">1 GbE — TCP RPC</option>
                        <option value="10gbe">10 GbE — TCP RPC</option>
                        <option value="25gbe">25 GbE — RDMA/RoCE</option>
                        <option value="100gbe" selected>100 GbE — RDMA/RoCE</option>
                        <option value="200gbe">200 GbE — RDMA/RoCE</option>
                        <option value="400gbe">400 GbE — RDMA/RoCE</option>
                        <option value="800gbe">800 GbE — RDMA/RoCE</option>
                    </select>
                </div>
                <div class="field">
                    <label>Overhead %</label>
                    <input type="number" id="overheadInput" min="0" max="50" step="1" value="20">
                </div>
                <div class="field">
                    <label>Derived min</label>
                    <div class="readout" id="minBReadout">__MIN_B__B</div>
                </div>
                <div class="field">
                    <label>Derived max</label>
                    <div class="readout" id="maxBReadout">__MAX_B__B</div>
                </div>
            </div>
            <div id="boundReadout" class="bound-box">Cluster bound: —</div>
            <div class="formula-hint">
                Memory ≈ weights (effective bits) + KV(context) + overhead floor.
                Usable GB: blank = auto (Apple 75% Metal cap, others 1 − overhead%); set it explicitly if you
                raised <code>iogpu.wired_limit_mb</code> (e.g. 60/64 on Apple Silicon, 120/128 Strix Halo).
                tok/s ≈ memory bandwidth / bytes read per token
                (MoE uses active params). Cluster decode is sequential pipeline: stage times sum, plus
                per-hop transfer + latency. Thunderbolt = IP-over-TB bridge (TCP, not RDMA); RDMA assumed
                only on ≥25GbE (RoCE). Slow links barely change single-stream decode (only a ~KB hidden
                state crosses per token) — they hurt prefill/TTFT, model load, and tensor-parallel instead.
                Estimates are theoretical ceilings.
                Refresh reloads the every-2h CDN snapshot — Shift-click bypasses local cache.
            </div>
        </div>

        <div class="search-container">
            <svg class="search-icon" width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
            </svg>
            <input type="text" id="searchInput" class="search-input" placeholder="Search by name, size (e.g. 35B), creator…">
        </div>

        <div class="stats">
            <span id="matchCount">Showing __TOTAL__ models</span>
            <div class="stats-right">
                <span id="refreshStatus" class="refresh-status"></span>
                <span id="generatedAt">Data as of: __DATE__</span>
                <button type="button" id="refreshBtn" class="refresh-btn" title="Reload CDN snapshot (Shift-click: bypass cache)">
                    <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                    Refresh
                </button>
            </div>
        </div>

        <div class="table-wrapper">
            <table id="modelsTable">
                <thead>
                    <tr>
                        <th style="width: 38%;" onclick="sortTable(0)">Model ID<span class="sort-arrow" id="arrow0"></span></th>
                        <th style="width: 10%; text-align: right;" onclick="sortTable(1)">Size<span class="sort-arrow" id="arrow1"></span></th>
                        <th style="width: 10%; text-align: right;" onclick="sortTable(2)">Fit<span class="sort-arrow" id="arrow2"></span></th>
                        <th style="width: 12%; text-align: right;" onclick="sortTable(3)">Est. tok/s<span class="sort-arrow" id="arrow3"></span></th>
                        <th style="width: 14%; text-align: right;" onclick="sortTable(4)">Downloads<span class="sort-arrow" id="arrow4"> ▼</span></th>
                        <th style="width: 16%; text-align: right;" onclick="sortTable(5)">Last Modified<span class="sort-arrow" id="arrow5"></span></th>
                    </tr>
                </thead>
                <tbody id="tableBody">
__TABLE_ROWS__
                </tbody>
            </table>
            <div id="noResults" class="no-results">No matching models found.</div>
        </div>
        <div class="pager" id="pager" hidden>
            <div class="pager-size">
                <label for="pageSizeSelect">Per page</label>
                <select id="pageSizeSelect" aria-label="Results per page">
                    <option value="50" selected>50</option>
                    <option value="100">100</option>
                    <option value="150">150</option>
                </select>
            </div>
            <div class="pager-controls">
                <button type="button" class="pager-btn" id="pagePrev" aria-label="Previous page">Prev</button>
                <span class="pager-meta" id="pageMeta">Page 1</span>
                <button type="button" class="pager-btn" id="pageNext" aria-label="Next page">Next</button>
            </div>
        </div>
    </div>

    <script>
        const STORAGE_KEY = 'modelExplorer.public.v1';
        const CACHE_KEY = 'modelExplorer.snapshotCache.v1';
        const CACHE_TTL_MS = 60 * 60 * 1000;
        const BYTES_PER_PARAM = __BYTES_PER_PARAM_JSON__;
        const QUANT_BITS = __QUANT_BITS_JSON__;
        const SIZE_BAND_EDGES = __BAND_EDGES_JSON__;
        const SEED_MAX_B = __SEED_MAX_B__;
        const MIN_PARAMS_B = 0;
        const BW_PRESETS = [
            { label: '68 (M1)', gbps: 68 },
            { label: '100 (M2 / M3)', gbps: 100 },
            { label: '120 (M4)', gbps: 120 },
            { label: '150 (M3 Pro)', gbps: 150 },
            { label: '153 (M5)', gbps: 153 },
            { label: '200 (M1/M2 Pro)', gbps: 200 },
            { label: '256 (Strix Halo)', gbps: 256 },
            { label: '273 (M4 Pro / Spark)', gbps: 273 },
            { label: '300 (M3 Max 14c)', gbps: 300 },
            { label: '307 (M5 Pro)', gbps: 307 },
            { label: '400 (M1/M2 Max, M3 Max 16c)', gbps: 400 },
            { label: '410 (M4 Max 14c)', gbps: 410 },
            { label: '460 (M5 Max 32c)', gbps: 460 },
            { label: '546 (M4 Max 16c)', gbps: 546 },
            { label: '614 (M5 Max 40c)', gbps: 614 },
            { label: '800 (M1/M2 Ultra)', gbps: 800 },
            { label: '819 (M3 Ultra)', gbps: 819 },
            { label: '936 (RTX 3090)', gbps: 936 },
            { label: '1008 (RTX 4090)', gbps: 1008 },
            { label: '1792 (RTX 5090)', gbps: 1792 },
            { label: '2039 (A100 SXM)', gbps: 2039 },
            { label: '3350 (H100 SXM)', gbps: 3350 },
            { label: '4800 (H200)', gbps: 4800 },
            { label: '8000 (B200)', gbps: 8000 },
            { label: 'Custom', gbps: 0 }
        ];
        const CLUSTER_PRESET = [
            { name: 'Node 1', gb: 128, usable: null, bw: 800 },
            { name: 'Node 2', gb: 64, usable: null, bw: 400 },
            { name: 'Node 3', gb: 24, usable: null, bw: 400 }
        ];
        // 72× B200 @ 192 GB = 13.824 TB HBM; single pool (NVLink domain), per-GPU BW ceiling.
        const NVL72_PRESET = [
            { name: 'GB200 NVL72', gb: 13824, usable: 13824, bw: 8000 }
        ];
        const DEFAULT_HOSTS = [{ name: 'My machine', gb: 16, usable: null, bw: 400 }];

        // Transport model: effective GB/s = gbps * eff; latMs = per-hop one-way latency.
        // Thunderbolt clustering (MLX distributed / exo) is IP-over-TB bridge — TCP, not RDMA.
        // RDMA (RoCE) assumed only for 25GbE+ NICs; 1/10GbE modeled as TCP RPC.
        // eff reflects real payload rates: USB4 tunnels ~32 Gbps data of the 40 Gbps
        // line rate (PCIe 3.0 x4), USB4v2 ~64 of 80 (PCIe 4.0 x4); v2 asymmetric mode
        // is 120/40 Gbps per the USB-IF spec.
        const LINKS = {
            tb4:      { gbps: 5,     eff: 0.80, latMs: 0.40 },  // ~4 GB/s data
            tb5:      { gbps: 10,    eff: 0.80, latMs: 0.40 },  // ~8 GB/s data
            tb5a:     { gbps: 15,    eff: 0.80, latMs: 0.40 },  // 120G asym, ~12 GB/s data
            '1gbe':   { gbps: 0.125, eff: 0.70, latMs: 0.30 },
            '10gbe':  { gbps: 1.25,  eff: 0.70, latMs: 0.30 },
            '25gbe':  { gbps: 3.125, eff: 0.90, latMs: 0.05 },
            '100gbe': { gbps: 12.5,  eff: 0.90, latMs: 0.05 },
            '200gbe': { gbps: 25,    eff: 0.90, latMs: 0.05 },
            '400gbe': { gbps: 50,    eff: 0.90, latMs: 0.05 },
            '800gbe': { gbps: 100,   eff: 0.90, latMs: 0.05 }
        };

        function kvBytesPerToken(sizeB) {
            if (sizeB == null) return 0.000128;
            if (sizeB < 4) return 0.000064;
            if (sizeB < 15) return 0.000128;
            if (sizeB < 40) return 0.0002;
            if (sizeB < 80) return 0.00032;
            return 0.0005;
        }

        const hostsList = document.getElementById('hostsList');
        const searchInput = document.getElementById('searchInput');
        const tableBody = document.getElementById('tableBody');
        const matchCount = document.getElementById('matchCount');
        const noResults = document.getElementById('noResults');
        const refreshBtn = document.getElementById('refreshBtn');
        const refreshStatus = document.getElementById('refreshStatus');
        const generatedAt = document.getElementById('generatedAt');
        const quantSelect = document.getElementById('quantSelect');
        const overheadInput = document.getElementById('overheadInput');
        const contextSelect = document.getElementById('contextSelect');
        const kvSelect = document.getElementById('kvSelect');
        const linkSelect = document.getElementById('linkSelect');
        const totalGbReadout = document.getElementById('totalGbReadout');
        const minBReadout = document.getElementById('minBReadout');
        const maxBReadout = document.getElementById('maxBReadout');
        const boundReadout = document.getElementById('boundReadout');
        const dateWindowHint = document.getElementById('dateWindowHint');
        const btnGguf = document.getElementById('btnGguf');
        const btnMlx = document.getElementById('btnMlx');
        const pager = document.getElementById('pager');
        const pageSizeSelect = document.getElementById('pageSizeSelect');
        const pagePrev = document.getElementById('pagePrev');
        const pageNext = document.getElementById('pageNext');
        const pageMeta = document.getElementById('pageMeta');

        const PAGE_SIZE_OPTIONS = [50, 100, 150];
        let state = {
            format: 'gguf',
            hosts: DEFAULT_HOSTS.map(h => ({ ...h })),
            quant: 'Q4_K_M',
            overheadPct: 20,
            context: 4096,
            kvBytes: 2,
            link: '100gbe',
            pageSize: 50,
            page: 1
        };
        let currentSortColumn = 4;
        let isAscending = false;

        function loadState() {
            try {
                const raw = localStorage.getItem(STORAGE_KEY);
                if (!raw) return;
                const saved = JSON.parse(raw);
                if (saved.format === 'gguf' || saved.format === 'mlx') state.format = saved.format;
                if (Array.isArray(saved.hosts) && saved.hosts.length) {
                    state.hosts = saved.hosts.map(h => ({
                        name: String(h.name || ''),
                        gb: Number(h.gb) || 0,
                        usable: Number(h.usable) > 0 ? Number(h.usable) : null,
                        bw: Number(h.bw) || 400
                    }));
                }
                if (saved.quant && BYTES_PER_PARAM[saved.quant]) state.quant = saved.quant;
                if (typeof saved.overheadPct === 'number') state.overheadPct = saved.overheadPct;
                if (typeof saved.context === 'number') state.context = saved.context;
                if (typeof saved.kvBytes === 'number') state.kvBytes = saved.kvBytes;
                if (typeof saved.link === 'string' && LINKS[saved.link]) state.link = saved.link;
                if (PAGE_SIZE_OPTIONS.includes(Number(saved.pageSize))) state.pageSize = Number(saved.pageSize);
                if (typeof saved.page === 'number' && saved.page >= 1) state.page = Math.floor(saved.page);
            } catch (_) {}
        }
        function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }

        function isAppleish(name) {
            return /mac|m1|m2|m3|m4|apple|studio|mini/i.test(name || '');
        }
        function hostUsableGb(h) {
            const raw = Number(h.gb) || 0;
            // Explicit override wins — for tuned machines (iogpu.wired_limit_mb on
            // Apple Silicon, GTT/UMA carve-out on Strix Halo) that free up nearly all RAM.
            const override = Number(h.usable);
            if (override > 0) return Math.min(override, raw || override);
            if (isAppleish(h.name)) return raw * 0.75;
            const overhead = Math.min(50, Math.max(0, state.overheadPct)) / 100;
            return raw * (1 - overhead);
        }
        function totalUsableGb() {
            return state.hosts.reduce((s, h) => s + hostUsableGb(h), 0);
        }
        function bpp() { return BYTES_PER_PARAM[state.quant] || BYTES_PER_PARAM.Q4_K_M; }
        function wantedBits() { return QUANT_BITS[state.quant] || null; }

        function autoMapRange() {
            const maxB = Math.max(1, Math.round(totalUsableGb() / bpp()));
            return { minB: MIN_PARAMS_B, maxB, totalGb: totalUsableGb() };
        }

        function updateDerivedReadouts() {
            const { minB, maxB, totalGb } = autoMapRange();
            totalGbReadout.textContent = `${totalGb.toFixed(1)} GB usable`;
            minBReadout.textContent = `${minB}B`;
            maxBReadout.textContent = `${maxB}B`;
        }

        function renderHosts() {
            hostsList.innerHTML = '';
            state.hosts.forEach((host, index) => {
                const row = document.createElement('div');
                row.className = 'host-row';

                const nameInput = document.createElement('input');
                nameInput.type = 'text';
                nameInput.placeholder = 'Host name';
                nameInput.value = host.name;
                nameInput.addEventListener('input', () => {
                    state.hosts[index].name = nameInput.value;
                    saveState(); updateDerivedReadouts(); applySearch(true);
                });

                const gbInput = document.createElement('input');
                gbInput.type = 'number'; gbInput.min = '0'; gbInput.step = '1';
                gbInput.placeholder = 'GB'; gbInput.value = host.gb;
                gbInput.addEventListener('input', () => {
                    state.hosts[index].gb = Number(gbInput.value) || 0;
                    saveState(); updateDerivedReadouts(); applySearch(true);
                });

                const usableInput = document.createElement('input');
                usableInput.type = 'number'; usableInput.min = '0'; usableInput.step = '1';
                usableInput.placeholder = 'auto';
                usableInput.title = 'Override usable GB (e.g. raised iogpu.wired_limit_mb). Blank = auto (Apple 75%, else overhead %).';
                usableInput.value = host.usable != null ? host.usable : '';
                usableInput.addEventListener('input', () => {
                    const v = Number(usableInput.value);
                    state.hosts[index].usable = v > 0 ? v : null;
                    saveState(); updateDerivedReadouts(); applySearch(true);
                });

                const bwSelect = document.createElement('select');
                BW_PRESETS.forEach(p => {
                    const opt = document.createElement('option');
                    opt.value = String(p.gbps);
                    opt.textContent = p.label;
                    bwSelect.appendChild(opt);
                });
                const match = BW_PRESETS.find(p => p.gbps === host.bw);
                bwSelect.value = match ? String(host.bw) : '0';
                if (!match) {
                    const custom = document.createElement('option');
                    custom.value = String(host.bw);
                    custom.textContent = `${host.bw} GB/s`;
                    bwSelect.appendChild(custom);
                    bwSelect.value = String(host.bw);
                }
                bwSelect.addEventListener('change', () => {
                    const v = Number(bwSelect.value);
                    if (v === 0) {
                        const entered = prompt('Custom bandwidth (GB/s)', String(host.bw || 400));
                        state.hosts[index].bw = Number(entered) || 400;
                    } else {
                        state.hosts[index].bw = v;
                    }
                    saveState(); applySearch();
                    renderHosts();
                });

                const removeBtn = document.createElement('button');
                removeBtn.type = 'button'; removeBtn.className = 'icon-btn danger';
                removeBtn.title = 'Remove host'; removeBtn.textContent = '×';
                removeBtn.disabled = state.hosts.length <= 1;
                removeBtn.addEventListener('click', () => {
                    if (state.hosts.length <= 1) return;
                    state.hosts.splice(index, 1);
                    saveState(); renderHosts(); updateDerivedReadouts(); applySearch(true);
                });

                row.append(nameInput, gbInput, usableInput, bwSelect, removeBtn);
                hostsList.appendChild(row);
            });
        }

        function updateQuantAvailability() {
            [...quantSelect.options].forEach(opt => {
                const v = opt.value;
                if (state.format === 'mlx') {
                    opt.hidden = !(v.startsWith('MLX') || v === 'MXFP4' || v === 'FP8');
                    opt.disabled = opt.hidden;
                } else {
                    opt.hidden = v.startsWith('MLX');
                    opt.disabled = opt.hidden;
                }
            });
            if (quantSelect.querySelector(`option[value="${state.quant}"]`)?.hidden) {
                state.quant = state.format === 'mlx' ? 'MLX4' : 'Q4_K_M';
            }
            quantSelect.value = state.quant;
        }

        function setFormat(format) {
            state.format = format;
            btnGguf.classList.toggle('active', format === 'gguf');
            btnMlx.classList.toggle('active', format === 'mlx');
            updateQuantAvailability();
            saveState(); updateDerivedReadouts(); applySearch(true);
        }

        function extractBits(modelId) {
            const id = String(modelId || '').toLowerCase();
            let m = id.match(/(?:^|[-_/])mlx-?([2-8])bit(?:$|[-_/])/);
            if (m) return Number(m[1]);
            m = id.match(/[-_]([2-8])bit(?:$|[-_/])/);
            if (m) return Number(m[1]);
            m = id.match(/(?:^|[-_/])(?:q|iq)([4568])(?:[_-]|$)/);
            if (m) return Number(m[1]);
            if (/(?:^|[-_/])(?:bf16|bfloat16)(?:$|[-_/])/.test(id) || /[-_]f16(?:$|[-_/])/.test(id)) return 16;
            if (/(?:^|[-_/])(?:fp8|f8)(?:$|[-_/])/.test(id)) return 8;
            if (/(?:^|[-_/])mxfp4(?:$|[-_/])/.test(id)) return 4;
            return null;
        }

        function formatSizeLabel(sizeB) {
            if (sizeB == null || Number.isNaN(sizeB)) return '—';
            if (sizeB < 1) return `${Math.round(sizeB * 1000)}M`;
            return `${Math.round(sizeB * 10) / 10}B`;
        }

        function rowSizeB(row) {
            if (row.dataset.sizeB !== undefined && row.dataset.sizeB !== '') return Number(row.dataset.sizeB);
            return null;
        }
        function rowActiveB(row) {
            if (row.dataset.activeB !== undefined && row.dataset.activeB !== '') return Number(row.dataset.activeB);
            return rowSizeB(row);
        }

        function memoryRequiredGb(sizeB) {
            if (sizeB == null) return null;
            const weights = sizeB * bpp();
            const kv = kvBytesPerToken(sizeB) * state.context * state.kvBytes;
            const overhead = 1.5 + Math.min(4, weights * 0.02);
            return weights + kv + overhead;
        }

        function fitVerdict(sizeB) {
            const need = memoryRequiredGb(sizeB);
            if (need == null) return { label: '—', cls: '', rank: 0 };
            const have = totalUsableGb();
            if (need <= have * 0.9) return { label: 'Fits', cls: 'fit-ok', rank: 2 };
            if (need <= have) return { label: 'Tight', cls: 'fit-tight', rank: 1 };
            return { label: "Won't", cls: 'fit-no', rank: 0 };
        }

        function estimateTokS(sizeB, activeB) {
            if (sizeB == null) return null;
            const active = activeB != null ? activeB : sizeB;
            const weightBytes = active * 1e9 * bpp();
            const hosts = state.hosts.filter(h => (Number(h.gb) || 0) > 0);
            if (!hosts.length) return null;
            const n = hosts.length;

            // Layer-split pipeline, single-stream decode: token N+1 depends on token N,
            // so stage times SUM (no overlap). Shards are memory-proportional (that is
            // how exo / llama.cpp-RPC place layers).
            const totalMem = hosts.reduce((s, h) => s + (Number(h.gb) || 0), 0);
            const computeSec = hosts.reduce((s, h) => {
                const shard = weightBytes * ((Number(h.gb) || 0) / totalMem);
                const bw = (Number(h.bw) || 400) * 1e9;
                return s + shard / bw;
            }, 0);

            // Per token, each stage boundary moves ~one hidden state over the link,
            // paying protocol efficiency + per-hop latency (TCP bridge vs RDMA).
            let networkSec = 0;
            if (n > 1) {
                const link = LINKS[state.link] || LINKS['100gbe'];
                const hiddenBytes = Math.min(16384, Math.max(2048, Math.round(Math.sqrt(active) * 512))) * 2;
                const perHop = hiddenBytes / (link.gbps * 1e9 * link.eff) + link.latMs / 1000;
                networkSec = (n - 1) * perHop;
            }

            const tps = 1 / (computeSec + networkSec);
            return {
                tps,
                computeTps: 1 / computeSec,
                networkTps: networkSec > 0 ? 1 / networkSec : null
            };
        }

        function updateBoundReadout() {
            const est = estimateTokS(70, 70);
            if (!est) { boundReadout.textContent = 'Cluster bound: —'; return; }
            const hosts = state.hosts.filter(h => (Number(h.gb) || 0) > 0).length;
            if (hosts <= 1) {
                boundReadout.textContent = `Single-host decode ceiling @ 70B dense: ~${Math.round(est.tps)} tok/s (memory-bandwidth-bound)`;
            } else {
                // share of token time spent on the interconnect = tps / networkTps
                const netShare = est.networkTps ? Math.round((est.tps / est.networkTps) * 100) : 0;
                const linkLabel = linkSelect.options[linkSelect.selectedIndex]
                    ? linkSelect.options[linkSelect.selectedIndex].textContent.trim() : '';
                // Decode barely feels the link (one ~KB hidden state/token), but prefill
                // ships every prompt token's activations across each hop — that is where
                // slow links hurt (time-to-first-token), so surface it.
                const link = LINKS[state.link] || LINKS['100gbe'];
                const hops = state.hosts.filter(h => (Number(h.gb) || 0) > 0).length - 1;
                const hiddenBytes = Math.min(16384, Math.max(2048, Math.round(Math.sqrt(70) * 512))) * 2;
                const prefillSec = hops * (2048 * hiddenBytes / (link.gbps * 1e9 * link.eff) + link.latMs / 1000);
                const prefillLabel = prefillSec >= 0.05 ? `+${prefillSec.toFixed(1)}s` : 'negligible';
                boundReadout.textContent =
                    `Cluster @ 70B dense: ~${Math.round(est.tps)} tok/s · ` +
                    `interconnect ${netShare <= 1 ? '<1' : netShare}% of token time (${linkLabel}) · ` +
                    `2k-prompt prefill on link: ${prefillLabel}`;
            }
        }

        function getRows() { return Array.from(tableBody.getElementsByTagName('tr')); }

        function applySearch(resetPage) {
            if (resetPage) state.page = 1;
            const query = searchInput.value.toLowerCase().trim();
            const rows = getRows();
            const want = wantedBits();
            const { minB, maxB } = autoMapRange();
            const matched = [];

            rows.forEach(row => {
                const id = row.dataset.modelId || '';
                const bits = row.dataset.bits !== undefined && row.dataset.bits !== ''
                    ? Number(row.dataset.bits) : extractBits(id);
                let quantOk = true;
                if (want != null) {
                    if (bits == null) quantOk = state.format === 'gguf';
                    else quantOk = bits === want;
                }
                const sizeB = rowSizeB(row);
                const activeB = rowActiveB(row);
                const sizeLabel = formatSizeLabel(sizeB).toLowerCase();
                const fit = fitVerdict(sizeB);
                const est = estimateTokS(sizeB, activeB);
                const rangeOk = sizeB == null || (sizeB >= minB && sizeB <= maxB);
                const textOk = !query || id.includes(query) || sizeLabel.includes(query)
                    || (sizeB != null && String(sizeB).includes(query));
                const show = quantOk && rangeOk && textOk && fit.rank > 0;

                const fitEl = row.querySelector('.fit-text');
                if (fitEl) { fitEl.textContent = fit.label; fitEl.className = `fit-text ${fit.cls}`; }
                const tpsEl = row.querySelector('.tps-text');
                if (tpsEl) tpsEl.textContent = est ? `~${Math.round(est.tps)}` : '—';

                row.style.display = 'none';
                if (show) matched.push(row);
            });

            const total = matched.length;
            const pageSize = state.pageSize;
            const totalPages = Math.max(1, Math.ceil(total / pageSize) || 1);
            if (state.page > totalPages) state.page = totalPages;
            if (state.page < 1) state.page = 1;
            const start = (state.page - 1) * pageSize;
            const end = Math.min(start + pageSize, total);
            for (let i = start; i < end; i++) matched[i].style.display = '';

            if (total === 0) {
                matchCount.textContent = `Showing 0 ${state.quant} models that fit`;
            } else {
                matchCount.textContent =
                    `Showing ${start + 1}–${end} of ${total} ${state.quant} models that fit`;
            }
            noResults.style.display = (rows.length === 0 || total === 0) ? 'block' : 'none';
            pager.hidden = total === 0;
            pageMeta.textContent = total === 0 ? 'Page 0 of 0' : `Page ${state.page} of ${totalPages}`;
            pagePrev.disabled = state.page <= 1;
            pageNext.disabled = state.page >= totalPages || total === 0;
            updateBoundReadout();
        }

        function sortTable(columnIndex) {
            const rowsArray = getRows();
            if (currentSortColumn === columnIndex) isAscending = !isAscending;
            else { currentSortColumn = columnIndex; isAscending = columnIndex === 0; }
            for (let i = 0; i < 6; i++) {
                const arrow = document.getElementById('arrow' + i);
                if (arrow) arrow.textContent = i === columnIndex ? (isAscending ? ' ▲' : ' ▼') : '';
            }
            rowsArray.sort((a, b) => {
                if (columnIndex === 0) {
                    return isAscending
                        ? a.dataset.modelId.localeCompare(b.dataset.modelId)
                        : b.dataset.modelId.localeCompare(a.dataset.modelId);
                }
                if (columnIndex === 1) {
                    const na = rowSizeB(a) ?? -1, nb = rowSizeB(b) ?? -1;
                    return isAscending ? na - nb : nb - na;
                }
                if (columnIndex === 2) {
                    const na = fitVerdict(rowSizeB(a)).rank, nb = fitVerdict(rowSizeB(b)).rank;
                    return isAscending ? na - nb : nb - na;
                }
                if (columnIndex === 3) {
                    const ea = estimateTokS(rowSizeB(a), rowActiveB(a));
                    const eb = estimateTokS(rowSizeB(b), rowActiveB(b));
                    const na = ea ? ea.tps : -1, nb = eb ? eb.tps : -1;
                    return isAscending ? na - nb : nb - na;
                }
                if (columnIndex === 4) {
                    const na = parseInt(a.querySelector('.badge-downloads').textContent.replace(/,/g, ''), 10) || 0;
                    const nb = parseInt(b.querySelector('.badge-downloads').textContent.replace(/,/g, ''), 10) || 0;
                    return isAscending ? na - nb : nb - na;
                }
                const da = a.querySelector('.date-text').textContent;
                const db = b.querySelector('.date-text').textContent;
                return isAscending ? da.localeCompare(db) : db.localeCompare(da);
            });
            tableBody.innerHTML = '';
            rowsArray.forEach(r => tableBody.appendChild(r));
            applySearch(true);
        }

        function buildRow(rec) {
            const high = rec.downloads > 1000 ? 'high' : '';
            const tr = document.createElement('tr');
            tr.dataset.modelId = rec.id.toLowerCase();
            if (rec.bits != null) tr.dataset.bits = String(rec.bits);
            if (rec.sizeB != null) tr.dataset.sizeB = String(rec.sizeB);
            if (rec.activeB != null) tr.dataset.activeB = String(rec.activeB);
            tr.innerHTML = `
                <td><a class="model-link" href="https://huggingface.co/${rec.id}" target="_blank" rel="noopener">${rec.id}</a></td>
                <td style="text-align: right;"><span class="size-text">${formatSizeLabel(rec.sizeB)}</span></td>
                <td style="text-align: right;"><span class="fit-text">—</span></td>
                <td style="text-align: right;"><span class="tps-text">—</span></td>
                <td style="text-align: right;"><span class="badge-downloads ${high}">${(rec.downloads || 0).toLocaleString()}</span></td>
                <td style="text-align: right;"><span class="date-text">${rec.date || ''}</span></td>
            `;
            return tr;
        }

        function sizeBands(maxB) {
            const bands = [];
            const cap = Math.min(maxB, SEED_MAX_B);
            for (let i = 0; i < SIZE_BAND_EDGES.length - 1; i++) {
                const lo = SIZE_BAND_EDGES[i];
                if (lo >= cap) break;
                bands.push([lo, Math.min(SIZE_BAND_EDGES[i + 1], cap)]);
            }
            return bands;
        }

        function readCache() {
            try { return JSON.parse(localStorage.getItem(CACHE_KEY)) || {}; }
            catch (_) { return {}; }
        }
        function writeCache(cache) {
            try { localStorage.setItem(CACHE_KEY, JSON.stringify(cache)); }
            catch (_) { try { localStorage.removeItem(CACHE_KEY); } catch (_) {} }
        }
        function bandKey(lo, hi) { return `${state.format}|${lo}-${hi}`; }

        async function fetchBandSnapshot(lo, hi) {
            const url = `./data/${state.format}/${lo}-${hi}.json`;
            const response = await fetch(url, { cache: 'no-cache' });
            if (!response.ok) throw new Error(`Snapshot ${response.status}: ${url}`);
            return response.json();
        }

        async function loadSnapshots(force, onProgress) {
            // Always load the full published catalog (SEED_MAX_B). Fit filtering still
            // uses autoMapRange — large clusters (e.g. NVL72 @ 13.8 TB) must not request
            // overflow bands that the crawl never wrote.
            const bands = sizeBands(SEED_MAX_B);
            const cache = readCache();
            const now = Date.now();
            let done = 0, fetched = 0, missing = 0;
            const report = () => onProgress && onProgress(done, bands.length, fetched);

            const parts = await Promise.all(bands.map(async ([a, b]) => {
                const key = bandKey(a, b);
                const hit = cache[key];
                if (!force && hit && Array.isArray(hit.records) && now - hit.ts < CACHE_TTL_MS) {
                    done++; report();
                    return hit.records;
                }
                try {
                    const records = await fetchBandSnapshot(a, b);
                    cache[key] = { ts: now, records };
                    done++; fetched++; report();
                    return records;
                } catch (err) {
                    done++; report();
                    if (hit && Array.isArray(hit.records)) return hit.records;
                    missing++;
                    console.warn(err);
                    return [];
                }
            }));
            writeCache(cache);
            if (missing && missing === bands.length) {
                throw new Error('All snapshot bands unavailable');
            }
            const byId = new Map();
            for (const part of parts) for (const rec of part) byId.set(rec.id, rec);
            return { records: Array.from(byId.values()), fetched, missing };
        }

        function renderRecords(records) {
            const sorted = records.slice().sort((a, b) => (b.downloads || 0) - (a.downloads || 0));
            tableBody.innerHTML = '';
            sorted.forEach(rec => tableBody.appendChild(buildRow(rec)));
            currentSortColumn = 4; isAscending = false;
            for (let i = 0; i < 6; i++) {
                const arrow = document.getElementById('arrow' + i);
                if (arrow) arrow.textContent = i === 4 ? ' ▼' : '';
            }
            applySearch(true);
        }

        async function refreshModels(event) {
            const force = !!(event && event.shiftKey);
            refreshBtn.disabled = true;
            refreshBtn.classList.add('spinning');
            refreshStatus.style.color = 'var(--muted)';
            refreshStatus.textContent = force ? 'Reloading snapshots…' : 'Refreshing…';
            try {
                const { records, fetched, missing } = await loadSnapshots(force, (done, total) => {
                    refreshStatus.textContent = `Bands ${done}/${total}…`;
                });
                renderRecords(records);
                try {
                    const meta = await fetch('./data/meta.json', { cache: 'no-cache' }).then(r => r.json());
                    if (meta.generated_at) {
                        generatedAt.textContent = `Data as of: ${meta.generated_at}`;
                        dateWindowHint.textContent = `Snapshot: ${meta.generated_at} · last 365 days of models`;
                    }
                } catch (_) {}
                if (missing) {
                    refreshStatus.textContent = `Loaded with ${missing} band(s) missing`;
                    refreshStatus.style.color = 'var(--warn)';
                } else {
                    refreshStatus.textContent = fetched ? '' : 'All bands from cache — Shift-click to force';
                }
            } catch (err) {
                console.error(err);
                refreshStatus.textContent = 'Snapshot unavailable — keeping previous results';
                refreshStatus.style.color = 'var(--danger)';
            } finally {
                refreshBtn.disabled = false;
                refreshBtn.classList.remove('spinning');
            }
        }

        btnGguf.addEventListener('click', () => setFormat('gguf'));
        btnMlx.addEventListener('click', () => setFormat('mlx'));
        quantSelect.addEventListener('change', () => {
            state.quant = quantSelect.value; saveState(); updateDerivedReadouts(); applySearch(true);
        });
        overheadInput.addEventListener('input', () => {
            state.overheadPct = Number(overheadInput.value) || 0;
            saveState(); updateDerivedReadouts(); applySearch(true);
        });
        contextSelect.addEventListener('change', () => {
            state.context = Number(contextSelect.value) || 4096;
            saveState(); applySearch(true);
        });
        kvSelect.addEventListener('change', () => {
            state.kvBytes = Number(kvSelect.value) || 2;
            saveState(); applySearch(true);
        });
        linkSelect.addEventListener('change', () => {
            state.link = LINKS[linkSelect.value] ? linkSelect.value : '100gbe';
            saveState(); applySearch();
        });
        document.getElementById('addHostBtn').addEventListener('click', () => {
            state.hosts.push({ name: '', gb: 0, usable: null, bw: 400 });
            saveState(); renderHosts(); updateDerivedReadouts(); applySearch(true);
        });
        document.getElementById('loadClusterBtn').addEventListener('click', () => {
            state.hosts = CLUSTER_PRESET.map(h => ({ ...h }));
            saveState(); renderHosts(); updateDerivedReadouts(); applySearch(true);
        });
        document.getElementById('loadNvl72Btn').addEventListener('click', () => {
            state.hosts = NVL72_PRESET.map(h => ({ ...h }));
            saveState(); renderHosts(); updateDerivedReadouts(); applySearch(true);
        });
        document.getElementById('resetHostsBtn').addEventListener('click', () => {
            state.hosts = DEFAULT_HOSTS.map(h => ({ ...h }));
            saveState(); renderHosts(); updateDerivedReadouts(); applySearch(true);
        });
        searchInput.addEventListener('input', () => applySearch(true));
        pageSizeSelect.addEventListener('change', () => {
            state.pageSize = Number(pageSizeSelect.value) || 50;
            state.page = 1;
            saveState();
            applySearch();
        });
        pagePrev.addEventListener('click', () => {
            if (state.page <= 1) return;
            state.page -= 1;
            saveState();
            applySearch();
        });
        pageNext.addEventListener('click', () => {
            state.page += 1;
            saveState();
            applySearch();
        });
        refreshBtn.addEventListener('click', refreshModels);

        loadState();
        overheadInput.value = state.overheadPct;
        contextSelect.value = String(state.context);
        kvSelect.value = String(state.kvBytes);
        linkSelect.value = state.link;
        pageSizeSelect.value = String(state.pageSize);
        setFormat(state.format);
        renderHosts();
        updateDerivedReadouts();
        applySearch();
    </script>
</body>
</html>
"""


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    if token:
        print("Using authenticated HF_TOKEN for crawl.")
    else:
        print("No HF_TOKEN set — crawling anonymously (fine for local builds).")

    total_gb = sum(h[1] for h in DEFAULT_HOSTS)
    min_b, max_b = auto_map_range(total_gb)
    print(
        f"Default host map @ {DEFAULT_QUANT}: {min_b}B–{max_b}B "
        f"(usable ~{total_gb * (1 - DEFAULT_OVERHEAD):.0f} GB)"
    )
    print(f"Seeding snapshots through {SEED_MAX_B}B…")

    snapshots, bands = fetch_all_snapshots(api, SEED_MAX_B)

    by_id = {}
    for records in snapshots["gguf"].values():
        for rec in records:
            by_id[rec["id"]] = rec
    seed = sorted(by_id.values(), key=lambda r: r["downloads"], reverse=True)
    print(f"Seed rows: {len(seed)} GGUF · bands: {len(bands)} · formats: gguf+mlx")

    site = write_site(snapshots, bands, seed, min_b, max_b)
    print(f"\nSite written to:\n  {site}/index.html\n  {site}/data/")
    print(f"Local preview:\n  cd {os.path.dirname(site)} && python -m http.server 8765 --directory _site")


if __name__ == "__main__":
    main()
