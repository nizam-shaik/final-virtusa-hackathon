# utils/diagram_renderer.py
# Backend-only renderer. One public function: render_diagrams(...)
from __future__ import annotations
from pathlib import Path
from typing import Dict
import os

def render_diagrams(
    mermaid_map: Dict[str, str],
    out_dir: str,
    want_images: bool = True,
    image_fmt: str = "png",       # "svg" | "png"
    renderer: str = "mmdc",       # "kroki" | "mmdc" - prefer mmdc for better reliability
    save_sources: bool = True,    # write <n>.mmd
) -> Dict[str, Dict[str, str]]:
    """
    Writes .mmd sources and (optionally) renders images into <out_dir>/img/.
    Returns: {"mmd": {name: path}, "images": {name: path}}
    """

    # --- helpers INSIDE the function (no extra public APIs) ---

    def _write_sources(_base: Path) -> Dict[str, str]:
        out = {}
        if not save_sources:
            return out
        for name, code in mermaid_map.items():
            p = _base / f"{name}.mmd"
            p.write_text(code, encoding="utf-8")
            out[name] = str(p)
        return out

    def _render_images_kroki(_img_dir: Path) -> Dict[str, str]:
        import requests  # local import
        out = {}
        kroki_url = os.getenv("KROKI_URL", "https://kroki.io").rstrip("/")
        for name, code in mermaid_map.items():
            endpoint = f"{kroki_url}/mermaid/{image_fmt}"
            # simple retry with backoff
            timeout = int(os.getenv("KROKI_TIMEOUT", "40"))
            retries = int(os.getenv("KROKI_RETRIES", "2"))
            last_err = None
            for attempt in range(retries + 1):
                try:
                    resp = requests.post(endpoint, data=code.encode("utf-8"), timeout=timeout)
                    resp.raise_for_status()
                    path = _img_dir / f"{name}.{image_fmt}"
                    path.write_bytes(resp.content)
                    out[name] = str(path)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
            # if still failing after retries, skip image for this diagram
            if last_err is not None:
                # swallow error and continue with Mermaid source only
                continue
        return out

    def _render_images_mmdc(_img_dir: Path) -> Dict[str, str]:
        import tempfile, subprocess  # local import
        out = {}
        for name, code in mermaid_map.items():
            with tempfile.TemporaryDirectory() as td:
                in_mmd = Path(td) / "diagram.mmd"
                in_mmd.write_text(code, encoding="utf-8")
                out_img = _img_dir / f"{name}.{image_fmt}"
                cmd = ["mmdc", "-i", str(in_mmd), "-o", str(out_img)]
                if image_fmt == "png":
                    cmd += ["-t", "default"]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out[name] = str(out_img)
                except Exception:
                    # swallow mmdc errors; continue with Mermaid source only
                    continue
        return out

    # --- main flow ---
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, str]] = {"mmd": {}, "images": {}}
    results["mmd"] = _write_sources(base)

    if not want_images:
        return results

    img_dir = base / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    if renderer == "kroki":
        results["images"] = _render_images_kroki(img_dir)
    elif renderer == "mmdc":
        results["images"] = _render_images_mmdc(img_dir)
    else:
        raise ValueError("Unknown renderer (use 'kroki' or 'mmdc')")

    return results