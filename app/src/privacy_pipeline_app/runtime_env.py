"""Bootstrap App runtime paths so research backends work without manual exports.

Called automatically from ``run_web.py``, ``run_cli.py``, and Gradio ``main()``.
Defaults match the project tree layout; existing env vars are never overwritten.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_CONFIGURED = False

# app/src/privacy_pipeline_app/runtime_env.py -> parents[3] = repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_SRC = PROJECT_ROOT / "app" / "src"


def configure_app_runtime(*, force: bool = False) -> dict[str, str]:
    """Ensure PYTHONPATH, CUDA toolkit, and research backend defaults are set."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return _snapshot()

    # Package import paths.
    for path in (str(PROJECT_ROOT), str(APP_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)

    # Prefer project venv binaries (ninja, etc.)
    venv_bin = PROJECT_ROOT / ".venv" / "bin"
    if venv_bin.is_dir():
        os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")

    # CUDA toolkit shipped with the torch nvidia wheel (needed by FALCO/GenForce).
    # Prefer a full toolkit dir (nvcc + include). Avoid partial nvidia/* bins
    # (e.g. cusparselt) which break torch cpp_extension builds.
    site = PROJECT_ROOT / ".venv" / "lib"
    preferred: Path | None = None
    for cand in sorted(site.glob("python*/site-packages/nvidia/cu*")):
        if (cand / "bin" / "nvcc").exists() and (cand / "include").is_dir():
            preferred = cand
            break
    if preferred is None:
        for cand in sorted(site.glob("python*/site-packages/nvidia/cu*")):
            if (cand / "include").is_dir():
                preferred = cand
                break
    if preferred is not None:
        # Always pin CUDA_HOME to the full toolkit when available (even if a bad
        # value was already exported from the shell).
        current = os.environ.get("CUDA_HOME", "")
        if (not current) or ("cusparselt" in current) or ("cublas" in current and "cu1" not in Path(current).name):
            os.environ["CUDA_HOME"] = str(preferred)
        nvcc_bin = preferred / "bin"
        if nvcc_bin.is_dir():
            os.environ["PATH"] = str(nvcc_bin) + os.pathsep + os.environ.get("PATH", "")


    defaults = {
        "RIDDLE_SOURCE_ROOT": str(PROJECT_ROOT / "third_party" / "riddle"),
        "RIDDLE_ASSET_ROOT": str(PROJECT_ROOT / "data" / "models" / "riddle"),
        "FALCO_SOURCE_ROOT": str(PROJECT_ROOT / "third_party" / "falco"),
    }
    for key, value in defaults.items():
        if not os.environ.get(key):
            # Only set when path exists so empty trees do not pretend to be configured
            if Path(value).exists():
                os.environ[key] = value

    # Repair common broken HF cache symlink for RF-DETR if a local copy exists.
    _ensure_rfdetr_checkpoint()

    _CONFIGURED = True
    return _snapshot()


def _ensure_rfdetr_checkpoint() -> None:
    """If the HF-cache RF-DETR symlink is broken but a local .pth exists, re-link it."""
    cache_link = (
        PROJECT_ROOT
        / "data/models/face_detection_candidates/rfdetr_hf_cache/"
        / "models--Herojayjay--RFDETR-Face-Detection/snapshots/"
        / "597fcce941997900080ce8127b53a5d24e330225/rfdetr_medium_face.pth"
    )
    sources = [
        PROJECT_ROOT / "data/models/face_detection_candidates/rfdetr_medium_face.pth",
        PROJECT_ROOT / "data/models/face_detection_candidates/rfdetr_download/rfdetr_medium_face.pth",
    ]
    try:
        if cache_link.exists() and cache_link.is_file() and cache_link.stat().st_size > 0:
            return
    except OSError:
        pass
    blob_dir = (
        PROJECT_ROOT
        / "data/models/face_detection_candidates/rfdetr_hf_cache/"
        / "models--Herojayjay--RFDETR-Face-Detection/blobs"
    )
    for src in sources:
        if not src.is_file() or src.stat().st_size <= 0:
            continue
        try:
            blob_dir.mkdir(parents=True, exist_ok=True)
            cache_link.parent.mkdir(parents=True, exist_ok=True)
            # Prefer a real file at the expected path (most robust for path.exists()).
            if cache_link.is_symlink() or not cache_link.exists():
                import shutil

                shutil.copy2(src, cache_link)
            return
        except OSError:
            continue


def _snapshot() -> dict[str, str]:
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "RIDDLE_SOURCE_ROOT": os.environ.get("RIDDLE_SOURCE_ROOT", ""),
        "RIDDLE_ASSET_ROOT": os.environ.get("RIDDLE_ASSET_ROOT", ""),
        "FALCO_SOURCE_ROOT": os.environ.get("FALCO_SOURCE_ROOT", ""),
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
    }
