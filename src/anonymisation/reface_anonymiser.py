"""Reversible face anonymiser adapter backed by G2Face when assets are installed."""

from __future__ import annotations

from pathlib import Path

from src.anonymisation.external_command_anonymiser import PROJECT_ROOT, ExternalCommandAnonymiser


DEFAULT_BACKEND_ROOT = PROJECT_ROOT / "third_party" / "g2face"
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "anonymisation" / "run_g2face_backend.py"
DEFAULT_MODEL_PATH = DEFAULT_BACKEND_ROOT / "weights" / "G2Face.pth"
REQUIRED_IMPORTS = ("torch", "numpy", "PIL", "kornia", "nvdiffrast")


class RefaceAnonymiser(ExternalCommandAnonymiser):
    """Run reversible face anonymisation through the official G2Face backend."""

    method_name = "reface"

    def __init__(
        self,
        *,
        backend_root: Path = DEFAULT_BACKEND_ROOT,
        runner_path: Path = DEFAULT_RUNNER,
        model_path: Path = DEFAULT_MODEL_PATH,
        python_executable: str | None = None,
        donor_strategy: str = "nearest_non_self",
        seed: int = 0,
    ) -> None:
        extra_args = ["--donor-strategy", donor_strategy, "--seed", str(seed)]
        required_assets = (
            Path(backend_root) / "weights" / "epoch_20.pth",
            Path(backend_root) / "pretrain" / "ms1mv3_arcface_r50.pth",
            Path(backend_root) / "model" / "d3dfr" / "BFM" / "BFM_model_front.mat",
        )
        kwargs = {
            "backend_root": backend_root,
            "runner_path": runner_path,
            "model_path": model_path,
            "extra_args": extra_args,
            "required_imports": REQUIRED_IMPORTS,
            "required_paths": required_assets,
        }
        if python_executable is not None:
            kwargs["python_executable"] = python_executable
        super().__init__(**kwargs)
