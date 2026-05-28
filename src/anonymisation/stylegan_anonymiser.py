"""StyleGAN-family face anonymiser adapter."""

from __future__ import annotations

from pathlib import Path

from src.anonymisation.external_command_anonymiser import PROJECT_ROOT, ExternalCommandAnonymiser


DEFAULT_BACKEND_ROOT = PROJECT_ROOT / "third_party" / "styleid"
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "anonymisation" / "run_styleid_backend.py"
DEFAULT_MODEL_PATH = DEFAULT_BACKEND_ROOT / "pretrained_models" / "stylegan2-ffhq-config-f.pt"
REQUIRED_IMPORTS = ("torch", "numpy", "PIL")


class StyleGANAnonymiser(ExternalCommandAnonymiser):
    """Run a StyleGAN/e4e-family anonymiser through an external backend."""

    method_name = "stylegan"

    def __init__(
        self,
        *,
        backend_root: Path = DEFAULT_BACKEND_ROOT,
        runner_path: Path = DEFAULT_RUNNER,
        model_path: Path = DEFAULT_MODEL_PATH,
        python_executable: str | None = None,
        truncation_psi: float = 0.7,
        seed: int = 0,
    ) -> None:
        extra_args = ["--truncation-psi", str(truncation_psi), "--seed", str(seed)]
        required_assets = (
            Path(backend_root) / "pretrained_models" / "psp_celebs_seg_to_face.pt",
            Path(backend_root) / "pretrained_models" / "CurricularFace_Backbone.pth",
            Path(backend_root) / "pretrained_models" / "mobilenet_celeba.pth",
            Path(backend_root) / "pretrained_models" / "unet_model.pth",
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
