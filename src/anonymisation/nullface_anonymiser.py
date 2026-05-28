"""NullFace adapter."""

from __future__ import annotations

from pathlib import Path

from src.anonymisation.external_command_anonymiser import PROJECT_ROOT, ExternalCommandAnonymiser


DEFAULT_BACKEND_ROOT = PROJECT_ROOT / "third_party" / "nullface"
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "anonymisation" / "run_nullface_backend.py"
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_INSIGHTFACE_ROOT = str(Path.home() / ".insightface")
REQUIRED_IMPORTS = (
    "torch",
    "diffusers",
    "transformers",
    "cv2",
    "insightface",
)


class NullFaceAnonymiser(ExternalCommandAnonymiser):
    """Run NullFace through an external backend."""

    method_name = "nullface"

    def __init__(
        self,
        *,
        backend_root: Path = DEFAULT_BACKEND_ROOT,
        runner_path: Path = DEFAULT_RUNNER,
        model_id: str = DEFAULT_MODEL_ID,
        insightface_model_path: str = DEFAULT_INSIGHTFACE_ROOT,
        crop_padding_ratio: float = 0.9,
        guidance_scale: float = 10.0,
        num_diffusion_steps: int = 60,
        eta: float = 1.0,
        skip: int = 40,
        ip_adapter_scale: float = 1.0,
        id_emb_scale: float = 1.0,
        det_thresh: float = 0.1,
        det_size: int = 640,
        seed: int = 0,
        mask_delay_steps: int = 10,
        python_executable: str | None = None,
    ) -> None:
        extra_args = [
            "--insightface-model-path",
            insightface_model_path,
            "--crop-padding-ratio",
            str(crop_padding_ratio),
            "--guidance-scale",
            str(guidance_scale),
            "--num-diffusion-steps",
            str(num_diffusion_steps),
            "--eta",
            str(eta),
            "--skip",
            str(skip),
            "--ip-adapter-scale",
            str(ip_adapter_scale),
            "--id-emb-scale",
            str(id_emb_scale),
            "--det-thresh",
            str(det_thresh),
            "--det-size",
            str(det_size),
            "--seed",
            str(seed),
            "--mask-delay-steps",
            str(mask_delay_steps),
        ]
        kwargs = {
            "backend_root": backend_root,
            "runner_path": runner_path,
            "model_id": model_id,
            "extra_args": extra_args,
            "required_imports": REQUIRED_IMPORTS,
            "required_paths": (
                Path(backend_root) / "anonymize_face.py",
                Path(backend_root) / "ddm_inversion" / "inversion_utils.py",
                Path(backend_root) / "prompt_to_prompt" / "ptp_utils.py",
                Path(backend_root) / "utils" / "face_embedding.py",
            ),
        }
        if python_executable is not None:
            kwargs["python_executable"] = python_executable
        super().__init__(**kwargs)
