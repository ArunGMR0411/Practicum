"""Face Anonymization Made Simple adapter."""

from __future__ import annotations

from pathlib import Path

from src.anonymisation.external_command_anonymiser import PROJECT_ROOT, ExternalCommandAnonymiser


DEFAULT_BACKEND_ROOT = PROJECT_ROOT / "third_party" / "face_anon_simple"
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "anonymisation" / "run_fams_backend.py"
DEFAULT_MODEL_ID = "hkung/face-anon-simple"
DEFAULT_BASE_MODEL_ID = str(PROJECT_ROOT / "data" / "models" / "stable-diffusion-inpainting")
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
REQUIRED_IMPORTS = (
    "torch",
    "diffusers",
    "transformers",
    "face_alignment",
    "cv2",
    "huggingface_hub",
)


class FAMSAnonymiser(ExternalCommandAnonymiser):
    """Run Face Anonymization Made Simple through an external backend."""

    method_name = "fams"

    def __init__(
        self,
        *,
        backend_root: Path = DEFAULT_BACKEND_ROOT,
        runner_path: Path = DEFAULT_RUNNER,
        model_id: str = DEFAULT_MODEL_ID,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
        python_executable: str | None = None,
        face_image_size: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        anonymization_degree: float = 1.25,
        overlap_iou_threshold: float = 0.15,
        seed: int = 0,
        enable_model_cpu_offload: bool = True,
    ) -> None:
        extra_args = [
            "--base-model-id",
            base_model_id,
            "--clip-model-id",
            clip_model_id,
            "--face-image-size",
            str(face_image_size),
            "--num-inference-steps",
            str(num_inference_steps),
            "--guidance-scale",
            str(guidance_scale),
            "--anonymization-degree",
            str(anonymization_degree),
            "--overlap-iou-threshold",
            str(overlap_iou_threshold),
            "--seed",
            str(seed),
        ]
        if enable_model_cpu_offload:
            extra_args.append("--enable-model-cpu-offload")
        kwargs = {
            "backend_root": backend_root,
            "runner_path": runner_path,
            "model_id": model_id,
            "extra_args": extra_args,
            "required_imports": REQUIRED_IMPORTS,
            "required_paths": (
                Path(backend_root) / "utils" / "anonymize_faces_in_image.py",
                Path(backend_root) / "utils" / "extractor.py",
                Path(backend_root) / "utils" / "merger.py",
                Path(backend_root) / "src" / "diffusers" / "pipelines" / "referencenet" / "pipeline_referencenet.py",
            ),
        }
        if python_executable is not None:
            kwargs["python_executable"] = python_executable
        super().__init__(**kwargs)
