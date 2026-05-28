"""Reverse Personalization anonymiser adapter backed by the official upstream project."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser, BoundingBox
from src.utils.system_config import read_accelerator_memory_gb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKEND_ROOT = PROJECT_ROOT / "third_party" / "reverse_personalization"
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "anonymisation" / "run_reverse_personalization.py"
DEFAULT_PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
REQUIRED_IMPORTS = ("torch", "diffusers", "transformers", "insightface", "face_alignment", "peft")
LOW_VRAM_THRESHOLD_GB = 10.0


class ReversePersonalizationAnonymiser(BaseAnonymiser):
    """Run face anonymisation through the official Reverse Personalization codebase."""

    method_name = "reverse_personalization"

    def __init__(
        self,
        *,
        backend_root: Path = DEFAULT_BACKEND_ROOT,
        runner_path: Path = DEFAULT_RUNNER,
        python_executable: str = str(DEFAULT_PROJECT_PYTHON if DEFAULT_PROJECT_PYTHON.is_file() else "python3"),
        attribute_prompt: str | None = None,
        sd_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        insightface_model_path: str = "~/.insightface",
        device_num: int = 0,
        skip: float = 0.7,
        id_emb_scale: float = 1.0,
        guidance_scale: float = -10.0,
        num_inversion_steps: int = 100,
        face_image_size: int = 1024,
        det_thresh: float = 0.1,
        ip_adapter_scale: float = 1.0,
        det_size: int = 640,
        seed: int = 0,
        enable_face_detection: bool = True,
        use_model_cpu_offload: bool | None = None,
    ) -> None:
        self.backend_root = Path(backend_root)
        self.runner_path = Path(runner_path)
        self.python_executable = python_executable
        self.attribute_prompt = attribute_prompt
        self.sd_model_path = sd_model_path
        self.insightface_model_path = insightface_model_path
        self.device_num = int(device_num)
        self.skip = float(skip)
        self.id_emb_scale = float(id_emb_scale)
        self.guidance_scale = float(guidance_scale)
        self.num_inversion_steps = int(num_inversion_steps)
        self.face_image_size = int(face_image_size)
        self.det_thresh = float(det_thresh)
        self.ip_adapter_scale = float(ip_adapter_scale)
        self.det_size = int(det_size)
        self.seed = int(seed)
        self.enable_face_detection = bool(enable_face_detection)
        self.use_model_cpu_offload = self._resolve_cpu_offload_setting(use_model_cpu_offload)
        self._apply_local_hardware_defaults()
        self.reason = self._availability_reason()

    def _resolve_cpu_offload_setting(self, requested: bool | None) -> bool:
        if requested is not None:
            return bool(requested)
        total_gb, _available_gb = read_accelerator_memory_gb()
        return 0.0 < total_gb < LOW_VRAM_THRESHOLD_GB

    def _apply_local_hardware_defaults(self) -> None:
        if not self.use_model_cpu_offload:
            return
        self.num_inversion_steps = min(self.num_inversion_steps, 5)
        self.face_image_size = min(self.face_image_size, 512)
        self.det_size = min(self.det_size, 320)

    def _availability_reason(self) -> str:
        if not self.backend_root.exists():
            return f"Reverse Personalization backend directory not found at {self.backend_root}"
        if not self.runner_path.is_file():
            return f"Reverse Personalization runner missing at {self.runner_path}"

        probe = "\n".join(
            [
                "import importlib",
                "import sys",
                f"mods = {REQUIRED_IMPORTS!r}",
                "missing = []",
                "for name in mods:",
                "    try:",
                "        importlib.import_module(name)",
                "    except Exception:",
                "        missing.append(name)",
                "if missing:",
                "    sys.exit(','.join(missing))",
            ]
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = self._build_pythonpath(env.get("PYTHONPATH"))
        try:
            result = subprocess.run(
                [self.python_executable, "-c", probe],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        except FileNotFoundError:
            return f"Python executable not found: {self.python_executable}"
        if result.returncode != 0:
            missing = (result.stdout or result.stderr).strip() or "unknown dependencies"
            return f"Reverse Personalization preflight failed (rc={result.returncode}): {missing}"
        return ""

    def _build_pythonpath(self, existing: str | None) -> str:
        paths = [str(self.backend_root)]
        if existing:
            paths.append(existing)
        return os.pathsep.join(paths)

    def _build_command(self, input_path: Path, output_path: Path) -> list[str]:
        command = [
            self.python_executable,
            str(self.runner_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--sd-model-path",
            self.sd_model_path,
            "--insightface-model-path",
            self.insightface_model_path,
            "--device-num",
            str(self.device_num),
            "--skip",
            str(self.skip),
            "--id-emb-scale",
            str(self.id_emb_scale),
            "--guidance-scale",
            str(self.guidance_scale),
            "--num-inversion-steps",
            str(self.num_inversion_steps),
            "--face-image-size",
            str(self.face_image_size),
            "--det-thresh",
            str(self.det_thresh),
            "--ip-adapter-scale",
            str(self.ip_adapter_scale),
            "--det-size",
            str(self.det_size),
            "--seed",
            str(self.seed),
        ]
        if self.attribute_prompt:
            command.extend(["--attribute-prompt", self.attribute_prompt])
        if self.use_model_cpu_offload:
            command.append("--use-model-cpu-offload")
        if self.enable_face_detection:
            command.append("--enable-face-detection")
        return command

    def _run_backend_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = self._build_pythonpath(env.get("PYTHONPATH"))
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def anonymise(self, image: Image.Image, boxes: list[BoundingBox]) -> AnonymiserResult:
        output = image.copy().convert("RGB")
        valid_boxes = self.validate_boxes(output, boxes)
        if not valid_boxes:
            return AnonymiserResult(
                image=output,
                metadata={
                    "method": self.method_name,
                    "boxes_processed": 0,
                    "backend_uses_internal_detection": self.enable_face_detection,
                    "backend_ready": not bool(self.reason),
                },
            )
        if self.reason:
            raise NotImplementedError(f"{self.method_name} unavailable: {self.reason}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            input_path = temp_dir / "input.png"
            output_path = temp_dir / "output.png"
            output.save(input_path)
            result = self._run_backend_command(self._build_command(input_path, output_path))
            if result.returncode != 0:
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                detail = stderr or stdout or "Reverse Personalization subprocess failed without output"
                raise RuntimeError(f"Reverse Personalization subprocess failed: {detail}")
            if not output_path.is_file():
                raise RuntimeError("Reverse Personalization did not produce an output image")
            anonymised = Image.open(output_path).convert("RGB")
            anonymised.load()

        return AnonymiserResult(
            image=anonymised,
            metadata={
                "method": self.method_name,
                "boxes_processed": len(valid_boxes),
                "backend_uses_internal_detection": self.enable_face_detection,
                "backend_root": str(self.backend_root),
                "runner_path": str(self.runner_path),
                "python_executable": self.python_executable,
                "attribute_prompt": self.attribute_prompt,
                "sd_model_path": self.sd_model_path,
                "device_num": self.device_num,
                "num_inversion_steps": self.num_inversion_steps,
                "face_image_size": self.face_image_size,
                "det_size": self.det_size,
                "use_model_cpu_offload": self.use_model_cpu_offload,
            },
        )
