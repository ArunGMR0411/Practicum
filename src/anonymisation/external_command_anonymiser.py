"""Shared adapter for anonymisers backed by external research code."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser, BoundingBox


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


class ExternalCommandAnonymiser(BaseAnonymiser):
    """Run an anonymiser through a stable image-in/image-out command contract."""

    method_name = "external"

    def __init__(
        self,
        *,
        backend_root: Path,
        runner_path: Path,
        model_path: Path | None = None,
        model_id: str | None = None,
        python_executable: str = str(DEFAULT_PROJECT_PYTHON if DEFAULT_PROJECT_PYTHON.is_file() else "python3"),
        extra_args: list[str] | None = None,
        required_imports: tuple[str, ...] = (),
        required_paths: tuple[Path, ...] = (),
    ) -> None:
        self.backend_root = Path(backend_root)
        self.runner_path = Path(runner_path)
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_id = model_id
        self.python_executable = python_executable
        self.extra_args = list(extra_args or [])
        self.required_imports = required_imports
        self.required_paths = tuple(Path(path) for path in required_paths)
        self.reason = self._availability_reason()

    def _availability_reason(self) -> str:
        if not self.backend_root.exists():
            return f"{self.method_name} backend directory not found at {self.backend_root}"
        if not self.runner_path.is_file():
            return f"{self.method_name} runner missing at {self.runner_path}"
        if self.model_path is not None and not self.model_path.exists():
            return f"{self.method_name} model/checkpoint missing at {self.model_path}"
        if self.model_path is None and not self.model_id:
            return f"{self.method_name} requires either a model_path or a model_id"
        missing_paths = [str(path) for path in self.required_paths if not path.exists()]
        if missing_paths:
            return f"{self.method_name} required assets missing: {missing_paths}"
        if self.required_imports:
            return self._dependency_reason()
        return ""

    def _dependency_reason(self) -> str:
        probe = "\n".join(
            [
                "import importlib",
                "import sys",
                "import warnings",
                "warnings.filterwarnings('ignore')",
                f"mods = {self.required_imports!r}",
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
        try:
            result = subprocess.run(
                [self.python_executable, "-c", probe],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=self._build_env(),
            )
        except FileNotFoundError:
            return f"Python executable not found: {self.python_executable}"
        if result.returncode != 0:
            missing = (result.stdout or result.stderr).strip() or "unknown dependencies"
            return f"{self.method_name} dependency preflight failed (rc={result.returncode}): {missing}"
        return ""

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        paths = [str(self.backend_root)]
        if existing:
            paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    def _build_command(self, input_path: Path, output_path: Path, boxes_path: Path) -> list[str]:
        command = [
            self.python_executable,
            str(self.runner_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--boxes-json",
            str(boxes_path),
        ]
        if self.model_path is not None:
            command.extend(["--model-path", str(self.model_path)])
        if self.model_id:
            command.extend(["--model-id", self.model_id])
        command.extend(self.extra_args)
        return command

    def _run_backend_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=self._build_env(),
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
                    "backend_ready": not bool(self.reason),
                },
            )
        if self.reason:
            raise NotImplementedError(f"{self.method_name} unavailable: {self.reason}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            input_path = temp_dir / "input.png"
            output_path = temp_dir / "output.png"
            boxes_path = temp_dir / "boxes.json"
            output.save(input_path)
            boxes_path.write_text(json.dumps({"boxes": valid_boxes}), encoding="utf-8")
            result = self._run_backend_command(self._build_command(input_path, output_path, boxes_path))
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or f"{self.method_name} subprocess failed"
                raise RuntimeError(f"{self.method_name} subprocess failed: {detail}")
            if not output_path.is_file():
                raise RuntimeError(f"{self.method_name} did not produce an output image")
            anonymised = Image.open(output_path).convert("RGB")
            anonymised.load()

        return AnonymiserResult(
            image=anonymised,
            metadata={
                "method": self.method_name,
                "boxes_processed": len(valid_boxes),
                "backend_root": str(self.backend_root),
                "runner_path": str(self.runner_path),
                "model_path": str(self.model_path) if self.model_path is not None else None,
                "model_id": self.model_id,
                "python_executable": self.python_executable,
            },
        )
