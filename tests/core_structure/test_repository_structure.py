import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


REQUIRED_DIRECTORIES = [
    "src",
    "src/data",
    "src/detection",
    "src/anonymisation",
    "src/routing",
    "src/evaluation",
    "src/utils",
    "app",
    "app/src/privacy_pipeline_app",
    "app/models",
    "app/inputs",
    "data",
    "data/castle2024",
    "data/castle2024/raw",
    "data/models",
    "configs",
    "scripts",
    "outputs",
    "tests",
    "docs",
    "third_party",
]

REQUIRED_APP_MODULES = [
    "app/src/privacy_pipeline_app/wizard_workflow.py",
    "app/src/privacy_pipeline_app/production_app.py",
    "app/src/privacy_pipeline_app/objective_policy.py",
    "app/src/privacy_pipeline_app/method_catalog.py",
    "app/src/privacy_pipeline_app/runtime_policy.py",
    "app/src/privacy_pipeline_app/thesis_face_detector.py",
    "app/src/privacy_pipeline_app/detection_reviewer.py",
    "app/run_web.py",
    "app/run_cli.py",
]

REQUIRED_EVIDENCE_FILES = [
    "outputs/09_traceability/01_evidence_index.csv",
    "outputs/03_anonymisation/01_all_methods_comparison.csv",
    "outputs/05_oapr/12_oapr_full_metric_summary.csv",
    "outputs/04_multimodal_privacy/01_multimodal_250_evidence/11_rq3_final_summary.md",
    "src/detection/multimodal_precision_stack.py",
]


class TestRepositoryStructure(unittest.TestCase):
    def test_required_directories_exist(self) -> None:
        missing = [directory for directory in REQUIRED_DIRECTORIES if not (PROJECT_ROOT / directory).is_dir()]
        self.assertFalse(missing, f"Missing required directories: {missing}")

    def test_required_directories_are_unique(self) -> None:
        self.assertEqual(len(REQUIRED_DIRECTORIES), len(set(REQUIRED_DIRECTORIES)))

    def test_core_project_files_exist(self) -> None:
        for path in ("config.yaml", "requirements.txt", "app/requirements.txt"):
            with self.subTest(path=path):
                self.assertTrue((PROJECT_ROOT / path).exists(), f"Missing expected file: {path}")

    def test_app_modules_exist(self) -> None:
        missing = [path for path in REQUIRED_APP_MODULES if not (PROJECT_ROOT / path).is_file()]
        self.assertFalse(missing, f"Missing app modules: {missing}")

    def test_required_evidence_files_exist(self) -> None:
        missing = [path for path in REQUIRED_EVIDENCE_FILES if not (PROJECT_ROOT / path).exists()]
        self.assertFalse(missing, f"Missing evidence/source files: {missing}")


if __name__ == "__main__":
    unittest.main()
