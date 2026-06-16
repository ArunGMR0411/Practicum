import unittest
import numpy as np
from PIL import Image
import os

from src.evaluation.reid_evaluator import ReIDEvaluator

class TestReIDEvaluation(unittest.TestCase):
    def setUp(self):
        # We can construct the evaluator using paths to actual checkpoints
        self.adaface_ckpt = "data/models/adaface_ir50_ms1mv2.ckpt"
        self.arcface_onnx = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
        
        # Check if the models exist before trying to load, else skip those tests
        self.models_exist = os.path.exists(self.adaface_ckpt) and os.path.exists(self.arcface_onnx)
        
        if self.models_exist:
            self.evaluator = ReIDEvaluator(self.adaface_ckpt, self.arcface_onnx, device='cpu')

    def test_compute_reid_metrics_perfect_match(self):
        # Create a dummy ReIDEvaluator instance (we can just pass dummy strings if we don't load the models,
        # but let's test compute_reid_metrics directly without requiring models to exist)
        evaluator_stub = object.__new__(ReIDEvaluator)
        
        # Matched identity embeddings (perfect correlation)
        gallery = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        query = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        metrics = evaluator_stub.compute_reid_metrics(gallery, query)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)
        self.assertAlmostEqual(metrics["reid_rate"], 1.0)

    def test_compute_reid_metrics_random_match(self):
        evaluator_stub = object.__new__(ReIDEvaluator)
        
        gallery = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Query where first and second are swapped, third matches
        query = np.array([
            [0.0, 1.0, 0.0], # swaps with gallery index 1
            [1.0, 0.0, 0.0], # swaps with gallery index 0
            [0.0, 0.0, 1.0]  # matches gallery index 2
        ], dtype=np.float32)
        
        metrics = evaluator_stub.compute_reid_metrics(gallery, query)
        # Cosine similarity for matched pairs: diag of dot(query, gallery.T)
        # Dot product of row 0: query[0] dot gallery[0] = 0.0
        # Dot product of row 1: query[1] dot gallery[1] = 0.0
        # Dot product of row 2: query[2] dot gallery[2] = 1.0
        # Mean cosine similarity = (0 + 0 + 1) / 3 = 0.33333
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0/3.0, places=5)
        # Re-ID rate: argmax of sim_matrix:
        # Row 0 sim: [0, 1, 0] -> argmax = 1 (wrong)
        # Row 1 sim: [1, 0, 0] -> argmax = 0 (wrong)
        # Row 2 sim: [0, 0, 1] -> argmax = 2 (correct)
        # Re-ID rate = 1/3
        self.assertAlmostEqual(metrics["reid_rate"], 1.0/3.0, places=5)

    def test_model_inference(self):
        if not self.models_exist:
            self.skipTest("AdaFace or ArcFace model files not found. Skipping model inference tests.")
        
        # Create a few dummy images
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        crops = [img, img]
        
        # Extract AdaFace
        ada_feats = self.evaluator.extract_embeddings_adaface(crops)
        self.assertEqual(ada_feats.shape, (2, 512))
        # Verify L2 normalization
        norms = np.linalg.norm(ada_feats, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
        
        # Extract ArcFace
        arc_feats = self.evaluator.extract_embeddings_arcface(crops)
        self.assertEqual(arc_feats.shape, (2, 512))
        # Verify L2 normalization
        norms = np.linalg.norm(arc_feats, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
