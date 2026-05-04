"""unittest 入口：与 `python -m evaluation.run_benchmark` 共用同一套指标函数。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bench_asr_tags import run_asr_tag_benchmark
from evaluation.bench_face import run_face_benchmark
from evaluation.bench_fusion import run_fusion_benchmark
from evaluation.bench_physio import run_physio_benchmark
from evaluation.bench_vad import run_vad_micro_benchmark


class TestTriModalMetrics(unittest.TestCase):
    def test_physio_golden_and_noise(self):
        r = run_physio_benchmark(golden_count=100, random_trials=3, random_bytes_each=8000, seed=123)
        self.assertTrue(r["golden_stream_decode_exact_match"])
        self.assertTrue(r["sticky_packet_full_recovery"])
        self.assertEqual(r["random_noise_total_spurious_frames"], 0)

    def test_fusion_rules(self):
        r = run_fusion_benchmark()
        self.assertEqual(r["rule_case_pass_rate"], 1.0)

    def test_asr_tags(self):
        r = run_asr_tag_benchmark()
        self.assertEqual(r["pass_rate"], 1.0)

    def test_vad_gate(self):
        r = run_vad_micro_benchmark()
        self.assertTrue(r["pass_low_silent"])
        self.assertTrue(r["pass_high_active"])

    def test_face_determinism_if_weights_present(self):
        r = run_face_benchmark(stability_runs=20, timing_runs=5)
        if r.get("skipped"):
            self.skipTest(r.get("skipped_reason", "skipped"))
        self.assertEqual(r["unique_argmax_count"], 1)
        self.assertGreaterEqual(r["label_consistency_ratio"], 0.999)


if __name__ == "__main__":
    unittest.main()
