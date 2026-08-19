import argparse
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import metrics


def write_csv(path: Path, rows):
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class MetricsScriptTest(unittest.TestCase):
    def test_normalize_run_label_accepts_e7_without_metadata(self):
        self.assertEqual(metrics.normalize_run_label(Path("e6"), {}), "E6")
        self.assertEqual(metrics.normalize_run_label(Path("e7"), {}), "E7")

    def test_build_tables_from_minimal_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            run = root / "e5"
            run.mkdir(parents=True)

            (run / "experiment.json").write_text(
                json.dumps(
                    {
                        "method": "dpga",
                        "experiment_stage": "E5",
                        "world_size": 2,
                        "warmup_enabled": True,
                        "filtering_enabled": True,
                        "projection_enabled": True,
                        "norm_cap_enabled": True,
                        "gate_enabled": True,
                    }
                ),
                encoding="utf-8",
            )
            write_csv(
                run / "metrics.csv",
                [
                    {
                        "epoch": 0,
                        "method": "dpga",
                        "seconds": 10.0,
                        "AP": 0.10,
                        "AP50": 0.20,
                        "MR-2_generic": 0.80,
                        "ODAM_quality": 0.30,
                        "ODAM_quality_mean_iou": 0.60,
                        "loss_det": 2.0,
                        "loss_odam": 0.5,
                        "raw_loss_sum": 2.1,
                        "loss_proxy": 2.0,
                        "odam_num_candidates": 100,
                        "odam_num_kept": 20,
                        "odam_keep_ratio": 0.2,
                    },
                    {
                        "epoch": 1,
                        "method": "dpga",
                        "seconds": 12.0,
                        "AP": 0.15,
                        "AP50": 0.25,
                        "MR-2_generic": 0.75,
                        "ODAM_quality": 0.35,
                        "ODAM_quality_mean_iou": 0.62,
                        "loss_det": 1.5,
                        "loss_odam": 0.4,
                        "raw_loss_sum": 1.6,
                        "loss_proxy": 1.5,
                        "odam_num_candidates": 100,
                        "odam_num_kept": 10,
                        "odam_keep_ratio": 0.1,
                    },
                ],
            )
            write_csv(
                run / "gradient_diagnostics_rank0.csv",
                [
                    {
                        "epoch": 0,
                        "step": 0,
                        "rank": 0,
                        "gradient_scope": "global_ddp_mean",
                        "loss_odam": 0.5,
                        "cosine_raw": -0.2,
                        "det_gradient_norm": 2.0,
                        "raw_odam_norm": 1.0,
                        "final_odam_norm": 0.2,
                        "aux_to_det_raw": 0.5,
                        "conflict_raw": 1,
                        "projected": 1,
                        "cap_active": 1,
                        "unsafe_descent": 0,
                        "gate": 0.5,
                        "alpha": 0.2,
                        "effective_weight": 0.1,
                    },
                    {
                        "epoch": 0,
                        "step": 1,
                        "rank": 0,
                        "gradient_scope": "global_ddp_mean",
                        "loss_odam": 0.0,
                        "cosine_raw": 0.0,
                        "det_gradient_norm": 2.0,
                        "raw_odam_norm": 0.0,
                        "final_odam_norm": 0.0,
                        "aux_to_det_raw": 0.0,
                        "conflict_raw": 0,
                        "projected": 0,
                        "cap_active": 0,
                        "unsafe_descent": 0,
                        "gate": 0.0,
                        "alpha": 0.2,
                        "effective_weight": 0.0,
                    },
                ],
            )
            (run / "odam_quality_epoch_001.json").write_text(
                json.dumps(
                    [
                        {"dam_energy_in_gt": 0.7, "iou": 0.6},
                        {"dam_energy_in_gt": 0.9, "iou": 0.8},
                    ]
                ),
                encoding="utf-8",
            )
            (run / "xai_quality_best.json").write_text(
                json.dumps(
                    [
                        {
                            "dam_energy_in_gt": 0.6,
                            "iou": 0.7,
                            "pointing_game_hit": 1,
                            "saliency_iou": 0.2,
                        },
                        {
                            "dam_energy_in_gt": 0.8,
                            "iou": 0.9,
                            "pointing_game_hit": 0,
                            "saliency_iou": 0.4,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            tables = metrics.build_tables(
                argparse.Namespace(
                    outputs=root,
                    output_dir=Path(tmp) / "report",
                    runs=None,
                    citypersons_mr_csv=None,
                )
            )

            detection = tables["detection"][0]
            gradient = tables["gradient"][0]
            xai = tables["xai"][0]
            cost = tables["cost"][0]

            self.assertEqual(detection["run"], "E5")
            self.assertAlmostEqual(detection["best_AP"], 0.15)
            self.assertAlmostEqual(detection["final_AP50"], 0.25)
            self.assertAlmostEqual(detection["best_MR-2_generic"], 0.75)
            self.assertTrue(math.isnan(detection["MR-2_Reasonable"]))
            self.assertEqual(detection["citypersons_mr_source"], "not_available")

            self.assertEqual(gradient["gradient_active_steps"], 1)
            self.assertAlmostEqual(gradient["gradient_conflict_rate"], 1.0)
            self.assertAlmostEqual(gradient["projection_rate"], 1.0)
            self.assertAlmostEqual(gradient["norm_cap_rate"], 1.0)
            self.assertTrue(math.isnan(gradient["final_gradient_norm"]))

            self.assertEqual(xai["xai_source"], "export_xai_metrics")
            self.assertAlmostEqual(xai["bbox_energy_ratio_mean"], 0.7)
            self.assertAlmostEqual(xai["detection_match_iou_mean"], 0.8)
            self.assertAlmostEqual(xai["pointing_game"], 0.5)
            self.assertAlmostEqual(xai["saliency_iou"], 0.3)
            self.assertAlmostEqual(cost["time_per_epoch_mean_s"], 11.0)

    def test_offline_detection_metrics_use_saved_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            run = root / "baseline"
            run.mkdir(parents=True)
            ann_path = Path(tmp) / "valid.json"

            ann_path.write_text(
                json.dumps(
                    {
                        "info": {},
                        "licenses": [],
                        "images": [
                            {"id": 1, "file_name": "a.png", "width": 100, "height": 100},
                            {"id": 2, "file_name": "b.png", "width": 100, "height": 100},
                        ],
                        "categories": [
                            {"id": 1, "name": "person", "supercategory": "person"}
                        ],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [10, 10, 20, 60],
                                "area": 1200,
                                "iscrowd": 0,
                                "ignore": 0,
                                "height": 60,
                                "vis_ratio": 0.8,
                                "segmentation": [],
                            },
                            {
                                "id": 2,
                                "image_id": 2,
                                "category_id": 1,
                                "bbox": [40, 10, 20, 60],
                                "area": 1200,
                                "iscrowd": 1,
                                "ignore": 1,
                                "height": 60,
                                "vis_ratio": 1.0,
                                "segmentation": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            write_csv(
                run / "metrics.csv",
                [
                    {
                        "epoch": 0,
                        "method": "baseline",
                        "seconds": 1.0,
                        "AP": 0.0,
                        "AP50": 0.0,
                        "MR-2_generic": 1.0,
                    },
                    {
                        "epoch": 1,
                        "method": "baseline",
                        "seconds": 1.0,
                        "AP": 1.0,
                        "AP50": 1.0,
                        "MR-2_generic": 0.0,
                    },
                ],
            )
            (run / "predictions_epoch_001.json").write_text(
                json.dumps(
                    [
                        {
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [10, 10, 20, 60],
                            "score": 0.9,
                        },
                        {
                            "image_id": 2,
                            "category_id": 1,
                            "bbox": [40, 10, 20, 60],
                            "score": 0.8,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            tables = metrics.build_tables(
                argparse.Namespace(
                    outputs=root,
                    output_dir=Path(tmp) / "report",
                    runs=None,
                    citypersons_mr_csv=None,
                    val_ann=ann_path,
                    offline_prediction_epoch="both",
                    offline_best_metric="AP",
                    offline_iou_threshold=0.5,
                    citypersons_ignore_ioa=0.5,
                )
            )

            detection = tables["detection"][0]
            self.assertEqual(detection["offline_detection_source"], "saved_predictions")
            self.assertEqual(detection["offline_best_epoch"], 1)
            self.assertEqual(detection["offline_final_epoch"], 1)
            self.assertGreater(detection["offline_best_AP50"], 0.99)
            self.assertLess(detection["offline_best_MR-2_Reasonable"], 1e-6)
            self.assertLess(detection["offline_best_MR-2_Small"], 1e-6)
            self.assertTrue(math.isnan(detection["offline_best_MR-2_Heavy"]))

    def test_root_xai_summary_populates_xai_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            run = root / "e3"
            run.mkdir(parents=True)

            (run / "experiment.json").write_text(
                json.dumps({"method": "dpga", "experiment_stage": "E3"}),
                encoding="utf-8",
            )
            write_csv(
                run / "metrics.csv",
                [
                    {
                        "epoch": 0,
                        "method": "dpga",
                        "seconds": 1.0,
                        "AP": 0.1,
                        "AP50": 0.2,
                        "MR-2_generic": 0.8,
                    }
                ],
            )
            (root / "xai_quality_best_summary.json").write_text(
                json.dumps(
                    [
                        {
                            "run_dir": "e3",
                            "samples": 12,
                            "bbox_energy_ratio": 0.75,
                            "pointing_game": 0.5,
                            "saliency_iou": 0.25,
                            "detection_match_iou": 0.6,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            tables = metrics.build_tables(
                argparse.Namespace(
                    outputs=root,
                    output_dir=Path(tmp) / "report",
                    runs=None,
                    citypersons_mr_csv=None,
                )
            )

            xai = tables["xai"][0]
            self.assertEqual(xai["xai_source"], "export_xai_metrics_summary")
            self.assertEqual(xai["bbox_energy_samples"], 12)
            self.assertAlmostEqual(xai["bbox_energy_ratio_mean"], 0.75)
            self.assertAlmostEqual(xai["pointing_game"], 0.5)
            self.assertAlmostEqual(xai["saliency_iou"], 0.25)
            self.assertAlmostEqual(xai["detection_match_iou_mean"], 0.6)


if __name__ == "__main__":
    unittest.main()
