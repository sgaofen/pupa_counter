from __future__ import annotations

import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.cellpose_postprocess import build_annotated_png_supplement


def test_build_annotated_png_supplement_adds_unmatched_strong_classical_candidates():
    cfg = AppConfig()
    cfg.detector.cellpose_annotated_png_supplement_max_unmatched_ratio = 1.0

    cellpose_df = pd.DataFrame(
        [
            {
                "component_id": "cp_1",
                "bbox_x0": 10,
                "bbox_y0": 10,
                "bbox_x1": 30,
                "bbox_y1": 30,
                "area_px": 120.0,
                "mean_v": 120.0,
                "color_score": 0.40,
                "local_contrast": 12.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "is_active": True,
                "label": "pupa",
                "confidence": 0.90,
                "cluster_area_threshold": 0.0,
            }
        ]
    )

    classical_df = pd.DataFrame(
        [
            {
                "component_id": "cc_1",
                "bbox_x0": 12,
                "bbox_y0": 12,
                "bbox_x1": 29,
                "bbox_y1": 29,
                "area_px": 118.0,
                "mean_v": 118.0,
                "color_score": 0.41,
                "local_contrast": 13.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "is_active": True,
                "label": "pupa",
                "confidence": 0.90,
                "cluster_area_threshold": 0.0,
            },
            {
                "component_id": "cc_2",
                "bbox_x0": 80,
                "bbox_y0": 50,
                "bbox_x1": 96,
                "bbox_y1": 72,
                "area_px": 92.0,
                "mean_v": 110.0,
                "color_score": 0.33,
                "local_contrast": 10.0,
                "blue_overlap_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "is_active": True,
                "label": "pupa",
                "confidence": 0.72,
                "cluster_area_threshold": 0.0,
            },
        ]
    )

    supplement = build_annotated_png_supplement(
        cellpose_df,
        classical_df,
        source_type="annotated_png",
        cfg=cfg,
    )

    assert supplement["component_id"].tolist() == ["cc_2"]
    assert supplement["detector_source"].tolist() == ["annotated_classical_addon"]
