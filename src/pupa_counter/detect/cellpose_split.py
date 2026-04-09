"""Conservative post-split for large Cellpose masks.

Cellpose handles most single pupae well, but on high-quality annotated sheets
it still occasionally returns one fat mask for two touching pupae. We only
touch a very small subset of detections:

- clearly larger than the image-level median pupa area
- too round / too low-eccentricity to look like a single pupa
- showing multiple distance-map peaks inside the mask

If any of those conditions fail, the original Cellpose mask is kept as-is.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, measure, segmentation

from pupa_counter.config import AppConfig
from pupa_counter.detect.components import build_component_row


def split_large_cellpose_instances(
    components_df: pd.DataFrame,
    image_shape,
    *,
    source_type: str,
    cfg: AppConfig,
    guide_image: np.ndarray | None = None,
    restrict_to_dense_patch: bool = False,
) -> pd.DataFrame:
    if components_df.empty or not cfg.detector.cellpose_overlap_split_enabled:
        return components_df.copy()
    if source_type not in {"annotated_png", "clean_png"}:
        return components_df.copy()

    median_area = float(components_df["area_px"].median()) if "area_px" in components_df.columns else 0.0
    if median_area <= 0:
        return components_df.copy()

    rows = []
    child_rows = []
    min_child_area_px = max(
        float(cfg.components.min_area_px),
        median_area * cfg.detector.cellpose_overlap_split_min_child_area_ratio,
    )

    def _normalized(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        masked = values[mask]
        if masked.size == 0:
            return np.zeros_like(values, dtype=np.float32)
        low = float(masked.min())
        high = float(masked.max())
        if high <= low:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values.astype(np.float32) - low) / (high - low), 0.0, 1.0)

    def _build_combo_map(local_mask: np.ndarray, row_dict: dict, *, brown_weight: float) -> np.ndarray | None:
        if guide_image is None:
            return None
        y0 = int(row_dict["bbox_y0"])
        x0 = int(row_dict["bbox_x0"])
        y1 = int(row_dict["bbox_y1"])
        x1 = int(row_dict["bbox_x1"])
        if y1 <= y0 or x1 <= x0:
            return None
        local_patch = guide_image[y0:y1, x0:x1]
        if local_patch.shape[:2] != local_mask.shape:
            return None
        rgb = local_patch.astype(np.float32)
        brownness = rgb[:, :, 0] - 0.6 * rgb[:, :, 2] - 0.4 * rgb[:, :, 1]
        brownness = _normalized(brownness, local_mask)
        distance = ndi.distance_transform_edt(local_mask)
        distance = _normalized(distance, local_mask)
        combo = ((1.0 - brown_weight) * distance + brown_weight * brownness) * local_mask.astype(np.float32)
        return combo

    def _row_aspect_ratio(row_dict: dict) -> float:
        ratio = row_dict.get("aspect_ratio")
        if ratio is not None and float(ratio) > 0.0:
            return float(ratio)
        major_axis = float(row_dict.get("major_axis_px", 0.0) or 0.0)
        minor_axis = max(float(row_dict.get("minor_axis_px", 0.0) or 0.0), 1.0)
        if major_axis <= 0.0:
            return 0.0
        return major_axis / minor_axis

    def _combo_regions(local_mask: np.ndarray, row_dict: dict):
        if (
            source_type != "annotated_png"
            or guide_image is None
            or not cfg.detector.cellpose_overlap_split_combo_enabled
        ):
            return []

        area_px = float(row_dict.get("area_px", 0.0) or 0.0)
        if area_px < median_area * float(cfg.detector.cellpose_overlap_split_combo_area_ratio):
            return []
        if bool(row_dict.get("touches_image_border", False)):
            return []
        if float(row_dict.get("border_touch_ratio", 0.0) or 0.0) > float(
            cfg.detector.cellpose_overlap_split_combo_max_border_touch_ratio
        ):
            return []

        combo = _build_combo_map(
            local_mask,
            row_dict,
            brown_weight=float(cfg.detector.cellpose_overlap_split_combo_brown_weight),
        )
        if combo is None:
            return []

        peaks = feature.peak_local_max(
            combo,
            min_distance=cfg.detector.cellpose_overlap_split_combo_peak_min_distance,
            threshold_abs=cfg.detector.cellpose_overlap_split_combo_peak_abs_threshold,
            labels=local_mask.astype(np.uint8),
        )
        combo_peak_cap = max(cfg.detector.cellpose_overlap_split_max_children * 3, 8)
        if peaks.shape[0] < 2:
            return []
        if peaks.shape[0] > combo_peak_cap:
            return []
        if peaks.shape[0] > cfg.detector.cellpose_overlap_split_max_children:
            peaks = peaks[:combo_peak_cap]

        markers = np.zeros(local_mask.shape, dtype=np.int32)
        for marker_index, (peak_row, peak_col) in enumerate(peaks, start=1):
            markers[peak_row, peak_col] = marker_index
        markers, _ = ndi.label(markers > 0)
        labels = segmentation.watershed(-combo, markers, mask=local_mask)
        combo_min_child_area_px = max(
            float(cfg.components.min_area_px),
            area_px * float(cfg.detector.cellpose_overlap_split_combo_min_child_area_ratio),
        )
        regions = [
            region
            for region in measure.regionprops(labels)
            if float(region.area) >= combo_min_child_area_px
        ]
        if len(regions) < 2 or len(regions) > cfg.detector.cellpose_overlap_split_max_children:
            return []
        child_areas = [float(region.area) for region in regions]
        if min(child_areas) <= 0.0:
            return []
        if max(child_areas) / min(child_areas) > float(
            cfg.detector.cellpose_overlap_split_combo_max_child_area_ratio
        ):
            return []
        return regions

    def _pairlike_split(local_mask: np.ndarray, row_dict: dict):
        if (
            source_type != "annotated_png"
            or guide_image is None
            or not cfg.detector.cellpose_overlap_split_pairlike_enabled
        ):
            return None, []
        if bool(row_dict.get("touches_image_border", False)):
            return None, []
        if float(row_dict.get("border_touch_ratio", 0.0) or 0.0) > float(
            cfg.detector.cellpose_overlap_split_combo_max_border_touch_ratio
        ):
            return None, []

        area_px = float(row_dict.get("area_px", 0.0) or 0.0)
        area_ratio = area_px / max(median_area, 1.0)
        if area_ratio < float(cfg.detector.cellpose_overlap_split_pairlike_min_area_ratio):
            return None, []
        if area_ratio > float(cfg.detector.cellpose_overlap_split_pairlike_max_area_ratio):
            return None, []

        aspect_ratio = _row_aspect_ratio(row_dict)
        eccentricity = float(row_dict.get("eccentricity", 0.0) or 0.0)
        extent = float(row_dict.get("extent", 1.0) or 1.0)
        if aspect_ratio < float(cfg.detector.cellpose_overlap_split_pairlike_min_aspect_ratio):
            return None, []
        if aspect_ratio > float(cfg.detector.cellpose_overlap_split_pairlike_max_aspect_ratio):
            return None, []
        if eccentricity < float(cfg.detector.cellpose_overlap_split_pairlike_min_eccentricity):
            return None, []
        if extent > float(cfg.detector.cellpose_overlap_split_pairlike_max_extent):
            return None, []

        combo = _build_combo_map(
            local_mask,
            row_dict,
            brown_weight=float(cfg.detector.cellpose_overlap_split_pairlike_brown_weight),
        )
        if combo is None:
            return None, []
        sigma = float(cfg.detector.cellpose_overlap_split_pairlike_gaussian_sigma)
        if sigma > 0.0:
            combo = ndi.gaussian_filter(combo, sigma=sigma) * local_mask.astype(np.float32)

        peaks = feature.peak_local_max(
            combo,
            min_distance=cfg.detector.cellpose_overlap_split_pairlike_peak_min_distance,
            threshold_abs=cfg.detector.cellpose_overlap_split_pairlike_peak_abs_threshold,
            labels=local_mask.astype(np.uint8),
        )
        if peaks.shape[0] < 2:
            return None, []
        if peaks.shape[0] > 4:
            peaks = peaks[:4]
        candidate_pairs = [np.asarray(pair, dtype=np.int32) for pair in itertools.combinations(peaks.tolist(), 2)]

        min_peak_distance = max(
            4.0,
            float(row_dict.get("minor_axis_px", 0.0) or 0.0)
            * float(cfg.detector.cellpose_overlap_split_pairlike_min_peak_distance_scale),
        )
        rr, cc = np.nonzero(local_mask)
        best_score = None
        best_labels = None
        best_regions = []
        for pair_arr in candidate_pairs:
            peak_distance = float(np.linalg.norm(pair_arr[0] - pair_arr[1]))
            if peak_distance < min_peak_distance:
                continue

            distances = []
            for peak_row, peak_col in pair_arr:
                distances.append((rr - int(peak_row)) ** 2 + (cc - int(peak_col)) ** 2)
            distances = np.stack(distances, axis=1)
            assignment = np.argmin(distances, axis=1) + 1

            candidate_labels = np.zeros(local_mask.shape, dtype=np.int32)
            candidate_labels[rr, cc] = assignment
            candidate_regions = [
                region
                for region in measure.regionprops(candidate_labels)
                if float(region.area) >= max(
                    float(cfg.components.min_area_px),
                    area_px * float(cfg.detector.cellpose_overlap_split_pairlike_min_child_area_ratio),
                )
            ]
            if len(candidate_regions) != 2:
                continue
            child_areas = [float(region.area) for region in candidate_regions]
            if min(child_areas) <= 0.0:
                continue
            balance = max(child_areas) / min(child_areas)
            if balance > float(cfg.detector.cellpose_overlap_split_pairlike_max_child_area_ratio):
                continue

            peak_strength = float(min(combo[int(pr), int(pc)] for pr, pc in pair_arr))
            score = (peak_strength, -balance, peak_distance)
            if best_score is None or score > best_score:
                best_score = score
                best_labels = candidate_labels
                best_regions = candidate_regions

        if best_labels is None:
            return None, []
        labels = best_labels
        regions = best_regions

        pair_min_child_area_px = max(
            float(cfg.components.min_area_px),
            area_px * float(cfg.detector.cellpose_overlap_split_pairlike_min_child_area_ratio),
        )
        regions = [
            region
            for region in measure.regionprops(labels)
            if float(region.area) >= pair_min_child_area_px
        ]
        if len(regions) != 2:
            return None, []
        child_areas = [float(region.area) for region in regions]
        if min(child_areas) <= 0.0:
            return None, []
        if max(child_areas) / min(child_areas) > float(
            cfg.detector.cellpose_overlap_split_pairlike_max_child_area_ratio
        ):
            return None, []
        return labels, regions

    for _, row in components_df.iterrows():
        row_dict = row.to_dict()
        if restrict_to_dense_patch and not bool(row_dict.get("dense_patch_refined", False)):
            rows.append(row_dict)
            continue
        area_px = float(row_dict.get("area_px", 0.0) or 0.0)
        aspect_ratio = _row_aspect_ratio(row_dict)
        eccentricity = float(row_dict.get("eccentricity", 1.0) or 1.0)

        split_area_ratio = float(cfg.detector.cellpose_overlap_split_area_ratio)
        if source_type == "annotated_png":
            split_area_ratio = min(split_area_ratio, 1.45)

        area_gate = area_px >= median_area * split_area_ratio
        if source_type == "annotated_png" and cfg.detector.cellpose_overlap_split_annotated_ignore_shape:
            eligible = area_gate
        else:
            eligible = (
                area_gate
                and aspect_ratio <= cfg.detector.cellpose_overlap_split_max_aspect_ratio
                and eccentricity <= cfg.detector.cellpose_overlap_split_max_eccentricity
            )

        local_mask = row_dict["mask"].astype(bool)
        valid_regions = []
        split_labels = None
        if eligible:
            distance = ndi.distance_transform_edt(local_mask)
            peaks = feature.peak_local_max(
                distance,
                min_distance=cfg.detector.cellpose_overlap_split_peak_min_distance,
                threshold_abs=cfg.detector.cellpose_overlap_split_peak_abs_threshold,
                labels=local_mask.astype(np.uint8),
            )
            if 2 <= peaks.shape[0] <= cfg.detector.cellpose_overlap_split_max_children:
                markers = np.zeros(local_mask.shape, dtype=np.int32)
                for marker_index, (peak_row, peak_col) in enumerate(peaks, start=1):
                    markers[peak_row, peak_col] = marker_index
                markers, _ = ndi.label(markers > 0)
                split_labels = segmentation.watershed(-distance, markers, mask=local_mask)
                valid_regions = [
                    region
                    for region in measure.regionprops(split_labels)
                    if float(region.area) >= min_child_area_px
                ]

        if len(valid_regions) < 2 or len(valid_regions) > cfg.detector.cellpose_overlap_split_max_children:
            valid_regions = _combo_regions(local_mask, row_dict)
            if valid_regions:
                combo = _build_combo_map(
                    local_mask,
                    row_dict,
                    brown_weight=float(cfg.detector.cellpose_overlap_split_combo_brown_weight),
                )
                combo_peaks = feature.peak_local_max(
                    combo,
                    min_distance=cfg.detector.cellpose_overlap_split_combo_peak_min_distance,
                    threshold_abs=cfg.detector.cellpose_overlap_split_combo_peak_abs_threshold,
                    labels=local_mask.astype(np.uint8),
                )
                combo_peak_cap = max(cfg.detector.cellpose_overlap_split_max_children * 3, 8)
                if combo_peaks.shape[0] > combo_peak_cap:
                    combo_peaks = combo_peaks[:combo_peak_cap]
                markers = np.zeros(local_mask.shape, dtype=np.int32)
                for marker_index, (peak_row, peak_col) in enumerate(combo_peaks, start=1):
                    markers[peak_row, peak_col] = marker_index
                markers, _ = ndi.label(markers > 0)
                split_labels = segmentation.watershed(-combo, markers, mask=local_mask)
        if len(valid_regions) < 2 or len(valid_regions) > cfg.detector.cellpose_overlap_split_max_children:
            pair_labels, valid_regions = _pairlike_split(local_mask, row_dict)
            if valid_regions:
                split_labels = pair_labels
        if len(valid_regions) < 2 or len(valid_regions) > cfg.detector.cellpose_overlap_split_max_children:
            rows.append(row_dict)
            continue

        row_dict["is_active"] = False
        row_dict["cellpose_overlap_split_applied"] = True
        row_dict["split_children_count"] = len(valid_regions)
        rows.append(row_dict)

        y0 = int(row_dict["bbox_y0"])
        x0 = int(row_dict["bbox_x0"])
        for child_index, region in enumerate(valid_regions, start=1):
            child_mask = split_labels == region.label
            child_id = "%s_split_%02d" % (row_dict["component_id"], child_index)
            child_row = build_component_row(
                child_mask,
                y0,
                x0,
                image_shape,
                child_id,
                parent_component_id=row_dict["component_id"],
                split_from_cluster=False,
            )
            child_row["cellpose_overlap_child"] = True
            child_rows.append(child_row)

    if child_rows:
        combined = pd.concat([pd.DataFrame(rows), pd.DataFrame(child_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(rows)
    if combined.empty:
        return components_df.copy()
    return combined.sort_values(["component_id"]).reset_index(drop=True)
