import argparse
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class SegmentationConfig:
    hue_min: int = 20
    hue_max: int = 95
    sat_min: int = 35
    sat_max: int = 255
    val_min: int = 80
    val_max: int = 255
    dark_val_max: int = 85
    dark_sat_max: int = 140
    dark_adjacency_radius: int = 19
    close_kernel: int = 5
    open_kernel: int = 3
    component_min_area: int = 800
    envelope_step: float = 8.0
    boundary_window: int = 5
    boundary_min_bin_points: int = 1
    boundary_point_window: int = 3
    boundary_point_neighborhood: int = 1
    boundary_point_min_prominence: float = 1.0
    boundary_point_min_separation_px: float = 0.0
    guide_spacing: float = 15.0
    guide_window: int = 150
    coil_pair_max_s_gap: float = 1500.0
    coil_pair_max_tangent_offset: float = 950.0
    coil_pair_min_normal_extent: float = 8.0
    coil_pair_min_width_px: float = 8.0
    coil_pair_max_width_px: float = 900.0
    coil_midpoint_min_separation_px: float = 0.0
def choose_video_file() -> str:
    if platform.system() == "Darwin":
        script = """
        POSIX path of (choose file with prompt "Select a slinky video" of type {"public.movie"})
        """
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        if result.returncode == 1:
            return ""
        raise RuntimeError(result.stderr.strip() or "macOS file picker failed")

    raise RuntimeError("A file picker is only implemented for macOS in this script.")


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def scale_for_display(frame: np.ndarray, max_width: int = 1600) -> np.ndarray:
    if max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def odd_window(window: int, length: int) -> int:
    window = max(3, int(window))
    window = min(window, length if length % 2 == 1 else length - 1)
    if window < 3:
        return 0
    if window % 2 == 0:
        window -= 1
    return window


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if len(values) == 0 or not np.any(valid):
        return values
    if np.count_nonzero(valid) == 1:
        values[~valid] = values[valid][0]
        return values
    idx = np.arange(len(values), dtype=np.float32)
    values[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    return values


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 3:
        return values
    window = odd_window(window, len(values))
    if window < 3:
        return values
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def cumulative_arclength(points_xy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if points_xy is None or len(points_xy) < 2:
        return None
    diffs = np.diff(np.asarray(points_xy, dtype=np.float32), axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)


def resample_polyline_by_spacing(points_xy: Optional[np.ndarray], spacing_px: float) -> Optional[np.ndarray]:
    if points_xy is None or len(points_xy) < 2:
        return None
    s = cumulative_arclength(points_xy)
    if s is None or s[-1] <= 1e-6:
        return None
    spacing_px = max(2.0, float(spacing_px))
    sample_s = np.arange(0.0, s[-1] + 1e-6, spacing_px, dtype=np.float32)
    if len(sample_s) < 2:
        sample_s = np.array([0.0, s[-1]], dtype=np.float32)
    points_xy = np.asarray(points_xy, dtype=np.float32)
    x = np.interp(sample_s, s, points_xy[:, 0])
    y = np.interp(sample_s, s, points_xy[:, 1])
    return np.column_stack([x, y]).astype(np.float32)


def smooth_polyline(points_xy: Optional[np.ndarray], window: int) -> Optional[np.ndarray]:
    if points_xy is None or len(points_xy) < 3:
        return points_xy
    points_xy = np.asarray(points_xy, dtype=np.float32)
    x = smooth_series(points_xy[:, 0], window)
    y = smooth_series(points_xy[:, 1], window)
    return np.column_stack([x, y]).astype(np.float32)


def compute_polyline_tangents(points_xy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if points_xy is None or len(points_xy) < 2:
        return None
    points_xy = np.asarray(points_xy, dtype=np.float32)
    tangents = np.zeros_like(points_xy, dtype=np.float32)
    tangents[0] = points_xy[1] - points_xy[0]
    tangents[-1] = points_xy[-1] - points_xy[-2]
    if len(points_xy) > 2:
        tangents[1:-1] = points_xy[2:] - points_xy[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-6
    tangents[valid] = tangents[valid] / norms[valid]
    tangents[~valid] = np.nan
    return tangents


def compute_polyline_normals(tangents_xy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if tangents_xy is None or len(tangents_xy) == 0:
        return None
    tangents_xy = np.asarray(tangents_xy, dtype=np.float32)
    normals = np.column_stack([-tangents_xy[:, 1], tangents_xy[:, 0]]).astype(np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-6
    normals[valid] = normals[valid] / norms[valid]
    normals[~valid] = np.nan
    return normals


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    panel = image.copy()
    cv2.putText(panel, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(panel, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def draw_points(
    frame_bgr: np.ndarray,
    points_xy: Optional[np.ndarray],
    color,
    radius: int = 9,
    center_color=None,
) -> None:
    if points_xy is None or len(points_xy) == 0:
        return
    if center_color is None:
        center_color = (255, 255, 255)
    pts = np.round(points_xy).astype(np.int32)
    for x, y in pts:
        center = (int(x), int(y))
        cv2.circle(frame_bgr, center, radius + 2, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame_bgr, center, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame_bgr, center, max(2, radius // 3), center_color, -1, lineType=cv2.LINE_AA)


def draw_segments(frame_bgr: np.ndarray, segments_xy, color, thickness: int = 3) -> None:
    if segments_xy is None:
        return
    for segment in segments_xy:
        if segment is None or len(segment) != 2:
            continue
        p0 = tuple(np.round(segment[0]).astype(np.int32))
        p1 = tuple(np.round(segment[1]).astype(np.int32))
        cv2.line(frame_bgr, p0, p1, color, thickness, lineType=cv2.LINE_AA)


def compute_pca_axes(points_xy: np.ndarray):
    centroid = np.mean(points_xy, axis=0)
    centered = points_xy - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = eigvecs[:, order[1]]
    if major[0] < 0:
        major = -major
        minor = -minor
    return centroid.astype(np.float32), major.astype(np.float32), minor.astype(np.float32)


def estimate_axis(component_mask: np.ndarray):
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 10:
        return None
    points = np.column_stack([xs, ys]).astype(np.float32)
    centroid, major, minor = compute_pca_axes(points)
    return {"points": points, "centroid": centroid, "major": major, "minor": minor}


def extract_boundaries(component_mask: np.ndarray, cfg: SegmentationConfig):
    axis_info = estimate_axis(component_mask)
    if axis_info is None:
        return None, None

    points = axis_info["points"]
    centroid = axis_info["centroid"]
    major = axis_info["major"]
    minor = axis_info["minor"]

    rel = points - centroid
    u = rel @ major
    v = rel @ minor
    u_min = float(np.min(u))
    u_max = float(np.max(u))
    if (u_max - u_min) <= 1e-6:
        return None, None

    step = max(1.0, float(cfg.envelope_step))
    u_samples = np.arange(u_min, u_max + step, step, dtype=np.float32)
    if len(u_samples) < 2:
        return None, None

    half_step = 0.5 * step
    bins = np.floor((u - (u_min - half_step)) / step).astype(np.int32)
    keep = (bins >= 0) & (bins < len(u_samples))
    if not np.any(keep):
        return None, None

    bins = bins[keep]
    values = v[keep]
    counts = np.bincount(bins, minlength=len(u_samples))
    upper_v = np.full(len(u_samples), np.inf, dtype=np.float32)
    lower_v = np.full(len(u_samples), -np.inf, dtype=np.float32)
    np.minimum.at(upper_v, bins, values)
    np.maximum.at(lower_v, bins, values)

    min_points = max(1, int(cfg.boundary_min_bin_points))
    upper_v[counts < min_points] = np.nan
    lower_v[counts < min_points] = np.nan
    valid = np.isfinite(upper_v) & np.isfinite(lower_v)
    if np.count_nonzero(valid) < 2:
        return None, None

    upper_v = smooth_series(interpolate_nans(upper_v), cfg.boundary_window)
    lower_v = smooth_series(interpolate_nans(lower_v), cfg.boundary_window)
    upper = centroid + np.outer(u_samples, major) + np.outer(upper_v, minor)
    lower = centroid + np.outer(u_samples, major) + np.outer(lower_v, minor)
    return upper.astype(np.float32), lower.astype(np.float32)


def build_guide_curve(
    upper_boundary_xy: Optional[np.ndarray],
    lower_boundary_xy: Optional[np.ndarray],
    cfg: SegmentationConfig,
) -> Optional[np.ndarray]:
    if upper_boundary_xy is None or lower_boundary_xy is None:
        return None
    guide_xy = 0.5 * (np.asarray(upper_boundary_xy, dtype=np.float32) + np.asarray(lower_boundary_xy, dtype=np.float32))
    guide_xy = smooth_polyline(guide_xy, cfg.guide_window)
    guide_xy = resample_polyline_by_spacing(guide_xy, cfg.guide_spacing)
    guide_xy = smooth_polyline(guide_xy, cfg.guide_window)
    return None if guide_xy is None else guide_xy.astype(np.float32)


def project_points_to_guide(
    points_xy: Optional[np.ndarray],
    guide_xy: Optional[np.ndarray],
):
    if points_xy is None or len(points_xy) == 0 or guide_xy is None or len(guide_xy) < 2:
        return None, None
    points_xy = np.asarray(points_xy, dtype=np.float32)
    guide_xy = np.asarray(guide_xy, dtype=np.float32)
    dists = np.linalg.norm(points_xy[:, None, :] - guide_xy[None, :, :], axis=2)
    guide_indices = np.argmin(dists, axis=1).astype(np.int32)
    guide_s = cumulative_arclength(guide_xy)
    if guide_s is None:
        return guide_indices, None
    return guide_indices, guide_s[guide_indices].astype(np.float32)


def boundary_point_prominence(y_values: np.ndarray, index: int, mode: str, radius: int) -> float:
    left = y_values[max(0, index - radius):index]
    right = y_values[index + 1:min(len(y_values), index + 1 + radius)]
    if len(left) == 0 or len(right) == 0:
        return 0.0
    center = float(y_values[index])
    if mode == "upper":
        return float(min(np.max(left) - center, np.max(right) - center))
    return float(min(center - np.min(left), center - np.min(right)))


def extract_boundary_points(boundary_xy: Optional[np.ndarray], mode: str, cfg: SegmentationConfig) -> Optional[np.ndarray]:
    if boundary_xy is None or len(boundary_xy) < 3:
        return None

    y_values = smooth_series(boundary_xy[:, 1], cfg.boundary_point_window)
    neighborhood = max(1, int(cfg.boundary_point_neighborhood))
    prominence_radius = max(neighborhood * 2, 3)
    candidates = []

    for index in range(neighborhood, len(y_values) - neighborhood):
        left_value = float(y_values[index - 1])
        center_value = float(y_values[index])
        right_value = float(y_values[index + 1])
        window = y_values[index - neighborhood:index + neighborhood + 1]

        if mode == "upper":
            is_extremum = center_value <= left_value and center_value <= right_value and center_value <= float(np.min(window))
        else:
            is_extremum = center_value >= left_value and center_value >= right_value and center_value >= float(np.max(window))

        if not is_extremum:
            continue

        prominence = boundary_point_prominence(y_values, index, mode=mode, radius=prominence_radius)
        if prominence < float(cfg.boundary_point_min_prominence):
            continue

        candidates.append({"index": index, "prominence": prominence})

    if not candidates:
        return None

    if float(cfg.boundary_point_min_separation_px) <= 0.0:
        indices = [candidate["index"] for candidate in candidates]
        return boundary_xy[indices].astype(np.float32)

    min_separation = max(1, int(round(cfg.boundary_point_min_separation_px / max(float(cfg.envelope_step), 1.0))))
    selected = []
    for candidate in candidates:
        if not selected:
            selected.append(candidate)
            continue
        if candidate["index"] - selected[-1]["index"] < min_separation:
            if candidate["prominence"] > selected[-1]["prominence"]:
                selected[-1] = candidate
            continue
        selected.append(candidate)

    indices = [candidate["index"] for candidate in selected]
    return boundary_xy[indices].astype(np.float32)


def pair_boundary_points(
    upper_points_xy: Optional[np.ndarray],
    lower_points_xy: Optional[np.ndarray],
    guide_xy: Optional[np.ndarray],
    cfg: SegmentationConfig,
):
    empty = {
        "guide_xy": guide_xy,
        "cross_sections_xy": [],
        "midpoints_xy": None,
    }
    if (
        guide_xy is None
        or len(guide_xy) < 3
        or upper_points_xy is None
        or lower_points_xy is None
        or len(upper_points_xy) == 0
        or len(lower_points_xy) == 0
    ):
        return empty

    upper_points_xy = np.asarray(upper_points_xy, dtype=np.float32)
    lower_points_xy = np.asarray(lower_points_xy, dtype=np.float32)
    guide_xy = np.asarray(guide_xy, dtype=np.float32)
    guide_tangents_xy = compute_polyline_tangents(guide_xy)
    guide_normals_xy = compute_polyline_normals(guide_tangents_xy)
    if guide_tangents_xy is None or guide_normals_xy is None:
        return empty

    upper_guide_indices, upper_s = project_points_to_guide(upper_points_xy, guide_xy)
    lower_guide_indices, lower_s = project_points_to_guide(lower_points_xy, guide_xy)
    if upper_s is None or lower_s is None:
        return empty

    upper_order = np.argsort(upper_s)
    lower_order = np.argsort(lower_s)
    upper_points_xy = upper_points_xy[upper_order]
    lower_points_xy = lower_points_xy[lower_order]
    upper_s = upper_s[upper_order]
    lower_s = lower_s[lower_order]
    upper_guide_indices = upper_guide_indices[upper_order]
    lower_guide_indices = lower_guide_indices[lower_order]

    merged_events = sorted(
        [(float(value), "upper") for value in upper_s] + [(float(value), "lower") for value in lower_s],
        key=lambda item: item[0],
    )
    adjacent_opposite_gaps = [
        merged_events[index + 1][0] - merged_events[index][0]
        for index in range(len(merged_events) - 1)
        if merged_events[index][1] != merged_events[index + 1][1]
    ]
    target_s_gap = (
        float(np.median(np.asarray(adjacent_opposite_gaps, dtype=np.float32)))
        if adjacent_opposite_gaps
        else 0.5 * float(cfg.coil_pair_max_s_gap)
    )
    target_s_gap = max(12.0, target_s_gap)

    def make_candidate(upper_index: int, lower_index: int):
        s_gap = abs(float(lower_s[lower_index] - upper_s[upper_index]))
        if s_gap <= 0.0 or s_gap > float(cfg.coil_pair_max_s_gap):
            return None
        if s_gap < 0.15 * target_s_gap:
            return None

        guide_index = int(
            np.clip(
                int(round(0.5 * (int(upper_guide_indices[upper_index]) + int(lower_guide_indices[lower_index])))),
                0,
                len(guide_xy) - 1,
            )
        )
        tangent_xy = guide_tangents_xy[guide_index]
        normal_xy = guide_normals_xy[guide_index]
        pair_vector_xy = lower_points_xy[lower_index] - upper_points_xy[upper_index]
        pair_width = float(np.linalg.norm(pair_vector_xy))
        if pair_width < float(cfg.coil_pair_min_width_px) or pair_width > float(cfg.coil_pair_max_width_px):
            return None

        tangent_offset = 0.0
        normal_extent = pair_width
        if np.all(np.isfinite(tangent_xy)) and np.all(np.isfinite(normal_xy)):
            tangent_offset = abs(float(pair_vector_xy @ tangent_xy))
            normal_extent = abs(float(pair_vector_xy @ normal_xy))
        if tangent_offset > float(cfg.coil_pair_max_tangent_offset):
            return None
        if normal_extent < float(cfg.coil_pair_min_normal_extent):
            return None

        return {
            "upper_index": int(upper_index),
            "lower_index": int(lower_index),
            "center_s": 0.5 * (float(upper_s[upper_index]) + float(lower_s[lower_index])),
            "s_gap": s_gap,
            "tangent_offset": float(tangent_offset),
            "pair_width": float(pair_width),
            "segment_xy": np.vstack([upper_points_xy[upper_index], lower_points_xy[lower_index]]).astype(np.float32),
        }

    candidate_lookup = {}
    candidate_widths = []
    candidate_scores = []
    for upper_index in range(len(upper_points_xy)):
        for lower_index in range(len(lower_points_xy)):
            candidate = make_candidate(upper_index, lower_index)
            if candidate is None:
                continue
            candidate_lookup[(upper_index, lower_index)] = candidate
            candidate_widths.append(candidate["pair_width"])

    if not candidate_lookup:
        return empty

    target_width = float(np.median(np.asarray(candidate_widths, dtype=np.float32)))
    for candidate in candidate_lookup.values():
        candidate["score"] = (
            abs(candidate["s_gap"] - target_s_gap)
            + 0.03 * candidate["tangent_offset"]
            + 0.08 * abs(candidate["pair_width"] - target_width)
        )
        candidate_scores.append(candidate["score"])

    base_skip_penalty = max(
        0.55 * target_s_gap,
        float(np.median(np.asarray(candidate_scores, dtype=np.float32))) + 35.0,
    )

    upper_count = len(upper_points_xy)
    lower_count = len(lower_points_xy)
    dp = np.full((upper_count + 1, lower_count + 1), np.inf, dtype=np.float32)
    move = np.zeros((upper_count + 1, lower_count + 1), dtype=np.int8)
    dp[0, 0] = 0.0

    def skip_penalty(index: int, length: int) -> float:
        if index < 2 or index >= length - 2:
            return 0.65 * base_skip_penalty
        return base_skip_penalty

    def maybe_update(next_i: int, next_j: int, new_cost: float, move_code: int) -> None:
        current_cost = float(dp[next_i, next_j])
        if new_cost < current_cost - 1e-6:
            dp[next_i, next_j] = float(new_cost)
            move[next_i, next_j] = move_code
            return
        if abs(new_cost - current_cost) <= 1e-6 and move_code == 3 and move[next_i, next_j] != 3:
            dp[next_i, next_j] = float(new_cost)
            move[next_i, next_j] = move_code

    for upper_done in range(upper_count + 1):
        for lower_done in range(lower_count + 1):
            current_cost = float(dp[upper_done, lower_done])
            if not np.isfinite(current_cost):
                continue

            if upper_done < upper_count:
                maybe_update(
                    upper_done + 1,
                    lower_done,
                    current_cost + skip_penalty(upper_done, upper_count),
                    1,
                )
            if lower_done < lower_count:
                maybe_update(
                    upper_done,
                    lower_done + 1,
                    current_cost + skip_penalty(lower_done, lower_count),
                    2,
                )
            if upper_done < upper_count and lower_done < lower_count:
                candidate = candidate_lookup.get((upper_done, lower_done))
                if candidate is not None:
                    maybe_update(
                        upper_done + 1,
                        lower_done + 1,
                        current_cost + float(candidate["score"]),
                        3,
                    )

    selected_pairs = []
    upper_done = upper_count
    lower_done = lower_count
    while upper_done > 0 or lower_done > 0:
        move_code = int(move[upper_done, lower_done])
        if move_code == 3:
            candidate = candidate_lookup.get((upper_done - 1, lower_done - 1))
            if candidate is not None:
                selected_pairs.append(candidate)
            upper_done -= 1
            lower_done -= 1
            continue
        if move_code == 1:
            upper_done -= 1
            continue
        if move_code == 2:
            lower_done -= 1
            continue
        if upper_done > 0 and lower_done > 0:
            upper_done -= 1
            lower_done -= 1
        elif upper_done > 0:
            upper_done -= 1
        else:
            lower_done -= 1

    selected_pairs.reverse()

    if not selected_pairs:
        return empty

    selected_pairs = sorted(selected_pairs, key=lambda item: item["center_s"])
    if float(cfg.coil_midpoint_min_separation_px) > 0.0:
        filtered_pairs = []
        for candidate in selected_pairs:
            if not filtered_pairs:
                filtered_pairs.append(candidate)
                continue
            if candidate["center_s"] - filtered_pairs[-1]["center_s"] < float(cfg.coil_midpoint_min_separation_px):
                if candidate["score"] < filtered_pairs[-1]["score"]:
                    filtered_pairs[-1] = candidate
                continue
            filtered_pairs.append(candidate)
        selected_pairs = filtered_pairs

    cross_sections_xy = [candidate["segment_xy"] for candidate in selected_pairs]
    midpoints_xy = np.vstack(
        [0.5 * (candidate["segment_xy"][0] + candidate["segment_xy"][1]) for candidate in selected_pairs]
    ).astype(np.float32)
    return {
        "guide_xy": guide_xy.astype(np.float32),
        "cross_sections_xy": cross_sections_xy,
        "midpoints_xy": midpoints_xy,
    }


def render_side_by_side(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    upper_boundary_points: Optional[np.ndarray] = None,
    lower_boundary_points: Optional[np.ndarray] = None,
    cross_sections_xy=None,
    midpoint_points_xy: Optional[np.ndarray] = None,
) -> np.ndarray:
    original_overlay = frame_bgr.copy()
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    draw_segments(original_overlay, cross_sections_xy, (0, 215, 255), thickness=3)
    draw_segments(mask_bgr, cross_sections_xy, (0, 215, 255), thickness=3)
    draw_points(original_overlay, upper_boundary_points, (255, 0, 0), radius=12, center_color=(255, 220, 220))
    draw_points(original_overlay, lower_boundary_points, (255, 0, 0), radius=12, center_color=(255, 220, 220))
    draw_points(mask_bgr, upper_boundary_points, (255, 0, 0), radius=12, center_color=(255, 220, 220))
    draw_points(mask_bgr, lower_boundary_points, (255, 0, 0), radius=12, center_color=(255, 220, 220))
    draw_points(original_overlay, midpoint_points_xy, (0, 0, 255), radius=13, center_color=(210, 210, 255))
    draw_points(mask_bgr, midpoint_points_xy, (0, 0, 255), radius=13, center_color=(210, 210, 255))

    left = label_panel(original_overlay, "Original + Pairs")
    right = label_panel(mask_bgr, "Binary Mask + Pairs")
    return cv2.hconcat([left, right])


def draw_fps_box(frame_bgr: np.ndarray, fps_value: Optional[float]) -> np.ndarray:
    overlay = frame_bgr.copy()
    text = "FPS: --" if fps_value is None else f"FPS: {fps_value:4.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x = 16
    pad_y = 12
    margin = 18
    x1 = max(0, frame_bgr.shape[1] - text_w - (2 * pad_x) - margin)
    y1 = margin
    x2 = min(frame_bgr.shape[1] - 1, frame_bgr.shape[1] - margin)
    y2 = min(frame_bgr.shape[0] - 1, y1 + text_h + baseline + (2 * pad_y))

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (235, 235, 235), 2, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, frame_bgr, 0.28, 0.0, frame_bgr)

    text_x = x1 + pad_x
    text_y = y1 + pad_y + text_h
    cv2.putText(frame_bgr, text, (text_x, text_y), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame_bgr


def show_preview(window_name: str, preview: np.ndarray, max_width: int, fps_value: Optional[float] = None) -> bool:
    try:
        preview_for_display = draw_fps_box(preview.copy(), fps_value)
        cv2.imshow(window_name, scale_for_display(preview_for_display, max_width=max_width))
        return True
    except cv2.error as exc:
        print(f"[Warn] Live preview disabled because OpenCV could not open a display window: {exc}")
        return False


def handle_preview_controls(wait_ms: int) -> bool:
    key = cv2.waitKey(wait_ms) & 0xFF
    if key in (27, ord("q")):
        return False
    if key == ord(" "):
        while True:
            pause_key = cv2.waitKey(0) & 0xFF
            if pause_key in (27, ord("q")):
                return False
            if pause_key == ord(" "):
                break
    return True


def select_best_component(mask: np.ndarray, prev_centroid: Optional[Tuple[float, float]], cfg: SegmentationConfig):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_label = None
    best_score = -1e18
    fallback_label = None
    fallback_area = 0

    for label in range(1, num_labels):
        _, _, w, h, area = stats[label]
        if area > fallback_area:
            fallback_area = int(area)
            fallback_label = label
        if area < cfg.component_min_area:
            continue

        elongation = max(w, h) / max(1.0, min(w, h))
        score = float(area) + 300.0 * elongation

        if prev_centroid is not None:
            score -= 8.0 * math.hypot(
                centroids[label][0] - prev_centroid[0],
                centroids[label][1] - prev_centroid[1],
            )

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        best_label = fallback_label

    component = np.zeros_like(mask)
    centroid = None
    if best_label is not None:
        component = (labels == best_label).astype(np.uint8) * 255
        centroid = (float(centroids[best_label][0]), float(centroids[best_label][1]))

    return component, centroid


def segment_slinky_region(
    frame_bgr: np.ndarray,
    cfg: SegmentationConfig,
    prev_centroid: Optional[Tuple[float, float]] = None,
):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(
        hsv,
        np.array([cfg.hue_min, cfg.sat_min, cfg.val_min], dtype=np.uint8),
        np.array([cfg.hue_max, cfg.sat_max, cfg.val_max], dtype=np.uint8),
    )

    _, sat, val = cv2.split(hsv)
    dark_mask = ((val <= cfg.dark_val_max) & (sat <= cfg.dark_sat_max)).astype(np.uint8) * 255
    adjacency_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.dark_adjacency_radius, cfg.dark_adjacency_radius),
    )
    dark_mask = cv2.bitwise_and(dark_mask, cv2.dilate(color_mask, adjacency_kernel, iterations=1))

    merged_mask = cv2.bitwise_or(color_mask, dark_mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
    clean_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    component_mask, centroid = select_best_component(clean_mask, prev_centroid, cfg)
    upper_boundary, lower_boundary = extract_boundaries(component_mask, cfg)
    guide_xy = build_guide_curve(upper_boundary, lower_boundary, cfg)
    upper_boundary_points = extract_boundary_points(upper_boundary, mode="upper", cfg=cfg)
    lower_boundary_points = extract_boundary_points(lower_boundary, mode="lower", cfg=cfg)
    return {
        "binary_mask": component_mask,
        "centroid": centroid,
        "upper_boundary_curve": upper_boundary,
        "lower_boundary_curve": lower_boundary,
        "guide_xy": guide_xy,
        "color_mask": color_mask,
        "dark_mask": dark_mask,
        "merged_mask": merged_mask,
        "upper_boundary_points": upper_boundary_points,
        "lower_boundary_points": lower_boundary_points,
    }


def process_video(
    input_video: str,
    output_mask_video: str,
    output_side_by_side: str,
    output_summary: str,
    output_boundary_json: Optional[str] = None,
    mask_dir: Optional[str] = None,
    cfg: Optional[SegmentationConfig] = None,
    show: bool = True,
    display_max_width: int = 1600,
    playback_fps: float = 0.0,
    window_name: str = "Slinky Binary Segmentation",
) -> None:
    cfg = cfg or SegmentationConfig()
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        raise RuntimeError("Could not read the video frame size.")

    ensure_dir(os.path.dirname(output_mask_video))
    ensure_dir(os.path.dirname(output_side_by_side))
    ensure_dir(os.path.dirname(output_summary))
    if output_boundary_json:
        ensure_dir(os.path.dirname(output_boundary_json))
    if mask_dir:
        ensure_dir(mask_dir)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mask_writer = cv2.VideoWriter(output_mask_video, fourcc, fps, (frame_w, frame_h), False)
    side_writer = cv2.VideoWriter(output_side_by_side, fourcc, fps, (frame_w * 2, frame_h))
    if not mask_writer.isOpened() or not side_writer.isOpened():
        cap.release()
        mask_writer.release()
        side_writer.release()
        raise RuntimeError("Could not create one or more output video writers.")

    prev_centroid = None
    areas = []
    non_empty_frames = 0
    processed_frames = 0
    successful_boundary_frames = 0
    successful_midpoint_frames = 0
    upper_point_counts = []
    lower_point_counts = []
    midpoint_counts = []
    cross_section_counts = []
    boundary_records = []
    stopped_early = False
    preview_enabled = bool(show)
    preview_delay_ms = max(1, int(round(1000.0 / max(playback_fps if playback_fps > 0 else fps, 1.0))))
    last_preview_timestamp = None
    preview_fps_smoothed = None

    if preview_enabled:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            print(f"[Warn] Live preview disabled because OpenCV could not create a window: {exc}")
            preview_enabled = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = segment_slinky_region(frame, cfg, prev_centroid)
            mask = result["binary_mask"]
            prev_centroid = result["centroid"]
            upper_boundary_points = result["upper_boundary_points"]
            lower_boundary_points = result["lower_boundary_points"]
            pair_result = pair_boundary_points(
                upper_boundary_points,
                lower_boundary_points,
                result["guide_xy"],
                cfg,
            )
            cross_sections_xy = pair_result["cross_sections_xy"]
            midpoint_points_xy = pair_result["midpoints_xy"]

            area = int(np.count_nonzero(mask))
            areas.append(area)
            if area > 0:
                non_empty_frames += 1
            upper_count = 0 if upper_boundary_points is None else int(len(upper_boundary_points))
            lower_count = 0 if lower_boundary_points is None else int(len(lower_boundary_points))
            upper_point_counts.append(upper_count)
            lower_point_counts.append(lower_count)
            if upper_count > 0 and lower_count > 0:
                successful_boundary_frames += 1
            midpoint_count = 0 if midpoint_points_xy is None else int(len(midpoint_points_xy))
            midpoint_counts.append(midpoint_count)
            cross_section_counts.append(len(cross_sections_xy))
            if midpoint_count > 0:
                successful_midpoint_frames += 1

            preview = render_side_by_side(
                frame,
                mask,
                upper_boundary_points=upper_boundary_points,
                lower_boundary_points=lower_boundary_points,
                cross_sections_xy=cross_sections_xy,
                midpoint_points_xy=midpoint_points_xy,
            )
            mask_writer.write(mask)
            side_writer.write(preview)

            if mask_dir:
                mask_path = os.path.join(mask_dir, f"mask_{processed_frames:04d}.png")
                cv2.imwrite(mask_path, mask)

            if output_boundary_json:
                boundary_records.append(
                    {
                        "frame_index": int(processed_frames),
                        "centroid_xy": None if prev_centroid is None else [float(prev_centroid[0]), float(prev_centroid[1])],
                        "guide_xy": [] if pair_result["guide_xy"] is None else pair_result["guide_xy"].tolist(),
                        "upper_boundary_points_xy": [] if upper_boundary_points is None else upper_boundary_points.tolist(),
                        "lower_boundary_points_xy": [] if lower_boundary_points is None else lower_boundary_points.tolist(),
                        "cross_sections_xy": [segment.tolist() for segment in cross_sections_xy],
                        "midpoints_xy": [] if midpoint_points_xy is None else midpoint_points_xy.tolist(),
                    }
                )

            processed_frames += 1
            if preview_enabled:
                now = time.perf_counter()
                if last_preview_timestamp is not None:
                    dt = now - last_preview_timestamp
                    if dt > 1e-6:
                        preview_fps = 1.0 / dt
                        if preview_fps_smoothed is None:
                            preview_fps_smoothed = preview_fps
                        else:
                            preview_fps_smoothed = 0.85 * preview_fps_smoothed + 0.15 * preview_fps
                last_preview_timestamp = now
                preview_enabled = show_preview(
                    window_name,
                    preview,
                    max_width=display_max_width,
                    fps_value=preview_fps_smoothed,
                )
                if preview_enabled and not handle_preview_controls(preview_delay_ms):
                    stopped_early = True
                    break

            if processed_frames % 50 == 0:
                print(f"[Info] Processed {processed_frames}/{frame_count or '?'} frames...")
    finally:
        cap.release()
        mask_writer.release()
        side_writer.release()
        if show:
            cv2.destroyAllWindows()

    area_array = np.array(areas, dtype=np.float32) if areas else np.zeros(0, dtype=np.float32)
    summary = {
        "input_video": input_video,
        "output_mask_video": output_mask_video,
        "output_side_by_side": output_side_by_side,
        "output_boundary_json": output_boundary_json,
        "mask_dir": mask_dir,
        "frames_processed": processed_frames,
        "frame_count_reported_by_video": frame_count,
        "fps": fps,
        "frame_size": {"width": frame_w, "height": frame_h},
        "foreground_encoding": {
            "background_value": 0,
            "foreground_value": 255,
            "note": "255 is the 8-bit representation of logical white/1 in the written mask video and PNG files.",
        },
        "live_preview": {
            "enabled_requested": bool(show),
            "display_max_width": int(display_max_width),
            "playback_fps": float(playback_fps if playback_fps > 0 else fps),
            "window_name": window_name,
            "stopped_early": bool(stopped_early),
            "controls": "Press space to pause/resume. Press q or Esc to quit the preview and stop processing.",
        },
        "mask_area_stats": {
            "non_empty_frames": non_empty_frames,
            "min_foreground_pixels": int(area_array.min()) if len(area_array) else 0,
            "max_foreground_pixels": int(area_array.max()) if len(area_array) else 0,
            "mean_foreground_pixels": float(area_array.mean()) if len(area_array) else 0.0,
            "median_foreground_pixels": float(np.median(area_array)) if len(area_array) else 0.0,
        },
        "boundary_detection": {
            "frames_with_upper_and_lower_boundary_points": int(successful_boundary_frames),
            "envelope_step": float(cfg.envelope_step),
            "boundary_window": int(cfg.boundary_window),
            "boundary_min_bin_points": int(cfg.boundary_min_bin_points),
            "upper_point_count_mean": float(np.mean(upper_point_counts)) if upper_point_counts else 0.0,
            "lower_point_count_mean": float(np.mean(lower_point_counts)) if lower_point_counts else 0.0,
            "upper_point_count_max": int(max(upper_point_counts)) if upper_point_counts else 0,
            "lower_point_count_max": int(max(lower_point_counts)) if lower_point_counts else 0,
        },
        "coil_pairing": {
            "frames_with_midpoints": int(successful_midpoint_frames),
            "midpoint_count_mean": float(np.mean(midpoint_counts)) if midpoint_counts else 0.0,
            "midpoint_count_max": int(max(midpoint_counts)) if midpoint_counts else 0,
            "cross_section_count_mean": float(np.mean(cross_section_counts)) if cross_section_counts else 0.0,
            "cross_section_count_max": int(max(cross_section_counts)) if cross_section_counts else 0,
            "guide_spacing": float(cfg.guide_spacing),
            "guide_window": int(cfg.guide_window),
            "coil_pair_max_s_gap": float(cfg.coil_pair_max_s_gap),
            "coil_pair_max_tangent_offset": float(cfg.coil_pair_max_tangent_offset),
            "coil_pair_min_normal_extent": float(cfg.coil_pair_min_normal_extent),
            "coil_pair_min_width_px": float(cfg.coil_pair_min_width_px),
            "coil_pair_max_width_px": float(cfg.coil_pair_max_width_px),
            "coil_midpoint_min_separation_px": float(cfg.coil_midpoint_min_separation_px),
        },
        "config": asdict(cfg),
    }

    if output_boundary_json:
        with open(output_boundary_json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "input_video": input_video,
                    "frames": boundary_records,
                },
                fh,
                indent=2,
            )
    with open(output_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[Done] Saved binary mask video to: {output_mask_video}")
    print(f"[Done] Saved side-by-side video to: {output_side_by_side}")
    if output_boundary_json:
        print(f"[Done] Saved boundary coordinates to: {output_boundary_json}")
    print(f"[Done] Saved summary to: {output_summary}")
    if show:
        print("[Info] Live preview controls: space = pause/resume, q or Esc = quit.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment the full visible slinky as a binary foreground object.")
    parser.add_argument("--input", type=str, default="", help="Path to the input video. If omitted on macOS, a file picker is used.")
    parser.add_argument(
        "--output_mask_video",
        type=str,
        default="output/slinky_binary_mask.mp4",
        help="Path to the binary mask video.",
    )
    parser.add_argument(
        "--output_side_by_side",
        type=str,
        default="output/slinky_binary_side_by_side.mp4",
        help="Path to the original-vs-mask comparison video.",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="output/slinky_binary_summary.json",
        help="Path to the JSON run summary.",
    )
    parser.add_argument(
        "--output_boundary_json",
        type=str,
        default="output/slinky_binary_boundaries.json",
        help="Path to per-frame upper/lower boundary coordinates.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="",
        help="Optional directory to save one PNG binary mask per frame.",
    )
    parser.add_argument("--show", dest="show", action="store_true", help="Show the original and binary mask live while processing.")
    parser.add_argument("--no_show", dest="show", action="store_false", help="Disable the live preview window.")
    parser.set_defaults(show=True)
    parser.add_argument(
        "--display_max_width",
        type=int,
        default=1600,
        help="Maximum width of the live preview window contents before downscaling.",
    )
    parser.add_argument(
        "--playback_fps",
        type=float,
        default=0.0,
        help="Playback speed for the live preview. Use 0 to follow the source video FPS.",
    )
    parser.add_argument(
        "--window_name",
        type=str,
        default="Slinky Binary Segmentation",
        help="Window title used for the live preview.",
    )

    parser.add_argument("--hue_min", type=int, default=SegmentationConfig.hue_min)
    parser.add_argument("--hue_max", type=int, default=SegmentationConfig.hue_max)
    parser.add_argument("--sat_min", type=int, default=SegmentationConfig.sat_min)
    parser.add_argument("--sat_max", type=int, default=SegmentationConfig.sat_max)
    parser.add_argument("--val_min", type=int, default=SegmentationConfig.val_min)
    parser.add_argument("--val_max", type=int, default=SegmentationConfig.val_max)
    parser.add_argument("--dark_val_max", type=int, default=SegmentationConfig.dark_val_max)
    parser.add_argument("--dark_sat_max", type=int, default=SegmentationConfig.dark_sat_max)
    parser.add_argument("--dark_adjacency_radius", type=int, default=SegmentationConfig.dark_adjacency_radius)
    parser.add_argument("--close_kernel", type=int, default=SegmentationConfig.close_kernel)
    parser.add_argument("--open_kernel", type=int, default=SegmentationConfig.open_kernel)
    parser.add_argument("--component_min_area", type=int, default=SegmentationConfig.component_min_area)
    parser.add_argument("--envelope_step", type=float, default=SegmentationConfig.envelope_step)
    parser.add_argument("--boundary_window", type=int, default=SegmentationConfig.boundary_window)
    parser.add_argument("--boundary_min_bin_points", type=int, default=SegmentationConfig.boundary_min_bin_points)
    parser.add_argument("--boundary_point_window", type=int, default=SegmentationConfig.boundary_point_window)
    parser.add_argument("--boundary_point_neighborhood", type=int, default=SegmentationConfig.boundary_point_neighborhood)
    parser.add_argument("--boundary_point_min_prominence", type=float, default=SegmentationConfig.boundary_point_min_prominence)
    parser.add_argument(
        "--boundary_point_min_separation_px",
        type=float,
        default=SegmentationConfig.boundary_point_min_separation_px,
    )
    parser.add_argument("--guide_spacing", type=float, default=SegmentationConfig.guide_spacing)
    parser.add_argument("--guide_window", type=int, default=SegmentationConfig.guide_window)
    parser.add_argument("--coil_pair_max_s_gap", type=float, default=SegmentationConfig.coil_pair_max_s_gap)
    parser.add_argument(
        "--coil_pair_max_tangent_offset",
        type=float,
        default=SegmentationConfig.coil_pair_max_tangent_offset,
    )
    parser.add_argument(
        "--coil_pair_min_normal_extent",
        type=float,
        default=SegmentationConfig.coil_pair_min_normal_extent,
    )
    parser.add_argument("--coil_pair_min_width_px", type=float, default=SegmentationConfig.coil_pair_min_width_px)
    parser.add_argument("--coil_pair_max_width_px", type=float, default=SegmentationConfig.coil_pair_max_width_px)
    parser.add_argument(
        "--coil_midpoint_min_separation_px",
        type=float,
        default=SegmentationConfig.coil_midpoint_min_separation_px,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SegmentationConfig:
    return SegmentationConfig(
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        sat_min=args.sat_min,
        sat_max=args.sat_max,
        val_min=args.val_min,
        val_max=args.val_max,
        dark_val_max=args.dark_val_max,
        dark_sat_max=args.dark_sat_max,
        dark_adjacency_radius=args.dark_adjacency_radius,
        close_kernel=args.close_kernel,
        open_kernel=args.open_kernel,
        component_min_area=args.component_min_area,
        envelope_step=args.envelope_step,
        boundary_window=args.boundary_window,
        boundary_min_bin_points=args.boundary_min_bin_points,
        boundary_point_window=args.boundary_point_window,
        boundary_point_neighborhood=args.boundary_point_neighborhood,
        boundary_point_min_prominence=args.boundary_point_min_prominence,
        boundary_point_min_separation_px=args.boundary_point_min_separation_px,
        guide_spacing=args.guide_spacing,
        guide_window=args.guide_window,
        coil_pair_max_s_gap=args.coil_pair_max_s_gap,
        coil_pair_max_tangent_offset=args.coil_pair_max_tangent_offset,
        coil_pair_min_normal_extent=args.coil_pair_min_normal_extent,
        coil_pair_min_width_px=args.coil_pair_min_width_px,
        coil_pair_max_width_px=args.coil_pair_max_width_px,
        coil_midpoint_min_separation_px=args.coil_midpoint_min_separation_px,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.input:
        args.input = choose_video_file()
    if not args.input:
        raise SystemExit("No input video selected.")

    process_video(
        input_video=args.input,
        output_mask_video=args.output_mask_video,
        output_side_by_side=args.output_side_by_side,
        output_summary=args.output_summary,
        output_boundary_json=args.output_boundary_json or None,
        mask_dir=args.mask_dir or None,
        cfg=config_from_args(args),
        show=args.show,
        display_max_width=args.display_max_width,
        playback_fps=args.playback_fps,
        window_name=args.window_name,
    )
