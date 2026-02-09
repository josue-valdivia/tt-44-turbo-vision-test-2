#!/usr/bin/env python3
"""
Standalone keypoint scoring (no object scoring).

Scoring logic matches keypoints.py exactly, including:
  - Blacklist checking for suspect keypoint combinations
  - Edge proximity detection for keypoint pairs
  - Homography-based template projection
  - Mask validation and overlap scoring

Usage:
  uv run python test/keypoints_calculate_score.py \
    --video-url https://scoredata.me/chunks/5e58db821f044efd818fb42f588bea.mp4 \
    --miner-json test/miner_responses_ordered/test_5_new.json \
    [--video-id 5e58db821f044efd818fb42f588bea] \
    [--verbose]

Outputs:
  - scoring_result/<video_id>/###-<s>-<score>.jpg visualizations
    (s = parent dir name when miner-json is under temp_test/..., else "test")
  - keypoint_scores/<miner_json_stem>.json with per-frame scores and average
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

# Optional: process only specific frames (comment out to process all frames)
ONLY_FRAMES = list(range(2, 3))

# Debug flag: if True, output dtailed logs and step-by-step debug images
DEBUG_FLAG = False

# --------------------------- Data / Models --------------------------- #
class SVRunOutput:
    def __init__(
        self,
        success: bool = True,
        latency_ms: float = 0.0,
        predictions: Any | None = None,
        error: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.success = success
        self.latency_ms = latency_ms
        self.predictions = predictions
        self.error = error
        self.model = model


def parse_miner_prediction(miner_run: SVRunOutput) -> dict[int, dict]:
    predicted_frames = (miner_run.predictions or {}).get("frames") or []
    out: dict[int, dict] = {}
    for fr in predicted_frames:
        frame_id = fr.get("frame_id", fr.get("frame_number", -1))
        if frame_id is None:
            continue
        out[int(frame_id)] = {"keypoints": fr.get("keypoints") or []}
    return out


def _format_score_for_filename(score: float) -> str:
    return f"{score:.4f}"


# ------------------------- Video utilities -------------------------- #
class FrameStore:
    def __init__(self, source: str) -> None:
        self.cap = cv2.VideoCapture(source)
        self.video_path = source

    def get_frame(self, frame_id: int) -> np.ndarray:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id}")
        return frame

    def unlink(self) -> None:
        if self.cap:
            self.cap.release()


def download_video_cached(url: str, _frame_numbers: list[int] | None = None):
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as out:
                out.write(resp.read())
        except Exception as e:
            try:
                tmp_path.unlink()
            except Exception:
                pass
            raise RuntimeError(f"Failed to download video: {e}")
        return tmp_path, FrameStore(str(tmp_path))
    else:
        if not Path(url).exists():
            raise RuntimeError(f"Video path does not exist: {url}")
        return None, FrameStore(url)


# --------------------------- Keypoint data --------------------------- #
KEYPOINTS: list[tuple[int, int]] = [
    (5, 5), (5, 140), (5, 250), (5, 430), (5, 540), (5, 675),
    (55, 250), (55, 430),
    (110, 340),
    (165, 140), (165, 270), (165, 410), (165, 540),
    (527, 5), (527, 253), (527, 433), (527, 675),
    (888, 140), (888, 270), (888, 410), (888, 540),
    (940, 340),
    (998, 250), (998, 430),
    (1045, 5), (1045, 140), (1045, 250), (1045, 430), (1045, 540), (1045, 675),
    (435, 340), (615, 340),
]
# KEYPOINTS: list[tuple[int, int]] = [
#     (2.5, 2.5),
#     (2.5, 139.5),
#     (2.5, 249.5),
#     (2.5, 430.5),
#     (2.5, 540.5),
#     (2.5, 678),

#     (54.5, 249.5),
#     (54.5, 430.5),

#     (110.5, 340.5),

#     (164.5, 139.5),
#     (164.5, 269),
#     (164.5, 411),
#     (164.5, 540.5),

#     (525, 2.5),
#     (525, 249.5),
#     (525, 430.5),
#     (525, 678),

#     (886.5, 139.5),
#     (886.5, 269),
#     (886.5, 411),
#     (886.5, 540.5),

#     (940.5, 340.5),

#     (998, 249.5),
#     (998, 430.5),

#     (1048, 2.5),
#     (1048, 139.5),
#     (1048, 249.5),
#     (1048, 430.5),
#     (1048, 540.5),
#     (1048, 678),
    
#     (434.5, 340),
#     (615.5, 340)
# ]

# Corner keypoint indices for validation
INDEX_KEYPOINT_CORNER_BOTTOM_LEFT = 5
INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT = 29
INDEX_KEYPOINT_CORNER_TOP_LEFT = 0
INDEX_KEYPOINT_CORNER_TOP_RIGHT = 24


class InvalidMask(Exception):
    """Exception raised when mask validation fails."""
    pass


def challenge_template() -> np.ndarray:
    """Load the football pitch template image shipped alongside this script."""
    template_path = Path(__file__).resolve().parent / "football_pitch_template.png"
    img = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    return img


def _log_error(msg: str, frame_number: int | None = None, *, log_frame_number: bool = False) -> None:
    if log_frame_number and frame_number is not None:
        logging.error("%s (frame %s)", msg, frame_number)
    else:
        logging.error("%s", msg)


# --------------------------- Validation functions --------------------------- #
def has_a_wide_line(mask: np.ndarray, max_aspect_ratio: float = 1.0) -> bool:
    """Check if mask contains a line that is too wide (aspect ratio >= max_aspect_ratio)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        aspect_ratio = min(w, h) / max(w, h)
        if aspect_ratio >= max_aspect_ratio:
            return True
    return False


def is_bowtie(points: np.ndarray) -> bool:
    """Check if a quadrilateral (4 points) forms a bowtie (self-intersecting) shape."""
    def segments_intersect(p1: tuple[float, float], p2: tuple[float, float], 
                          q1: tuple[float, float], q2: tuple[float, float]) -> bool:
        def ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (
            ccw(p1, p2, q1) != ccw(p1, p2, q2)
        )

    pts = points.reshape(-1, 2)
    if len(pts) < 4:
        return False
    edges = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0])]
    return segments_intersect(*edges[0], *edges[2]) or segments_intersect(
        *edges[1], *edges[3]
    )


def validate_mask_lines(mask: np.ndarray) -> None:
    """Validate that the lines mask is reasonable."""
    if mask.sum() == 0:
        raise InvalidMask("No projected lines")
    if mask.sum() == mask.size:
        raise InvalidMask("Projected lines cover the entire image surface")
    if has_a_wide_line(mask=mask):
        raise InvalidMask("A projected line is too wide")


def validate_mask_ground(mask: np.ndarray) -> None:
    """Validate that the ground mask is reasonable."""
    num_labels, _ = cv2.connectedComponents(mask)
    num_distinct_regions = num_labels - 1
    if num_distinct_regions > 1:
        raise InvalidMask(
            f"Projected ground should be a single object, detected {num_distinct_regions}"
        )
    area_covered = mask.sum() / mask.size
    if area_covered >= 0.9:
        raise InvalidMask(
            f"Projected ground covers more than {area_covered:.2f}% of the image surface which is unrealistic"
        )


def validate_projected_corners(
    source_keypoints: list[tuple[int, int]], homography_matrix: np.ndarray
) -> None:
    """Validate that projected corners don't form a bowtie (twisted projection)."""
    if len(source_keypoints) <= max(INDEX_KEYPOINT_CORNER_BOTTOM_LEFT, 
                                     INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT,
                                     INDEX_KEYPOINT_CORNER_TOP_LEFT,
                                     INDEX_KEYPOINT_CORNER_TOP_RIGHT):
        return  # Not enough keypoints to validate
    
    src_corners = np.array(
        [
            source_keypoints[INDEX_KEYPOINT_CORNER_BOTTOM_LEFT],
            source_keypoints[INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT],
            source_keypoints[INDEX_KEYPOINT_CORNER_TOP_RIGHT],
            source_keypoints[INDEX_KEYPOINT_CORNER_TOP_LEFT],
        ],
        dtype=np.float32,
    )[None, :, :]

    warped_corners = cv2.perspectiveTransform(src_corners, homography_matrix)[0]

    if is_bowtie(warped_corners):
        raise InvalidMask("Projection twisted!")


def extract_mask_of_ground_lines_in_image(
    image: np.ndarray,
    ground_mask: np.ndarray,
    blur_ksize: int = 5,
    canny_low: int = 30,
    canny_high: int = 100,
    use_tophat: bool = True,
    dilate_kernel_size: int = 3,
    dilate_iterations: int = 3,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if use_tophat:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    if blur_ksize and blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges_on_ground = cv2.bitwise_and(edges, edges, mask=ground_mask)
    if dilate_kernel_size > 1:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size)
        )
        edges_on_ground = cv2.dilate(
            edges_on_ground, dilate_kernel, iterations=dilate_iterations
        )
    return (edges_on_ground > 0).astype(np.uint8)


# Blacklist configurations for suspect keypoint combinations (from keypoints.py)
blacklists = [
    [23, 24, 27, 28],
    [7, 8, 3, 4],
    [2, 10, 1, 14],
    [18, 26, 14, 25],
    [5, 13, 6, 17],
    [21, 29, 17, 30],
    [10, 11, 2, 3],
    [10, 11, 2, 7],
    [12, 13, 4, 5],
    [12, 13, 5, 8],
    [18, 19, 26, 27],
    [18, 19, 26, 23],
    [20, 21, 24, 29],
    [20, 21, 28, 29],
    [8, 4, 5, 13],
    [3, 7, 2, 10],
    [23, 27, 18, 26],
    [24, 28, 21, 29]
]


def near_edges(x: float, y: float, W: int, H: int, t: int = 50) -> set[str]:
    """Check if a point is near image edges."""
    edges = set()
    if x <= t:
        edges.add("left")
    if x >= W - t:
        edges.add("right")
    if y <= t:
        edges.add("top")
    if y >= H - t:
        edges.add("bottom")
    return edges


def both_points_same_direction(
    A: tuple[float, float], 
    B: tuple[float, float], 
    W: int, 
    H: int, 
    t: int = 100
) -> bool:
    """Check if both points are near the same edge(s) of the image."""
    edges_A = near_edges(A[0], A[1], W, H, t)
    edges_B = near_edges(B[0], B[1], W, H, t)
    
    if not edges_A or not edges_B:
        return False
    
    return not edges_A.isdisjoint(edges_B)


def evaluate_keypoints_for_frame(
    template_keypoints: list[tuple[int, int]],
    frame_keypoints: list[tuple[int, int]],
    frame: np.ndarray,
    floor_markings_template: np.ndarray,
    frame_number: int | None = None,
    *,
    log_frame_number: bool = False,
    debug_dir: Path | None = None,
) -> float:
    try:
        frame_id_str = f"Frame {frame_number}" if frame_number is not None else "Frame"
        frame_height, frame_width = frame.shape[:2]
        
        if DEBUG_FLAG:
            print(f"\n[DEBUG] {frame_id_str} - Starting keypoint evaluation")
            print(f"[DEBUG] {frame_id_str} - Input: template_keypoints={len(template_keypoints)}, frame_keypoints={len(frame_keypoints)}, frame_size=({frame_width}, {frame_height})")
        
        # Step 0: Validate and clamp keypoints that are out of bounds
        original_keypoints = frame_keypoints[:]
        frame_keypoints = [
            (0, 0) if (x, y) != (0, 0) and (x < 0 or y < 0 or x >= frame_width or y >= frame_height) else (x, y)
            for (x, y) in frame_keypoints
        ]
        
        if DEBUG_FLAG:
            clamped_count = sum(1 for orig, new in zip(original_keypoints, frame_keypoints) 
                              if orig != (0, 0) and new == (0, 0))
            if clamped_count > 0:
                print(f"[DEBUG] {frame_id_str} - Step 0: Clamped {clamped_count} out-of-bounds keypoints to (0, 0)")
        
        # Step 0.5: Blacklist checking (from keypoints.py) - check for suspect keypoint combinations
        non_idxs = []
        for idx, kpts in enumerate(frame_keypoints):
            if kpts[0] != 0 or kpts[1] != 0:
                non_idxs.append(idx + 1)
        
        for blacklist in blacklists:
            is_included = set(non_idxs).issubset(blacklist)
            if is_included:
                if both_points_same_direction(
                    frame_keypoints[blacklist[0] - 1], 
                    frame_keypoints[blacklist[1] - 1], 
                    frame_width, 
                    frame_height
                ):
                    if DEBUG_FLAG:
                        print(f"[DEBUG] {frame_id_str} - Step 0.5: Suspect keypoints detected (blacklist match)! Returning 0.0")
                    if log_frame_number and frame_number is not None:
                        logging.info("%s (frame %s)", "Suspect keypoints!", frame_number)
                    else:
                        logging.info("%s", "Suspect keypoints!")
                    return 0.0
        
        # Step 1-3: Use project_image_using_keypoints (matches original validator exactly)
        # This function handles filtering, homography, warping, and corner validation
        warped_template = None
        try:
            warped_template = project_image_using_keypoints(
                image=floor_markings_template,
                source_keypoints=template_keypoints,
                destination_keypoints=frame_keypoints,
                destination_width=frame_width,
                destination_height=frame_height,
            )
        except ValueError as e:
            # ValueError: < 4 valid keypoints or homography computation failed
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 1-3: Projection failed: {e}")
            if log_frame_number and frame_number is not None:
                logging.error("%s (frame %s)", str(e), frame_number)
            else:
                logging.error("%s", str(e))
            
            # Try to get warped_template even on error (before validation)
            if warped_template is None:
                try:
                    warped_template = _get_warped_template_before_validation(
                        image=floor_markings_template,
                        source_keypoints=template_keypoints,
                        destination_keypoints=frame_keypoints,
                        destination_width=frame_width,
                        destination_height=frame_height,
                    )
                except Exception:
                    pass  # If we can't get it, continue without saving
            
            # Save warped_template even on error if we have it
            if warped_template is not None and frame_number is not None:
                # Use debug_dir if provided, otherwise use default error output directory
                output_dir = debug_dir
                if output_dir is None:
                    output_dir = Path("error_outputs")
                
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_dir / f"frame_{frame_number:03d}_01_warped_template_ERROR_VALIDATION.png"), warped_template)
                    if DEBUG_FLAG:
                        print(f"[DEBUG] {frame_id_str} - Saved warped template (with ValueError) to {output_dir / f'frame_{frame_number:03d}_01_warped_template_ERROR_VALIDATION.png'}")
                except Exception as save_err:
                    if DEBUG_FLAG:
                        print(f"[DEBUG] {frame_id_str} - Failed to save warped template: {save_err}")
            
            return 0.0
        except InvalidMask as e:
            # InvalidMask: Projection twisted (bowtie check failed)
            # Note: warping succeeded but validation failed, so we can get the warped image
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 1-3: Projection failed: {e}")
            if log_frame_number and frame_number is not None:
                logging.error("%s (frame %s)", str(e), frame_number)
            else:
                logging.error("%s", str(e))
            
            # Always try to get warped_template since warping should have succeeded before validation
            # warped_template will be None because exception was raised before return
            try:
                warped_template = _get_warped_template_before_validation(
                    image=floor_markings_template,
                    source_keypoints=template_keypoints,
                    destination_keypoints=frame_keypoints,
                    destination_width=frame_width,
                    destination_height=frame_height,
                )
            except Exception:
                pass  # If we can't get it, continue without saving
            
            # Save warped_template even on error if we have it
            if warped_template is not None and frame_number is not None:
                # Use debug_dir if provided, otherwise use default error output directory
                output_dir = debug_dir
                if output_dir is None:
                    output_dir = Path("error_outputs")
                
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # Overlay keypoints as 3px red dots with index labels on the warped template
                    vis = warped_template.copy()
                    if vis.ndim == 2:  # grayscale -> BGR
                        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                    for idx_kp, kp in enumerate(frame_keypoints):
                        if kp and len(kp) >= 2 and not (kp[0] == 0 and kp[1] == 0):
                            x, y = int(kp[0]), int(kp[1])
                            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                                cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)  # red dot
                                cv2.putText(
                                    vis,
                                    str(idx_kp),
                                    (x + 6, y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 255),
                                    1,
                                    cv2.LINE_AA,
                                )

                    cv2.imwrite(str(output_dir / f"frame_{frame_number:03d}_01_warped_template_ERROR_BOWTIE.png"), vis)
                    if DEBUG_FLAG:
                        print(f"[DEBUG] {frame_id_str} - Saved warped template (with InvalidMask/bowtie) to {output_dir / f'frame_{frame_number:03d}_01_warped_template_ERROR_BOWTIE.png'}")
                except Exception as save_err:
                    if DEBUG_FLAG:
                        print(f"[DEBUG] {frame_id_str} - Failed to save warped template: {save_err}")
            
            return 0.0
        
        if DEBUG_FLAG and debug_dir is not None and frame_number is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            vis = warped_template.copy()
            if vis.ndim == 2:  # grayscale -> BGR
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            for idx_kp, kp in enumerate(frame_keypoints):
                if kp and len(kp) >= 2 and not (kp[0] == 0 and kp[1] == 0):
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)  # red dot
                        cv2.putText(
                            vis,
                            str(idx_kp),
                            (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
            cv2.imwrite(str(debug_dir / f"frame_{frame_number:03d}_01_warped_template.png"), vis)
            print(f"[DEBUG] {frame_id_str} - Step 3: Saved warped template with keypoints to {debug_dir / f'frame_{frame_number:03d}_01_warped_template.png'}")
        
        # Step 4: Extract masks with validation (matches original validator exactly)
        if DEBUG_FLAG:
            print(f"[DEBUG] {frame_id_str} - Step 4: Extracting ground and line masks from warped template")
        
        try:
            mask_ground_bin, mask_lines_expected = extract_masks_for_ground_and_lines(
                image=warped_template
            )
        except InvalidMask as e:
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 4: Mask validation failed: {e}")
            if log_frame_number and frame_number is not None:
                logging.error("%s (frame %s)", str(e), frame_number)
            else:
                logging.error("%s", str(e))
            return 0.0
        
        ground_pixels = int(mask_ground_bin.sum())
        expected_lines_pixels = int(mask_lines_expected.sum())
        
        if DEBUG_FLAG:
            print(f"[DEBUG] {frame_id_str} - Step 4: Ground mask pixels={ground_pixels}, Expected lines pixels={expected_lines_pixels}")
        
        if DEBUG_FLAG and debug_dir is not None and frame_number is not None:
            cv2.imwrite(str(debug_dir / f"frame_{frame_number:03d}_02_mask_ground.png"), mask_ground_bin * 255)
            cv2.imwrite(str(debug_dir / f"frame_{frame_number:03d}_03_mask_lines_expected.png"), mask_lines_expected * 255)
            print(f"[DEBUG] {frame_id_str} - Step 4: Saved ground mask and expected lines mask")
        
        # Step 5: Extract predicted lines from frame
        if DEBUG_FLAG:
            print(f"[DEBUG] {frame_id_str} - Step 5: Extracting predicted lines from frame using edge detection")
        
        # Extract raw edges from the full image (before masking)
        if DEBUG_FLAG:
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
            gray_tophat_full = cv2.morphologyEx(gray_full, cv2.MORPH_TOPHAT, kernel)
            gray_blur_full = cv2.GaussianBlur(gray_tophat_full, (5, 5), 0)
            edges_full = cv2.Canny(gray_blur_full, 30, 100)
            edges_full_pixels = int(edges_full.sum())
            print(f"[DEBUG] {frame_id_str} - Step 5: Full image edge detection pixels={edges_full_pixels}")
            
            if debug_dir is not None and frame_number is not None:
                # Create black background image
                edges_vis = np.zeros((edges_full.shape[0], edges_full.shape[1], 3), dtype=np.uint8)
                
                # Draw original keypoints as red plus signs (1px width) - bottom layer
                for kp in frame_keypoints:
                    if kp and len(kp) >= 2 and not (kp[0] == 0 and kp[1] == 0):
                        x, y = int(kp[0]), int(kp[1])
                        if 0 <= x < edges_vis.shape[1] and 0 <= y < edges_vis.shape[0]:
                            # Draw plus sign: horizontal and vertical lines, 5px length
                            line_length = 5
                            # Horizontal line
                            cv2.line(edges_vis, (x - line_length, y), (x + line_length, y), (0, 0, 255), 1)
                            # Vertical line
                            cv2.line(edges_vis, (x, y - line_length), (x, y + line_length), (0, 0, 255), 1)
                
                # Overlay white edge pixels on top layer
                white_edges = (edges_full > 0)
                edges_vis[white_edges] = (255, 255, 255)  # White (BGR)
                
                cv2.imwrite(str(debug_dir / f"frame_{frame_number:03d}_04a_edges_full_image.png"), edges_vis)
                print(f"[DEBUG] {frame_id_str} - Step 5: Saved full image edges with keypoints to {debug_dir / f'frame_{frame_number:03d}_04a_edges_full_image.png'}")
        
        mask_lines_predicted = extract_mask_of_ground_lines_in_image(
            image=frame, ground_mask=mask_ground_bin
        )
        
        predicted_lines_pixels = int(mask_lines_predicted.sum())
        
        if DEBUG_FLAG:
            print(f"[DEBUG] {frame_id_str} - Step 5: Predicted lines pixels (after ground mask)={predicted_lines_pixels}")
        
        if DEBUG_FLAG and debug_dir is not None and frame_number is not None:
            cv2.imwrite(str(debug_dir / f"frame_{frame_number:03d}_04_mask_lines_predicted.png"), mask_lines_predicted * 255)
            print(f"[DEBUG] {frame_id_str} - Step 5: Saved predicted lines mask (masked by ground)")
        
        # Step 6: Calculate overlap and perform additional validations
        if DEBUG_FLAG:
            print(f"[DEBUG] {frame_id_str} - Step 6: Calculating overlap between expected and predicted lines")
        
        pixels_overlapping_result = cv2.bitwise_and(
            mask_lines_expected, mask_lines_predicted
        )
        
        # Check bounding box area of expected lines (must cover >= 20% of frame)
        ys, xs = np.where(mask_lines_expected == 1)
        
        if len(xs) == 0:
            bbox = None
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 6: No expected lines found, returning 0.0")
            return 0.0
        else:
            min_x = xs.min()
            max_x = xs.max()
            min_y = ys.min()
            max_y = ys.max()
            bbox = (min_x, min_y, max_x, max_y)
        
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox is not None else 1
        frame_area = frame_height * frame_width
        
        if DEBUG_FLAG:
            bbox_ratio = bbox_area / frame_area if frame_area > 0 else 0.0
            print(f"[DEBUG] {frame_id_str} - Step 6: Bounding box area ratio={bbox_ratio:.4f} (bbox_area={bbox_area}, frame_area={frame_area})")
        
        if (bbox_area / frame_area) < 0.2:
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 6: Bounding box area too small ({bbox_area / frame_area:.4f} < 0.2), returning 0.0")
            return 0.0
        
        # Check valid keypoints existence
        valid_keypoints = [
            (x, y) for x, y in frame_keypoints
            if not (x == 0 and y == 0)
        ]
        if not valid_keypoints:
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 6: No valid keypoints found, returning 0.0")
            return 0.0
        
        # Check keypoint bounds and spread
        xs, ys = zip(*valid_keypoints)
        min_x_kp, max_x_kp = min(xs), max(xs)
        min_y_kp, max_y_kp = min(ys), max(ys)
        
        if max_x_kp < 0 or max_y_kp < 0 or min_x_kp >= frame_width or min_y_kp >= frame_height:
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 6: All keypoints are outside the frame, returning 0.0")
            return 0.0
        
        if (max_x_kp - min_x_kp) > 2 * frame_width or (max_y_kp - min_y_kp) > 2 * frame_height:
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 6: Keypoints spread too wide (width={max_x_kp - min_x_kp}, height={max_y_kp - min_y_kp}), returning 0.0")
            return 0.0
        
        # Check predicted lines outside expected lines (max 90%)
        inv_expected = cv2.bitwise_not(mask_lines_expected)
        pixels_rest = cv2.bitwise_and(inv_expected, mask_lines_predicted).sum()
        total_pixels = cv2.bitwise_or(mask_lines_expected, mask_lines_predicted).sum()
        
        if DEBUG_FLAG:
            print(f"[DEBUG] {frame_id_str} - Step 6: Predicted lines outside expected: {pixels_rest} pixels, Total pixels: {total_pixels}")
        
        if total_pixels == 0:
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 6: No total pixels found, returning 0.0")
            return 0.0
        
        if (pixels_rest / total_pixels) > 0.9:
            if DEBUG_FLAG:
                print(f"[DEBUG] {frame_id_str} - Step 6: Too many predicted lines outside expected ({pixels_rest / total_pixels:.4f} > 0.9), returning 0.0")
            return 0.0
        
        # Calculate final score
        pixels_overlapping = pixels_overlapping_result.sum()
        pixels_on_lines = mask_lines_expected.sum()
        overlap_ratio = float(pixels_overlapping) / float(pixels_on_lines + 1e-8)
        
        if DEBUG_FLAG:
            print(f"[DEBUG] {frame_id_str} - Step 6: Overlap pixels={pixels_overlapping}, Expected lines pixels={pixels_on_lines}, Ratio={overlap_ratio:.4f}")
        
        # Step 7: Create visualization
        if DEBUG_FLAG and debug_dir is not None and frame_number is not None:
            # Overlap visualization (expected in red on top layer, predicted in green)
            overlap_vis = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            # Predicted lines only: green (BGR: 0, 255, 0) - draw first
            predicted_only = (mask_lines_predicted > 0) & (mask_lines_expected == 0)
            overlap_vis[predicted_only, 1] = 255
            # Expected lines: red (BGR: 0, 0, 255) - draw on top layer
            expected_mask = (mask_lines_expected > 0)
            overlap_vis[expected_mask, 2] = 255
            cv2.imwrite(str(debug_dir / f"frame_{frame_number:03d}_05_overlap_visualization.png"), overlap_vis)
            
            # Save original frame with keypoints overlaid
            frame_with_keypoints = frame.copy()
            for idx, kp in enumerate(frame_keypoints):
                if kp and len(kp) >= 2 and not (kp[0] == 0 and kp[1] == 0):
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame_with_keypoints, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame_with_keypoints,
                        str(idx),
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
            cv2.imwrite(str(debug_dir / f"frame_{frame_number:03d}_00_original_with_keypoints.png"), frame_with_keypoints)
            print(f"[DEBUG] {frame_id_str} - Step 7: Saved overlap visualization and original frame with keypoints")
        
        if DEBUG_FLAG:
            print(f"[DEBUG] {frame_id_str} - Evaluation complete: score={overlap_ratio:.4f}")
        
        return overlap_ratio
    except Exception as e:
        if DEBUG_FLAG:
            print(f"[DEBUG] {frame_id_str if 'frame_id_str' in locals() else 'Frame'} - ERROR: Exception in evaluate_keypoints_for_frame: {e}")
        _log_error("evaluate_keypoints_for_frame failed", frame_number, log_frame_number=log_frame_number)
        
        # Try to save warped_template even on error if we have it
        if 'warped_template' in locals() and warped_template is not None and frame_number is not None:
            # Use debug_dir if provided, otherwise use default error output directory
            output_dir = debug_dir if 'debug_dir' in locals() else None
            if output_dir is None:
                output_dir = Path("error_outputs")
            
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_dir / f"frame_{frame_number:03d}_01_warped_template_EXCEPTION.png"), warped_template)
                if DEBUG_FLAG:
                    print(f"[DEBUG] {frame_id_str if 'frame_id_str' in locals() else 'Frame'} - Saved warped template (on exception) to {output_dir / f'frame_{frame_number:03d}_01_warped_template_EXCEPTION.png'}")
            except Exception as save_err:
                if DEBUG_FLAG:
                    print(f"[DEBUG] {frame_id_str if 'frame_id_str' in locals() else 'Frame'} - Failed to save warped template: {save_err}")
        
        return 0.0


def _get_warped_template_before_validation(
    image: np.ndarray,
    source_keypoints: list[tuple[int, int]],
    destination_keypoints: list[tuple[int, int]],
    destination_width: int,
    destination_height: int,
    inverse: bool = False,
) -> np.ndarray | None:
    """
    Get warped template without validation (for debugging/error cases).
    Returns None if warping fails.
    """
    try:
        filtered_src: list[tuple[int, int]] = []
        filtered_dst: list[tuple[int, int]] = []
        
        for src_pt, dst_pt in zip(source_keypoints, destination_keypoints, strict=True):
            if dst_pt[0] == 0.0 and dst_pt[1] == 0.0:  # ignore default / missing points
                continue
            filtered_src.append(src_pt)
            filtered_dst.append(dst_pt)
        
        if len(filtered_src) < 4:
            return None
        
        source_points = np.array(filtered_src, dtype=np.float32)
        destination_points = np.array(filtered_dst, dtype=np.float32)
        
        if inverse:
            H_inv, _ = cv2.findHomography(destination_points, source_points)
            if H_inv is None:
                return None
            return cv2.warpPerspective(image, H_inv, (destination_width, destination_height))
        
        H, _ = cv2.findHomography(source_points, destination_points)
        if H is None:
            return None
        
        projected_image = cv2.warpPerspective(image, H, (destination_width, destination_height))
        return projected_image
    except Exception:
        return None


def project_image_using_keypoints(
    image: np.ndarray,
    source_keypoints: list[tuple[int, int]],
    destination_keypoints: list[tuple[int, int]],
    destination_width: int,
    destination_height: int,
    inverse: bool = False,
) -> np.ndarray:
    """
    Project image using keypoints with validation.
    Matches original validator implementation exactly.
    """
    filtered_src: list[tuple[int, int]] = []
    filtered_dst: list[tuple[int, int]] = []
    
    # Use strict=True to match original (requires exact length match)
    for src_pt, dst_pt in zip(source_keypoints, destination_keypoints, strict=True):
        if dst_pt[0] == 0.0 and dst_pt[1] == 0.0:  # ignore default / missing points
            continue
        filtered_src.append(src_pt)
        filtered_dst.append(dst_pt)
    
    if len(filtered_src) < 4:
        raise ValueError("At least 4 valid keypoints are required for homography.")
    
    source_points = np.array(filtered_src, dtype=np.float32)
    destination_points = np.array(filtered_dst, dtype=np.float32)
    
    if inverse:
        H_inv, _ = cv2.findHomography(destination_points, source_points)
        return cv2.warpPerspective(image, H_inv, (destination_width, destination_height))
    
    H, _ = cv2.findHomography(source_points, destination_points)
    if H is None:
        raise ValueError("Homography computation failed")
    
    projected_image = cv2.warpPerspective(image, H, (destination_width, destination_height))
    # Validate corners AFTER warping (matches original)
    # Note: projected_image is available even if validation fails
    try:
        validate_projected_corners(source_keypoints=source_keypoints, homography_matrix=H)
    except InvalidMask:
        # Re-raise but projected_image is already computed
        raise
    return projected_image


def extract_masks_for_ground_and_lines(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ground and line masks with validation.
    Matches original validator implementation exactly.
    Assumes template colored s.t. ground = gray, lines = white, background = black
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_ground = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    _, mask_lines = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask_ground_binary = (mask_ground > 0).astype(np.uint8)
    mask_lines_binary = (mask_lines > 0).astype(np.uint8)
    
    # Validate mask_ground_binary: check for empty mask and perfect rectangle
    if cv2.countNonZero(mask_ground_binary) == 0:
        raise InvalidMask("No projected ground (empty mask)")
    pts = cv2.findNonZero(mask_ground_binary)
    if pts is None or len(pts) == 0:
        raise InvalidMask("No projected ground (empty mask)")
    x, y, w, h = cv2.boundingRect(pts)
    is_rect = cv2.countNonZero(mask_ground_binary) == (w * h)
    
    if is_rect:
        raise InvalidMask(
            f"Projected ground should not be rectangular"
        )
    
    # Validate masks (raises InvalidMask if invalid)
    validate_mask_ground(mask=mask_ground_binary)
    validate_mask_lines(mask=mask_lines_binary)
    
    return mask_ground_binary, mask_lines_binary


# --------------------------- Scoring logic --------------------------- #
def _draw_keypoints_overlay(
    frame: Any,
    kps: list[list[float]],
    *,
    score_text: str | None = None,
    error_text: str | None = None,
) -> Any:
    img = frame.copy()
    for idx, pt in enumerate(kps):
        if not pt or len(pt) < 2:
            continue
        if pt[0] == 0 and pt[1] == 0:
            continue
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            img,
            str(idx + 1),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    if score_text:
        cv2.putText(
            img,
            score_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if error_text:
        cv2.putText(
            img,
            error_text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def _keypoint_scores(
    miner_predictions: dict[int, dict],
    frame_store,
    *,
    log_frame_number: bool,
    save_dir: Path | None = None,
    file_name_suffix: str = "local",
) -> tuple[float, list[dict[str, float]]]:
    template_image = challenge_template()
    template_keypoints = KEYPOINTS
    scores: list[float] = []
    per_frame: list[dict[str, float]] = []
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create debug directory if DEBUG_FLAG is True
    debug_dir: Path | None = None
    if DEBUG_FLAG and save_dir:
        debug_dir = save_dir.parent / f"{save_dir.name}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Debug directory created: {debug_dir}")
    
    frame_ids = sorted(miner_predictions.keys())
    
    # Optional: process only specific frames if ONLY_FRAMES is defined
    only_frames = globals().get('ONLY_FRAMES')
    if only_frames:
        frame_ids = [fid for fid in frame_ids if fid in only_frames]
    
    total = len(frame_ids)
    for idx, fn in enumerate(frame_ids, start=1):
        print(f"\rKeypoints {idx}/{total} (frame {fn})", end="", flush=True)
        ann = miner_predictions[fn] or {}
        kps = ann.get("keypoints") or []
        frame_image = None
        try:
            frame_image = frame_store.get_frame(fn)
        except Exception:
            frame_image = None

        score = 0.0
        err_msg: str | None = None
        if frame_image is None:
            _log_error("Frame image missing", fn, log_frame_number=log_frame_number)
            err_msg = "Frame missing"
        elif len(kps) < 4:
            _log_error(
                "At least 4 valid keypoints are required for homography.",
                fn,
                log_frame_number=log_frame_number,
            )
            err_msg = "Need >=4 keypoints"
        elif len(kps) != len(template_keypoints):
            _log_error(
                f"Keypoint count mismatch (expected {len(template_keypoints)}, got {len(kps)})",
                fn,
                log_frame_number=log_frame_number,
            )
            err_msg = "Count mismatch"
        else:
            score = evaluate_keypoints_for_frame(
                template_keypoints=template_keypoints,
                frame_keypoints=kps,
                frame=frame_image,
                floor_markings_template=template_image.copy(),
                frame_number=fn,
                log_frame_number=log_frame_number,
                debug_dir=debug_dir,
            )
        if save_dir:
            score_text = f"KP score: {score:.4f}" if err_msg is None else None
            err_text = err_msg
            vis_frame = frame_image
            if vis_frame is None:
                vis_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                try:
                    warped = project_image_using_keypoints(
                        image=template_image.copy(),
                        source_keypoints=template_keypoints,
                        destination_keypoints=kps,
                        destination_width=vis_frame.shape[1],
                        destination_height=vis_frame.shape[0],
                    )
                    vis_frame = cv2.addWeighted(vis_frame, 0.7, warped, 0.3, 0)
                except Exception:
                    pass
            vis = _draw_keypoints_overlay(
                vis_frame,
                kps,
                score_text=score_text,
                error_text=err_text,
            )
            out_path = save_dir / f"{fn:03d}-{file_name_suffix}-{_format_score_for_filename(score)}.jpg"
            cv2.imwrite(str(out_path), vis)
        scores.append(score)
        per_frame.append({"frame": fn, "score": score})
    print("\rKeypoints processing done.       ")
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    return avg_score, per_frame


class _CountHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, dict[str, Any]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        entry = self.counts.get(msg) or {"count": 0, "frames": []}
        entry["count"] += 1
        fn = getattr(record, "frame_number", None)
        if fn is not None and len(entry["frames"]) < 3:
            entry["frames"].append(fn)
        self.counts[msg] = entry


def _load_miner_run(path: Path) -> SVRunOutput:
    data: dict[str, Any] = json.loads(path.read_text())
    predicted_frames: list[dict[str, Any]] = []
    frames_in = data.get("frames") or (data.get("predictions") or {}).get("frames") or []
    for frame in frames_in:
        frame_id = frame.get("frame_number")
        if frame_id is None:
            frame_id = frame.get("frame_id")
        if frame_id is None:
            continue
        predicted_frames.append(
            {
                "frame_id": int(frame_id),
                "keypoints": frame.get("keypoints") or [],
                "action": frame.get("action"),
            }
        )
    predictions = {"frames": predicted_frames}
    return SVRunOutput(
        success=bool(data.get("success", True)),
        latency_ms=float(data.get("latency_ms", 0.0)),
        predictions=predictions,
        error=data.get("error"),
        model=data.get("model"),
    )


def _file_name_suffix_from_miner_json(miner_json: Path) -> str:
    """Derive file name suffix 's': temp_test/<name>/... -> name, else 'test'."""
    resolved = miner_json.resolve()
    parts = resolved.parts
    if "temp_test" in parts:
        return miner_json.parent.name
    return "opt"


def calculate_keypoints(video_id: str, video_url: str, miner_json: Path) -> dict[str, Any]:
    miner_run = _load_miner_run(miner_json)
    miner_predictions = parse_miner_prediction(miner_run=miner_run)
    logging.getLogger("keypoints_calculate_score").info(
        "Parsed miner predictions for %d frames", len(miner_predictions)
    )

    file_name_suffix = _file_name_suffix_from_miner_json(miner_json)
    keypoints_score = 0.0
    kp_error_counts: dict[str, int] = {}
    kp_per_frame: list[dict[str, float]] = []
    tmp_path, frame_store = download_video_cached(video_url, _frame_numbers=[])
    try:
        kp_logger = logging.getLogger("scorevision.vlm_pipeline.non_vlm_scoring.keypoints")
        handler = _CountHandler()
        kp_logger.addHandler(handler)
        prev_level = kp_logger.level
        kp_logger.setLevel(logging.ERROR)
        try:
            save_dir = Path("scoring_result") / video_id
            keypoints_score, kp_per_frame = _keypoint_scores(
                miner_predictions=miner_predictions,
                frame_store=frame_store,
                log_frame_number=True,
                save_dir=save_dir,
                file_name_suffix=file_name_suffix,
            )
        finally:
            kp_logger.setLevel(prev_level)
            kp_logger.removeHandler(handler)
            kp_error_counts = handler.counts
    finally:
        frame_store.unlink()
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    return {
        "video_id": video_id,
        "overall": {
            "keypoints_score": keypoints_score,
            "keypoints_errors": kp_error_counts,
        },
        "keypoints_per_frame": kp_per_frame,
    }


# --------------------------- CLI / Main ------------------------------ #
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate keypoint scores only (no object scoring)."
    )
    parser.add_argument(
        "--video-id",
        required=False,
        help="Video ID (used for output paths). If omitted, derived from video URL basename.",
    )
    parser.add_argument(
        "--video-url",
        required=True,
        help="Video URL or local path.",
    )
    parser.add_argument(
        "--miner-json",
        required=True,
        type=Path,
        help="Path to miner response JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed scoring logs",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    def _derive_video_id(url_or_path: str, explicit: str | None) -> str:
        if explicit:
            return explicit
        from urllib.parse import urlparse

        p = urlparse(url_or_path)
        name = Path(p.path).name if p.scheme else Path(url_or_path).name
        stem = Path(name).stem
        if not stem:
            return ""
        return stem

    video_id = _derive_video_id(args.video_url, args.video_id)
    if not video_id:
        video_id = "video"

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = calculate_keypoints(
        video_id=video_id,
        video_url=args.video_url,
        miner_json=args.miner_json,
    )

    overall = result.get("overall") or {}
    keypoints_score = float(overall.get("keypoints_score", 0.0))
    keypoints_errors = overall.get("keypoints_errors") or {}
    print(f"Overall keypoint score: {keypoints_score:.4f}")
    if keypoints_errors:
        print("Failed keypoints/errors:")
        for err_msg, info in keypoints_errors.items():
            count = info.get("count", 0)
            frames = info.get("frames") or []
            frames_str = ", ".join(str(f) for f in frames) if frames else "n/a"
            print(f" - {err_msg} | count: {count}, sample frames: {frames_str}")
    else:
        print("Failed keypoints/errors: none")

    # Persist per-frame keypoint scores
    kp_frames = result.get("keypoints_per_frame") or []
    out_dir = Path("keypoint_scores")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.miner_json.stem}.json"
    payload = {
        "video_id": video_id,
        "keypoint_scores": kp_frames,
        "avg_keypoints_score": result.get("overall", {}).get("keypoints_score", 0.0),
        "keypoints_errors": result.get("overall", {}).get("keypoints_errors", {}),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote keypoint scores to {out_path}")

    # Stitch scoring_result/<video_id> frames into a 25fps mp4 named <video_id>.mp4
    # Skip video creation when DEBUG_FLAG is True
    if not DEBUG_FLAG:
        frames_dir = Path("scoring_result") / video_id
        if frames_dir.exists():
            frame_files = sorted(
                [str(p) for p in frames_dir.glob("*.png")] + [str(p) for p in frames_dir.glob("*.jpg")]
            )
            if frame_files:
                first = cv2.imread(frame_files[0])
                if first is not None:
                    h, w = first.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_path = frames_dir / f"{video_id}.mp4"
                    out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (w, h))
                    for fp in frame_files:
                        img = cv2.imread(fp)
                        if img is None:
                            continue
                        if img.shape[:2] != (h, w):
                            img = cv2.resize(img, (w, h))
                        out.write(img)
                    out.release()
                    logging.info("Wrote scoring video to %s", video_path)


if __name__ == "__main__":
    main()

