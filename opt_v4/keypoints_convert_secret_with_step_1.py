from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Number of threads for parallel Step 5 and Step 8 (1 = sequential).
NUM_THREADS: int = 8

STEP6_FILL_MISSING_ENABLED = True
# If False, skip Step 7 interpolation.
STEP7_ENABLED = True
# Each pass interpolates frames below the threshold using neighbours above it.
# Passes run in order, so later passes with higher thresholds can fix more frames
# using the keypoints that were already updated by earlier passes.
STEP7_SCORE_THRESHOLDS: list[float] = [0.3, 0.5]
# Max frame count between backward and forward good frames; if gap > this, skip.
STEP7_MAX_GAP: int = 20
# If False, skip Step 8 keypoint adjustment.
STEP8_ENABLED = True
# Optional: process only specific frames. Keep empty to process all.
# ONLY_FRAMES: list[int] = list(range(43, 44))
# If True, write step-by-step scoring debug images.
DEBUG_FLAG = False

# Step 5: weighted homography schemes (indices into keypoints / FOOTBALL_KEYPOINTS_CORRECTED)
STEP5_H1_WEIGHT_2_INDICES: list[int] = [4, 9, 10, 11, 12, 17, 18, 19, 20, 28]
STEP5_H2_WEIGHT_3_INDICES: list[int] = [13, 14, 15]
STEP5_H2_WEIGHT_4_INDICES: list[int] = [5, 16, 29]
STEP5_H3_WEIGHT_2_INDICES: list[int] = [4, 9, 10, 11, 12, 17, 18, 19, 20, 28]
STEP5_H3_WEIGHT_3_INDICES: list[int] = [13, 14, 15]
STEP5_H3_WEIGHT_4_INDICES: list[int] = [5, 16, 29]

FOOTBALL_KEYPOINTS: list[tuple[int, int]] = [
    (5, 5),
    (5, 140),
    (5, 250),
    (5, 430),
    (5, 540),
    (5, 675),
    (55, 250),
    (55, 430),
    (110, 340),
    (165, 140),
    (165, 270),
    (165, 410),
    (165, 540),
    (527, 5),
    (527, 253),
    (527, 433),
    (527, 675),
    (888, 140),
    (888, 270),
    (888, 410),
    (888, 540),
    (940, 340),
    (998, 250),
    (998, 430),
    (1045, 5),
    (1045, 140),
    (1045, 250),
    (1045, 430),
    (1045, 540),
    (1045, 675),
    (435, 340),
    (615, 340),
]

FOOTBALL_KEYPOINTS_CORRECTED: list[tuple[float, float]] = [
    (2.5, 2.5),
    (2.5, 139.5),
    (2.5, 249.5),
    (2.5, 430.5),
    (2.5, 540.5),
    (2.5, 678.0),
    (54.5, 249.5),
    (54.5, 430.5),
    (110.5, 340.5),
    (164.5, 139.5),
    (164.5, 269.0),
    (164.5, 411.0),
    (164.5, 540.5),
    (525.0, 2.5),
    (525.0, 249.5),
    (525.0, 430.5),
    (525.0, 678.0),
    (886.5, 139.5),
    (886.5, 269.0),
    (886.5, 411.0),
    (886.5, 540.5),
    (940.5, 340.5),
    (998.0, 249.5),
    (998.0, 430.5),
    (1048.0, 2.5),
    (1048.0, 139.5),
    (1048.0, 249.5),
    (1048.0, 430.5),
    (1048.0, 540.5),
    (1048.0, 678.0),
    (434.5, 340.0),
    (615.5, 340.0),
]

INDEX_KEYPOINT_CORNER_BOTTOM_LEFT = 5
INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT = 29
INDEX_KEYPOINT_CORNER_TOP_LEFT = 0
INDEX_KEYPOINT_CORNER_TOP_RIGHT = 24


def _get_all_template_points() -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in FOOTBALL_KEYPOINTS]


def _extract_frames_container(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(raw, list):
        wrapped = {"frames": raw}
        return wrapped, raw

    if not isinstance(raw, dict):
        wrapped = {"frames": []}
        return wrapped, wrapped["frames"]

    frames = raw.get("frames")
    if isinstance(frames, list):
        return raw, frames

    predictions = raw.get("predictions")
    if isinstance(predictions, dict):
        pred_frames = predictions.get("frames")
        if isinstance(pred_frames, list):
            return raw, pred_frames
        predictions["frames"] = []
        return raw, predictions["frames"]

    raw["frames"] = []
    return raw, raw["frames"]


def _get_video_dims(video_url: str) -> tuple[int | None, int | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        video_url,
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return None, None
    if "x" not in out:
        return None, None
    w_raw, h_raw = out.split("x", 1)
    try:
        width = int(w_raw.strip())
        height = int(h_raw.strip())
    except ValueError:
        return None, None
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(a)
    aug = [row[:] + [b[idx]] for idx, row in enumerate(a)]

    for col in range(n):
        pivot = col
        max_abs = abs(aug[pivot][col])
        for r in range(col + 1, n):
            cur = abs(aug[r][col])
            if cur > max_abs:
                max_abs = cur
                pivot = r
        if max_abs < 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        for c in range(col, n + 1):
            aug[col][c] /= pivot_val

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-15:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]

    return [aug[i][n] for i in range(n)]


def _mat_t_mul_mat(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a[0]) if a else 0
    cols = len(b[0]) if b else 0
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            s = 0.0
            for k in range(len(a)):
                s += a[k][i] * b[k][j]
            out[i][j] = s
    return out


def _mat_t_mul_vec(a: list[list[float]], v: list[float]) -> list[float]:
    rows = len(a[0]) if a else 0
    out = [0.0 for _ in range(rows)]
    for i in range(rows):
        s = 0.0
        for k in range(len(a)):
            s += a[k][i] * v[k]
        out[i] = s
    return out


def _find_homography_from_points(
    src_points: list[tuple[float, float]], dst_points: list[tuple[float, float]]
) -> list[list[float]] | None:
    n_pts = min(len(src_points), len(dst_points))
    if n_pts < 4:
        return None

    a_rows: list[list[float]] = []
    b_vals: list[float] = []
    for i in range(n_pts):
        x, y = src_points[i]
        u, v = dst_points[i]
        a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        b_vals.append(u)
        a_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b_vals.append(v)

    # Solve least-squares over all valid correspondences:
    # (A^T A) h = A^T b
    ata = _mat_t_mul_mat(a_rows, a_rows)
    atb = _mat_t_mul_vec(a_rows, b_vals)
    sol = _solve_linear_system(ata, atb)
    if sol is None:
        return None
    h11, h12, h13, h21, h22, h23, h31, h32 = sol
    return [[h11, h12, h13], [h21, h22, h23], [h31, h32, 1.0]]


def _perspective_transform(points: list[tuple[float, float]], h_mat: list[list[float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in points:
        denom = h_mat[2][0] * x + h_mat[2][1] * y + h_mat[2][2]
        if abs(denom) < 1e-12:
            out.append((0.0, 0.0))
            continue
        px = (h_mat[0][0] * x + h_mat[0][1] * y + h_mat[0][2]) / denom
        py = (h_mat[1][0] * x + h_mat[1][1] * y + h_mat[1][2]) / denom
        out.append((float(px), float(py)))
    return out


def _infer_dims_from_kps(kps: list[Any]) -> tuple[int | None, int | None]:
    max_x = 0.0
    max_y = 0.0
    has_any = False
    for kp in kps:
        if not isinstance(kp, list) or len(kp) < 2:
            continue
        x = float(kp[0])
        y = float(kp[1])
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            continue
        has_any = True
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    if not has_any:
        return None, None
    return int(max_x + 2.0), int(max_y + 2.0)


def _is_valid_kp(kp: Any) -> bool:
    if not isinstance(kp, (list, tuple)) or len(kp) < 2:
        return False
    x, y = float(kp[0]), float(kp[1])
    return not (x == 0.0 and y == 0.0)


def _clip_segment_to_frame(
    ax: float, ay: float, bx: float, by: float, w: int, h: int
) -> tuple[float, float] | None:
    """Return the point where segment A->B crosses the frame [0,w]x[0,h], or None if B is inside."""
    if 0 <= bx < w and 0 <= by < h:
        return None
    dx = bx - ax
    dy = by - ay
    best_t: float | None = None
    if abs(dx) > 1e-12:
        t = -ax / dx
        py = ay + t * dy
        if 0.0 < t <= 1.0 and 0 <= py <= h:
            if best_t is None or t < best_t:
                best_t = t
        t = (w - ax) / dx
        py = ay + t * dy
        if 0.0 < t <= 1.0 and 0 <= py <= h:
            if best_t is None or t < best_t:
                best_t = t
    if abs(dy) > 1e-12:
        t = -ay / dy
        px = ax + t * dx
        if 0.0 < t <= 1.0 and 0 <= px <= w:
            if best_t is None or t < best_t:
                best_t = t
        t = (h - ay) / dy
        px = ax + t * dx
        if 0.0 < t <= 1.0 and 0 <= px <= w:
            if best_t is None or t < best_t:
                best_t = t
    if best_t is None:
        return None
    return (float(ax + best_t * dx), float(ay + best_t * dy))


def _run_step1_infer_center_circle(
    frames: list[dict[str, Any]],
    default_width: int | None,
    default_height: int | None,
) -> int:
    """
    Step 1: Infer missing center-circle keypoint (kp[30] or kp[31]) from the other center
    point and the center of kp[14], kp[15], when only 14,15 and one of 30/31 are valid.
    """
    updated = 0
    indices_0_12 = list(range(0, 13))
    indices_17_29 = list(range(17, 30))
    for fe in frames:
        if not isinstance(fe, dict):
            continue
        kps = fe.get("keypoints")
        if not isinstance(kps, list) or len(kps) < 32:
            continue
        frame_width = fe.get("frame_width")
        frame_height = fe.get("frame_height")
        if frame_width is not None and frame_height is not None:
            w, h = int(frame_width), int(frame_height)
        else:
            w, h = default_width, default_height
        if w is None or h is None or w <= 0 or h <= 0:
            w, h = _infer_dims_from_kps(kps)
        if w is None or h is None:
            continue
        valid_14 = _is_valid_kp(kps[14])
        valid_15 = _is_valid_kp(kps[15])
        valid_30 = _is_valid_kp(kps[30])
        valid_31 = _is_valid_kp(kps[31])
        others_invalid = all(
            not _is_valid_kp(kps[i]) for i in indices_0_12 + indices_17_29
        )
        frame_id = fe.get("frame_id", fe.get("frame_number", -1))

        if valid_14 and valid_15 and valid_30 and not valid_31 and others_invalid:
            ax = (float(kps[14][0]) + float(kps[15][0])) / 2.0
            ay = (float(kps[14][1]) + float(kps[15][1])) / 2.0
            x30 = float(kps[30][0])
            y30 = float(kps[30][1])
            bx = 2.0 * ax - x30
            by = 2.0 * ay - y30
            if 0 <= bx < w and 0 <= by < h:
                kps[31] = [bx, by]
                updated += 1
                if DEBUG_FLAG:
                    print(f"Step 1 frame {frame_id}: inferred kp[31]={[bx, by]} (B inside frame)")
            else:
                clipped = _clip_segment_to_frame(ax, ay, bx, by, w, h)
                if clipped is not None:
                    bx, by = clipped
                    kps[31] = [bx, by]
                    cx = 2.0 * ax - bx
                    cy = 2.0 * ay - by
                    kps[30] = [cx, cy]
                    updated += 1
                    if DEBUG_FLAG:
                        print(f"Step 1 frame {frame_id}: B outside frame -> kp[31]={[bx, by]}, kp[30]={[cx, cy]}")
                else:
                    if DEBUG_FLAG:
                        print(f"Step 1 frame {frame_id}: B outside frame but no clip point found, skipped")
            continue

        if valid_14 and valid_15 and valid_31 and not valid_30 and others_invalid:
            ax = (float(kps[14][0]) + float(kps[15][0])) / 2.0
            ay = (float(kps[14][1]) + float(kps[15][1])) / 2.0
            x31 = float(kps[31][0])
            y31 = float(kps[31][1])
            bx = 2.0 * ax - x31
            by = 2.0 * ay - y31
            if 0 <= bx < w and 0 <= by < h:
                kps[30] = [bx, by]
                updated += 1
                if DEBUG_FLAG:
                    print(f"Step 1 frame {frame_id}: inferred kp[30]={[bx, by]} (B inside frame)")
            else:
                clipped = _clip_segment_to_frame(ax, ay, bx, by, w, h)
                if clipped is not None:
                    bx, by = clipped
                    kps[30] = [bx, by]
                    cx = 2.0 * ax - bx
                    cy = 2.0 * ay - by
                    kps[31] = [cx, cy]
                    updated += 1
                    if DEBUG_FLAG:
                        print(f"Step 1 frame {frame_id}: B outside frame -> kp[30]={[bx, by]}, kp[31]={[cx, cy]}")
                else:
                    if DEBUG_FLAG:
                        print(f"Step 1 frame {frame_id}: B outside frame but no clip point found, skipped")
    return updated


class _InvalidMask(Exception):
    pass


def _challenge_template() -> np.ndarray:
    template_path = Path(__file__).resolve().parent / "football_pitch_template.png"
    img = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    return img


def _has_a_wide_line(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        _, _, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        if min(w, h) / max(w, h) >= 1.0:
            return True
    return False


def _validate_masks(ground_mask: np.ndarray, line_mask: np.ndarray) -> None:
    if ground_mask.sum() == 0:
        raise _InvalidMask("No projected ground (empty mask)")
    pts = cv2.findNonZero(ground_mask)
    if pts is None:
        raise _InvalidMask("No projected ground (empty mask)")
    _, _, w, h = cv2.boundingRect(pts)
    if cv2.countNonZero(ground_mask) == w * h:
        raise _InvalidMask("Projected ground should not be rectangular")
    n_labels, _ = cv2.connectedComponents(ground_mask)
    if n_labels - 1 > 1:
        raise _InvalidMask("Projected ground should be a single object")
    if ground_mask.sum() / ground_mask.size >= 0.9:
        raise _InvalidMask("Projected ground covers too much of the image")
    if line_mask.sum() == 0:
        raise _InvalidMask("No projected lines")
    if line_mask.sum() == line_mask.size:
        raise _InvalidMask("Projected lines cover the entire image")
    if _has_a_wide_line(line_mask):
        raise _InvalidMask("A projected line is too wide")


def _is_bowtie(pts: np.ndarray) -> bool:
    def _ccw(a: tuple, b: tuple, c: tuple) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def _intersect(p1: tuple, p2: tuple, q1: tuple, q2: tuple) -> bool:
        return (_ccw(p1, q1, q2) != _ccw(p2, q1, q2)) and (_ccw(p1, p2, q1) != _ccw(p1, p2, q2))

    p = pts.reshape(-1, 2)
    if len(p) < 4:
        return False
    edges = [(p[0], p[1]), (p[1], p[2]), (p[2], p[3]), (p[3], p[0])]
    return _intersect(*edges[0], *edges[2]) or _intersect(*edges[1], *edges[3])


def _warp_template(
    template: np.ndarray,
    src_kps: list[tuple[float, float]],
    dst_kps: list[tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    src = np.array(src_kps, dtype=np.float32)
    dst = np.array(dst_kps, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise ValueError("Homography computation failed")
    warped = cv2.warpPerspective(template, H, (frame_width, frame_height))
    # Bowtie check on projected corners
    corner_indices = [
        INDEX_KEYPOINT_CORNER_BOTTOM_LEFT,
        INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT,
        INDEX_KEYPOINT_CORNER_TOP_RIGHT,
        INDEX_KEYPOINT_CORNER_TOP_LEFT,
    ]
    if len(src_kps) > max(corner_indices):
        src_corners = np.array(
            [[src_kps[i][0], src_kps[i][1]] for i in corner_indices],
            dtype=np.float32,
        ).reshape(1, 4, 2)
        proj_corners = cv2.perspectiveTransform(src_corners, H)[0]
        if _is_bowtie(proj_corners):
            raise _InvalidMask("Projection twisted!")
    return warped


def _extract_masks(warped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, m_ground = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    _, m_lines = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    ground_bin = (m_ground > 0).astype(np.uint8)
    lines_bin = (m_lines > 0).astype(np.uint8)
    _validate_masks(ground_bin, lines_bin)
    return ground_bin, lines_bin


def _predicted_lines_mask(frame: np.ndarray, ground_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    edges_on_ground = cv2.bitwise_and(edges, edges, mask=ground_mask)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_on_ground = cv2.dilate(edges_on_ground, dilate_kernel, iterations=3)
    return (edges_on_ground > 0).astype(np.uint8)


def _write_scoring_debug_images(
    *,
    frame_id: int,
    out_dir: Path,
    video_frame: np.ndarray,
    warped_template: np.ndarray,
    ground_mask: np.ndarray,
    line_mask: np.ndarray,
    predicted_mask: np.ndarray,
    score: float,
) -> None:
    frame_dir = out_dir / f"frame_{frame_id:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    # Remove files from previous scoring implementations
    for old in (
        "01_observed_points.jpg", "02_projected_and_observed.jpg", "03_reprojection_error.jpg",
        "02_ground_mask.jpg", "03_line_mask.jpg", "04_edges_on_ground.jpg",
        "05_dilated_edges_on_ground.jpg",
    ):
        p = frame_dir / old
        if p.exists():
            p.unlink(missing_ok=True)

    cv2.imwrite(str(frame_dir / "01_original_frame.jpg"), video_frame)
    cv2.imwrite(str(frame_dir / "02_warped_template.jpg"), warped_template)
    cv2.imwrite(str(frame_dir / "03_ground_mask.jpg"), ground_mask * 255)
    cv2.imwrite(str(frame_dir / "04_line_mask_expected.jpg"), line_mask * 255)
    cv2.imwrite(str(frame_dir / "05_predicted_lines_dilated.jpg"), predicted_mask * 255)

    # Overlap vis: expected lines in red, overlap hits in green, on original frame
    vis = video_frame.copy()
    vis[line_mask == 1] = (0, 0, 180)
    overlap = cv2.bitwise_and(line_mask, predicted_mask)
    vis[overlap == 1] = (0, 220, 0)
    cv2.imwrite(str(frame_dir / "06_overlap_on_frame.jpg"), vis)

    meta = {
        "frame": frame_id,
        "score": score,
        "line_pixels": int(line_mask.sum()),
        "overlap_pixels": int(overlap.sum()),
        "debug_images": [
            "01_original_frame.jpg",
            "02_warped_template.jpg",
            "03_ground_mask.jpg",
            "04_line_mask_expected.jpg",
            "05_predicted_lines_dilated.jpg",
            "06_overlap_on_frame.jpg",
        ],
    }
    (frame_dir / "score_meta.json").write_text(json.dumps(meta, indent=2))


class _FrameStore:
    """Keeps a single VideoCapture open; avoids re-opening + re-seeking for every frame."""

    def __init__(self, source: str) -> None:
        self._cap = cv2.VideoCapture(source)
        self._last_id: int | None = None

    def get(self, frame_id: int) -> np.ndarray | None:
        # Sequential read: just advance without seeking
        if self._last_id is not None and frame_id == self._last_id + 1:
            ok, frame = self._cap.read()
        elif self._last_id is None and frame_id == 0:
            ok, frame = self._cap.read()
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._last_id = frame_id
        return frame

    def close(self) -> None:
        self._cap.release()



def _weight_map_h1() -> dict[int, int]:
    return {i: 2 for i in STEP5_H1_WEIGHT_2_INDICES}


def _weight_map_h2() -> dict[int, int]:
    m: dict[int, int] = {}
    for i in STEP5_H2_WEIGHT_3_INDICES:
        m[i] = 3
    for i in STEP5_H2_WEIGHT_4_INDICES:
        m[i] = 4
    return m


def _weight_map_h3() -> dict[int, int]:
    m: dict[int, int] = {}
    for i in STEP5_H3_WEIGHT_2_INDICES:
        m[i] = 2
    for i in STEP5_H3_WEIGHT_3_INDICES:
        m[i] = 3
    for i in STEP5_H3_WEIGHT_4_INDICES:
        m[i] = 4
    return m


def _find_homography_weighted(
    valid_indices: list[int],
    valid_src: list[tuple[float, float]],
    valid_dst: list[tuple[float, float]],
    weight_by_index: dict[int, int],
) -> np.ndarray | None:
    """Build weighted point arrays (repeat by weight) and compute H. Returns None on failure."""
    src_list: list[tuple[float, float]] = []
    dst_list: list[tuple[float, float]] = []
    for idx, (s, d) in zip(valid_indices, zip(valid_src, valid_dst)):
        w = max(1, weight_by_index.get(idx, 1))
        for _ in range(w):
            src_list.append(s)
            dst_list.append(d)
    if len(src_list) < 4:
        return None
    src_np = np.array(src_list, dtype=np.float32)
    dst_np = np.array(dst_list, dtype=np.float32)
    H, _ = cv2.findHomography(src_np, dst_np)
    return H


def _score_given_H(
    H: np.ndarray,
    template_image: np.ndarray,
    video_frame: np.ndarray,
) -> float:
    """Score one homography: warp template, masks, overlap ratio. Returns 0.0 on failure."""
    try:
        h, w = video_frame.shape[:2]
        warped = cv2.warpPerspective(template_image, H, (w, h))
        ground_mask, line_mask = _extract_masks(warped)
        predicted_mask = _predicted_lines_mask(video_frame, ground_mask)
        overlap = cv2.bitwise_and(line_mask, predicted_mask)
        pixels_on_lines = int(line_mask.sum())
        pixels_overlap = int(overlap.sum())
        return float(pixels_overlap) / float(pixels_on_lines + 1e-8)
    except Exception:
        return 0.0


def _step5_process_one_frame(
    frame_entry: dict[str, Any],
    frame_id: int,
    video_frame: np.ndarray | None,
    template_image: np.ndarray,
    debug_dir: Path | None,
) -> tuple[int, float, str | None]:
    """Process one frame for Step 5. Returns (frame_id, score, label). Mutates frame_entry['keypoints']."""
    if video_frame is None:
        return (frame_id, 0.0, None)
    kps = frame_entry.get("keypoints")
    if not isinstance(kps, list) or len(kps) != len(FOOTBALL_KEYPOINTS_CORRECTED):
        return (frame_id, 0.0, None)
    valid_indices = []
    valid_src = []
    valid_dst = []
    for idx, kp in enumerate(kps):
        if not isinstance(kp, list) or len(kp) < 2:
            continue
        x, y = float(kp[0]), float(kp[1])
        if x == 0.0 and y == 0.0:
            continue
        valid_indices.append(idx)
        valid_src.append(FOOTBALL_KEYPOINTS_CORRECTED[idx])
        valid_dst.append((x, y))
    if len(valid_src) < 4:
        return (frame_id, 0.0, None)
    frame_height, frame_width = video_frame.shape[:2]
    w1, w2, w3 = _weight_map_h1(), _weight_map_h2(), _weight_map_h3()
    H1 = _find_homography_weighted(valid_indices, valid_src, valid_dst, w1)
    H2 = _find_homography_weighted(valid_indices, valid_src, valid_dst, w2)
    H3 = _find_homography_weighted(valid_indices, valid_src, valid_dst, w3)
    score1 = _score_given_H(H1, template_image, video_frame) if H1 is not None else 0.0
    score2 = _score_given_H(H2, template_image, video_frame) if H2 is not None else 0.0
    score3 = _score_given_H(H3, template_image, video_frame) if H3 is not None else 0.0
    best_score, best_H, best_label = score1, H1, "H1"
    if score2 > best_score:
        best_score, best_H, best_label = score2, H2, "H2"
    if score3 > best_score:
        best_score, best_H, best_label = score3, H3, "H3"
    if best_H is None:
        return (frame_id, 0.0, None)
    src_all = np.array(FOOTBALL_KEYPOINTS_CORRECTED, dtype=np.float32).reshape(1, -1, 2)
    projected = cv2.perspectiveTransform(src_all, best_H)[0]
    frame_entry["keypoints"] = [[float(projected[i][0]), float(projected[i][1])] for i in range(len(FOOTBALL_KEYPOINTS_CORRECTED))]
    if DEBUG_FLAG and debug_dir is not None:
        try:
            warped = cv2.warpPerspective(template_image, best_H, (frame_width, frame_height))
            ground_mask, line_mask = _extract_masks(warped)
            predicted_mask = _predicted_lines_mask(video_frame, ground_mask)
            _write_scoring_debug_images(
                frame_id=frame_id, out_dir=debug_dir, video_frame=video_frame,
                warped_template=warped, ground_mask=ground_mask, line_mask=line_mask,
                predicted_mask=predicted_mask, score=best_score,
            )
        except Exception:
            pass
    return (frame_id, best_score, best_label)


def _step5_chunk_worker(
    chunk: list[tuple[dict[str, Any], int]],
    video_url: str,
    template_image: np.ndarray,
    debug_dir: Path | None,
) -> list[tuple[int, float, str | None]]:
    """Process a chunk of frames for Step 5. Each thread has its own VideoCapture."""
    cap = _FrameStore(video_url)
    results: list[tuple[int, float, str | None]] = []
    try:
        for frame_entry, frame_id in chunk:
            video_frame = cap.get(frame_id)
            r = _step5_process_one_frame(frame_entry, frame_id, video_frame, template_image, debug_dir)
            results.append(r)
    finally:
        cap.close()
    return results


def _score_frames_from_input_keypoints(
    frames: list[dict[str, Any]],
    default_width: int | None,
    default_height: int | None,
    *,
    debug_dir: Path | None = None,
    frame_store: "_FrameStore | None" = None,
    video_url: str | None = None,
) -> tuple[list[dict[str, float | int]], float]:
    """
    Step 5: Score and re-project keypoints.

    Try three weighted homographies (H1, H2, H3), score each, pick the best,
    then re-project keypoints using the selected H.
    Pipeline per H: warp template -> ground/line masks -> predicted lines -> overlap ratio.
    """
    print("Step 5: scoring (H1/H2/H3 weighted) and re-projecting keypoints:")
    template_image = _challenge_template()
    per_frame: list[dict[str, float | int]] = []
    scores: list[float] = []
    only_frames = globals().get("ONLY_FRAMES")
    only_frames_set = set(int(x) for x in only_frames) if only_frames else None

    # Build work list: (frame_entry, frame_id) for frames we will process
    work_list: list[tuple[dict[str, Any], int]] = []
    for frame_entry in frames:
        if not isinstance(frame_entry, dict):
            continue
        frame_id_raw = frame_entry.get("frame_id", frame_entry.get("frame_number", -1))
        try:
            frame_id = int(frame_id_raw)
        except Exception:
            frame_id = -1
        if only_frames_set is not None and frame_id not in only_frames_set:
            continue
        kps = frame_entry.get("keypoints")
        if not isinstance(kps, list) or len(kps) != len(FOOTBALL_KEYPOINTS_CORRECTED):
            per_frame.append({"frame": frame_id, "score": 0.0})
            scores.append(0.0)
            continue
        work_list.append((frame_entry, frame_id))

    use_parallel = NUM_THREADS > 1 and video_url and len(work_list) > 0
    if use_parallel:
        n = min(NUM_THREADS, len(work_list))
        chunk_size = (len(work_list) + n - 1) // n
        chunks = [work_list[i : i + chunk_size] for i in range(0, len(work_list), chunk_size)]
        all_results: list[tuple[int, float, str | None]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(_step5_chunk_worker, ch, video_url, template_image, debug_dir)
                for ch in chunks
            ]
            for fut in concurrent.futures.as_completed(futures):
                all_results.extend(fut.result())
        all_results.sort(key=lambda t: t[0])
        for frame_id, score, label in all_results:
            per_frame.append({"frame": frame_id, "score": score})
            scores.append(score)
        for item in sorted(per_frame, key=lambda x: int(x["frame"])):
            fid = int(item["frame"])
            sc = float(item["score"])
            print(f"  frame {fid}: {sc:.4f}", end="\r", flush=True)
    else:
        # Sequential path (original loop)
        for frame_entry in frames:
            if not isinstance(frame_entry, dict):
                continue

            frame_id_raw = frame_entry.get("frame_id", frame_entry.get("frame_number", -1))
            try:
                frame_id = int(frame_id_raw)
            except Exception:
                frame_id = -1

            if only_frames_set is not None and frame_id not in only_frames_set:
                continue

            kps = frame_entry.get("keypoints")
            if not isinstance(kps, list) or len(kps) != len(FOOTBALL_KEYPOINTS_CORRECTED):
                per_frame.append({"frame": frame_id, "score": 0.0})
                scores.append(0.0)
                continue

            # Collect valid correspondences with indices (for weighted H)
            valid_indices: list[int] = []
            valid_src: list[tuple[float, float]] = []
            valid_dst: list[tuple[float, float]] = []
            for idx, kp in enumerate(kps):
                if not isinstance(kp, list) or len(kp) < 2:
                    continue
                x, y = float(kp[0]), float(kp[1])
                if x == 0.0 and y == 0.0:
                    continue
                valid_indices.append(idx)
                valid_src.append(FOOTBALL_KEYPOINTS_CORRECTED[idx])
                valid_dst.append((x, y))

            if len(valid_src) < 4:
                per_frame.append({"frame": frame_id, "score": 0.0})
                scores.append(0.0)
                continue

            video_frame = frame_store.get(frame_id) if frame_store else None
            if video_frame is None:
                per_frame.append({"frame": frame_id, "score": 0.0})
                scores.append(0.0)
                continue
            frame_height, frame_width = video_frame.shape[:2]

            # Try H1, H2, H3 with their weight schemes
            w1 = _weight_map_h1()
            w2 = _weight_map_h2()
            w3 = _weight_map_h3()
            H1 = _find_homography_weighted(valid_indices, valid_src, valid_dst, w1)
            H2 = _find_homography_weighted(valid_indices, valid_src, valid_dst, w2)
            H3 = _find_homography_weighted(valid_indices, valid_src, valid_dst, w3)

            score1 = _score_given_H(H1, template_image, video_frame) if H1 is not None else 0.0
            score2 = _score_given_H(H2, template_image, video_frame) if H2 is not None else 0.0
            score3 = _score_given_H(H3, template_image, video_frame) if H3 is not None else 0.0

            # Pick best H (and tie-break by H1 < H2 < H3)
            best_score = score1
            best_H = H1
            best_label = "H1"
            if score2 > best_score:
                best_score = score2
                best_H = H2
                best_label = "H2"
            if score3 > best_score:
                best_score = score3
                best_H = H3
                best_label = "H3"

            if best_H is None:
                per_frame.append({"frame": frame_id, "score": 0.0})
                scores.append(0.0)
                continue

            # Re-project keypoints using the selected H
            src_all = np.array(FOOTBALL_KEYPOINTS_CORRECTED, dtype=np.float32).reshape(1, -1, 2)
            projected = cv2.perspectiveTransform(src_all, best_H)[0]
            frame_entry["keypoints"] = [[float(projected[i][0]), float(projected[i][1])] for i in range(len(FOOTBALL_KEYPOINTS_CORRECTED))]

            if DEBUG_FLAG and debug_dir is not None:
                warped = cv2.warpPerspective(template_image, best_H, (frame_width, frame_height))
                try:
                    ground_mask, line_mask = _extract_masks(warped)
                    predicted_mask = _predicted_lines_mask(video_frame, ground_mask)
                    _write_scoring_debug_images(
                        frame_id=frame_id,
                        out_dir=debug_dir,
                        video_frame=video_frame,
                        warped_template=warped,
                        ground_mask=ground_mask,
                        line_mask=line_mask,
                        predicted_mask=predicted_mask,
                        score=best_score,
                    )
                except Exception:
                    pass

            print(f"  frame {frame_id}: {best_score:.4f} ({best_label})", end="\r", flush=True)
            per_frame.append({"frame": frame_id, "score": best_score})
            scores.append(best_score)

    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    return per_frame, float(avg_score)


def _score_single_frame(
    keypoints: list[list[float]],
    video_frame: np.ndarray,
    template_image: np.ndarray,
) -> float:
    """
    Score one frame's keypoints using the same pipeline as Step 5.
    Returns 0.0 on any failure or if < 4 valid correspondences.
    """
    if not isinstance(keypoints, list) or len(keypoints) != len(FOOTBALL_KEYPOINTS_CORRECTED):
        return 0.0
    valid_src: list[tuple[float, float]] = []
    valid_dst: list[tuple[float, float]] = []
    for idx, kp in enumerate(keypoints):
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue
        x, y = float(kp[0]), float(kp[1])
        if x == 0.0 and y == 0.0:
            continue
        valid_src.append(FOOTBALL_KEYPOINTS_CORRECTED[idx])
        valid_dst.append((x, y))
    if len(valid_src) < 4:
        return 0.0
    frame_height, frame_width = video_frame.shape[:2]
    try:
        warped = _warp_template(
            template_image.copy(), valid_src, valid_dst, frame_width, frame_height
        )
    except (ValueError, _InvalidMask):
        return 0.0
    try:
        ground_mask, line_mask = _extract_masks(warped)
    except _InvalidMask:
        return 0.0
    predicted_mask = _predicted_lines_mask(video_frame, ground_mask)
    overlap = cv2.bitwise_and(line_mask, predicted_mask)
    pixels_on_lines = int(line_mask.sum())
    pixels_overlap = int(overlap.sum())
    return float(pixels_overlap) / float(pixels_on_lines + 1e-8)


def _step8_process_one_frame(
    frame_entry: dict[str, Any],
    default_width: int | None,
    default_height: int | None,
) -> tuple[int, int]:
    """Process one frame for Step 8. Returns (processed_count, skipped_count)."""
    if not isinstance(frame_entry, dict):
        return (0, 1)

    kps = frame_entry.get("keypoints")
    if not isinstance(kps, list) or len(kps) != len(FOOTBALL_KEYPOINTS):
        return (0, 1)

    frame_width = frame_entry.get("frame_width")
    frame_height = frame_entry.get("frame_height")
    if frame_width is None or frame_height is None:
        frame_width, frame_height = default_width, default_height
    else:
        frame_width = int(frame_width)
        frame_height = int(frame_height)

    if frame_width is None or frame_height is None or frame_width <= 0 or frame_height <= 0:
        frame_width, frame_height = _infer_dims_from_kps(kps)
        if frame_width is None or frame_height is None:
            return (0, 1)

    filtered_src: list[tuple[float, float]] = []
    filtered_dst: list[tuple[float, float]] = []
    valid_indices: list[int] = []

    for idx, kp in enumerate(kps):
        if not isinstance(kp, list) or len(kp) < 2:
            continue
        x, y = float(kp[0]), float(kp[1])
        if x == 0.0 and y == 0.0:
            continue
        filtered_src.append(FOOTBALL_KEYPOINTS_CORRECTED[idx])
        filtered_dst.append((x, y))
        valid_indices.append(idx)

    if len(filtered_src) < 4:
        return (0, 1)

    src_np = np.array(filtered_src, dtype=np.float32)
    dst_np = np.array(filtered_dst, dtype=np.float32)
    H_corrected, _ = cv2.findHomography(src_np, dst_np)
    if H_corrected is None:
        return (0, 1)

    fk_np = np.array(FOOTBALL_KEYPOINTS, dtype=np.float32).reshape(1, -1, 2)
    projected_np = cv2.perspectiveTransform(fk_np, H_corrected)[0]
    num_kps = len(FOOTBALL_KEYPOINTS)
    valid_indices_set = set(valid_indices)
    adjusted_kps: list[list[float]] = [[0.0, 0.0] for _ in range(num_kps)]
    for idx in range(num_kps):
        x, y = float(projected_np[idx][0]), float(projected_np[idx][1])
        if not (0 <= x < frame_width and 0 <= y < frame_height):
            continue
        if STEP6_FILL_MISSING_ENABLED or idx in valid_indices_set:
            adjusted_kps[idx] = [x, y]

    frame_entry["keypoints"] = adjusted_kps
    return (1, 0)


def _step8_chunk_worker(
    chunk: list[dict[str, Any]],
    default_width: int | None,
    default_height: int | None,
) -> tuple[int, int]:
    """Process a chunk of frames for Step 8. Returns (processed_count, skipped_count)."""
    p, s = 0, 0
    for frame_entry in chunk:
        a, b = _step8_process_one_frame(frame_entry, default_width, default_height)
        p += a
        s += b
    return (p, s)


def _run_step8_adjustment(frames: list[dict[str, Any]], default_width: int | None, default_height: int | None) -> tuple[int, int]:
    only_frames = globals().get("ONLY_FRAMES")
    only_frames_set = set(int(x) for x in only_frames) if only_frames else None
    target_frames: list[dict[str, Any]] = []
    for frame_entry in frames:
        if not isinstance(frame_entry, dict):
            continue
        if only_frames_set is None:
            target_frames.append(frame_entry)
            continue
        frame_id_raw = frame_entry.get("frame_id", frame_entry.get("frame_number", -1))
        try:
            frame_id = int(frame_id_raw)
        except Exception:
            continue
        if frame_id in only_frames_set:
            target_frames.append(frame_entry)

    total_frames = len(target_frames)
    print(
        f"Step 8: Adjusting keypoints to use FOOTBALL_KEYPOINTS instead of "
        f"FOOTBALL_KEYPOINTS_CORRECTED (processing {total_frames} frames)..."
    )

    processed_count = 0
    skipped_count = 0

    use_parallel = NUM_THREADS > 1 and len(target_frames) > 0
    if use_parallel:
        n = min(NUM_THREADS, len(target_frames))
        chunk_size = (len(target_frames) + n - 1) // n
        chunks = [target_frames[i : i + chunk_size] for i in range(0, len(target_frames), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(_step8_chunk_worker, ch, default_width, default_height)
                for ch in chunks
            ]
            for fut in concurrent.futures.as_completed(futures):
                p, s = fut.result()
                processed_count += p
                skipped_count += s
    else:
        for frame_idx, frame_entry in enumerate(target_frames):
            if frame_idx % 100 == 0:
                print(
                    f"Step 8: Processed {frame_idx}/{total_frames} frames "
                    f"(adjusted: {processed_count}, skipped: {skipped_count})..."
                )

            if not isinstance(frame_entry, dict):
                skipped_count += 1
                continue

            kps = frame_entry.get("keypoints")
            if not isinstance(kps, list) or len(kps) != len(FOOTBALL_KEYPOINTS):
                skipped_count += 1
                continue

            frame_width = frame_entry.get("frame_width")
            frame_height = frame_entry.get("frame_height")
            if frame_width is None or frame_height is None:
                frame_width, frame_height = default_width, default_height
            else:
                frame_width = int(frame_width)
                frame_height = int(frame_height)

            if frame_width is None or frame_height is None or frame_width <= 0 or frame_height <= 0:
                frame_width, frame_height = _infer_dims_from_kps(kps)
                if frame_width is None or frame_height is None:
                    skipped_count += 1
                    continue

            filtered_src: list[tuple[float, float]] = []
            filtered_dst: list[tuple[float, float]] = []
            valid_indices: list[int] = []

            for idx, kp in enumerate(kps):
                if not isinstance(kp, list) or len(kp) < 2:
                    continue
                x, y = float(kp[0]), float(kp[1])
                if x == 0.0 and y == 0.0:
                    continue
                filtered_src.append(FOOTBALL_KEYPOINTS_CORRECTED[idx])
                filtered_dst.append((x, y))
                valid_indices.append(idx)

            if len(filtered_src) < 4:
                skipped_count += 1
                continue

            src_np = np.array(filtered_src, dtype=np.float32)
            dst_np = np.array(filtered_dst, dtype=np.float32)
            H_corrected, _ = cv2.findHomography(src_np, dst_np)
            if H_corrected is None:
                skipped_count += 1
                continue

            fk_np = np.array(FOOTBALL_KEYPOINTS, dtype=np.float32).reshape(1, -1, 2)
            projected_np = cv2.perspectiveTransform(fk_np, H_corrected)[0]
            num_kps = len(FOOTBALL_KEYPOINTS)
            valid_indices_set = set(valid_indices)
            adjusted_kps: list[list[float]] = [[0.0, 0.0] for _ in range(num_kps)]
            for idx in range(num_kps):
                x, y = float(projected_np[idx][0]), float(projected_np[idx][1])
                if not (0 <= x < frame_width and 0 <= y < frame_height):
                    continue
                if STEP6_FILL_MISSING_ENABLED or idx in valid_indices_set:
                    adjusted_kps[idx] = [x, y]

            frame_entry["keypoints"] = adjusted_kps
            processed_count += 1

    print(
        f"Step 8: Completed processing {total_frames} frames "
        f"(adjusted: {processed_count}, skipped: {skipped_count})."
    )
    return processed_count, skipped_count


def _draw_keypoints_on_frame(frame: np.ndarray, keypoints: list[Any], color: tuple[int, int, int] = (0, 255, 0), radius: int = 4) -> np.ndarray:
    """Draw keypoints on a copy of the frame. Non-zero keypoints only."""
    out = frame.copy()
    for kp in keypoints:
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue
        x, y = int(round(float(kp[0]))), int(round(float(kp[1])))
        if x == 0 and y == 0:
            continue
        cv2.circle(out, (x, y), radius, color, 2)
    return out


def _write_step7_debug_images(
    *,
    out_dir: Path,
    backward_id: int,
    forward_id: int,
    interp_id: int,
    weight: float,
    bwd_frame: np.ndarray,
    fwd_frame: np.ndarray,
    interp_frame: np.ndarray,
    bwd_kps: list[Any],
    fwd_kps: list[Any],
    new_kps: list[list[float]],
    template_image: np.ndarray,
    before_score: float,
    new_score: float,
    applied: bool,
) -> None:
    """Write Step 7 debug images: backward/forward with kps, averaged frame, interp frame with kps, score pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Backward frame with keypoints
    img_bwd = _draw_keypoints_on_frame(bwd_frame, bwd_kps, (0, 255, 0))
    cv2.imwrite(str(out_dir / "01_backward_frame_with_keypoints.jpg"), img_bwd)

    # 2) Forward frame with keypoints
    img_fwd = _draw_keypoints_on_frame(fwd_frame, fwd_kps, (255, 0, 0))
    cv2.imwrite(str(out_dir / "02_forward_frame_with_keypoints.jpg"), img_fwd)

    # 3) Averaged image (blend of backward and forward) for this frame
    blend = cv2.addWeighted(bwd_frame, 1.0 - weight, fwd_frame, weight, 0)
    cv2.imwrite(str(out_dir / "03_interp_frame_averaged.jpg"), blend)

    # 4) Interp frame with interpolated keypoints
    img_interp = _draw_keypoints_on_frame(interp_frame, new_kps, (0, 255, 255))
    cv2.imwrite(str(out_dir / "04_interp_frame_with_interpolated_keypoints.jpg"), img_interp)

    # 5–9) Score calculation pipeline from interpolated keypoints
    try:
        valid_src, valid_dst = [], []
        for idx, kp in enumerate(new_kps):
            if not kp or (float(kp[0]) == 0.0 and float(kp[1]) == 0.0):
                continue
            valid_src.append(FOOTBALL_KEYPOINTS_CORRECTED[idx])
            valid_dst.append((float(kp[0]), float(kp[1])))
        if len(valid_src) >= 4:
            src_np = np.array(valid_src, dtype=np.float32)
            dst_np = np.array(valid_dst, dtype=np.float32)
            H, _ = cv2.findHomography(src_np, dst_np)
            if H is not None:
                h, w = interp_frame.shape[:2]
                warped = cv2.warpPerspective(template_image, H, (w, h))
                cv2.imwrite(str(out_dir / "05_warped_template.jpg"), warped)
                ground_mask, line_mask = _extract_masks(warped)
                cv2.imwrite(str(out_dir / "06_ground_mask.jpg"), ground_mask * 255)
                cv2.imwrite(str(out_dir / "07_line_mask_expected.jpg"), line_mask * 255)
                predicted_mask = _predicted_lines_mask(interp_frame, ground_mask)
                cv2.imwrite(str(out_dir / "08_predicted_lines.jpg"), predicted_mask * 255)
                overlap = cv2.bitwise_and(line_mask, predicted_mask)
                vis = interp_frame.copy()
                vis[line_mask == 1] = (0, 0, 180)
                vis[overlap == 1] = (0, 220, 0)
                cv2.imwrite(str(out_dir / "09_overlap_on_frame.jpg"), vis)
    except Exception:
        pass

    meta = {
        "backward_id": backward_id,
        "forward_id": forward_id,
        "interp_id": interp_id,
        "weight": round(weight, 4),
        "before_score": round(before_score, 4),
        "new_score": round(new_score, 4),
        "applied": applied,
    }
    (out_dir / "score_meta.json").write_text(json.dumps(meta, indent=2))


def _common_indices_in_frame(
    a: list[Any],
    b: list[Any],
    frame_width: int,
    frame_height: int,
) -> list[int]:
    """Indices where both keypoints are non-zero and within frame bounds (0 <= x < w, 0 <= y < h)."""
    out: list[int] = []
    for i in range(min(len(a), len(b))):
        ka, kb = a[i], b[i]
        if not (isinstance(ka, (list, tuple)) and len(ka) >= 2):
            continue
        if not (isinstance(kb, (list, tuple)) and len(kb) >= 2):
            continue
        xa, ya = float(ka[0]), float(ka[1])
        xb, yb = float(kb[0]), float(kb[1])
        if xa == 0.0 and ya == 0.0:
            continue
        if xb == 0.0 and yb == 0.0:
            continue
        if not (0 <= xa < frame_width and 0 <= ya < frame_height):
            continue
        if not (0 <= xb < frame_width and 0 <= yb < frame_height):
            continue
        out.append(i)
    return out


def _run_step7_interpolation(
    frames: list[dict[str, Any]],
    per_frame_scores: list[dict[str, float | int]],
    segments: list[list[int]],
    frame_store: "_FrameStore | None" = None,
    debug_dir: Path | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> int:
    """
    Step 7: Multi-pass keypoint interpolation.

    Runs one pass per threshold in STEP7_SCORE_THRESHOLDS (ascending).
    In each pass, frames whose Step-5 score is below the threshold are
    "problematic"; the nearest backward/forward frames at or above the
    threshold are the anchors.  Interpolation is only applied when both
    anchors belong to the same Step-2 segment and the gap <= STEP7_MAX_GAP.

    For each candidate interpolation we score the new keypoints; we only
    write them when the new score is higher than the original (Step-5) score.

    Returns the total number of frame-keypoint replacements across all passes.
    """
    if not STEP7_ENABLED:
        print("Step 7: skipped (STEP7_ENABLED=False).")
        return 0

    template_image = _challenge_template()
    score_map: dict[int, float] = {int(item["frame"]): float(item["score"]) for item in per_frame_scores}
    sorted_ids = sorted(score_map.keys())
    if not sorted_ids:
        print("Step 7: no scored frames available.")
        return 0

    # frame_id -> segment index
    frame_to_seg: dict[int, int] = {}
    for seg_idx, seg in enumerate(segments):
        for fid in seg:
            frame_to_seg[fid] = seg_idx

    frame_data_map: dict[int, dict[str, Any]] = {}
    for fe in frames:
        if not isinstance(fe, dict):
            continue
        raw = fe.get("frame_id", fe.get("frame_number", -1))
        try:
            fid = int(raw)
        except Exception:
            continue
        frame_data_map[fid] = fe

    def _common_indices(a: list[Any], b: list[Any]) -> list[int]:
        out: list[int] = []
        for i in range(min(len(a), len(b))):
            ka, kb = a[i], b[i]
            if (isinstance(ka, (list, tuple)) and len(ka) >= 2
                    and not (float(ka[0]) == 0.0 and float(ka[1]) == 0.0)
                    and isinstance(kb, (list, tuple)) and len(kb) >= 2
                    and not (float(kb[0]) == 0.0 and float(kb[1]) == 0.0)):
                out.append(i)
        return out

    template_len = len(FOOTBALL_KEYPOINTS)
    total_updated = 0

    for threshold in STEP7_SCORE_THRESHOLDS:
        problematic = [fid for fid in sorted_ids if score_map[fid] < threshold]
        if not problematic:
            print(f"  pass {threshold}: no problematic frames")
            continue

        pass_count = 0
        already_rewritten: set[int] = set()

        for problem_id in problematic:
            backward_id: int | None = None
            for fid in reversed(sorted_ids):
                if fid < problem_id and score_map[fid] >= threshold:
                    backward_id = fid
                    break

            forward_id: int | None = None
            for fid in sorted_ids:
                if fid > problem_id and score_map[fid] >= threshold:
                    forward_id = fid
                    break

            if backward_id is None or forward_id is None:
                continue
            if frame_to_seg.get(backward_id) != frame_to_seg.get(forward_id):
                continue

            gap = forward_id - backward_id
            if gap > STEP7_MAX_GAP:
                continue
            bwd_fe = frame_data_map.get(backward_id)
            fwd_fe = frame_data_map.get(forward_id)
            if bwd_fe is None or fwd_fe is None:
                continue

            bwd_kps: list[Any] = bwd_fe.get("keypoints") or []
            fwd_kps: list[Any] = fwd_fe.get("keypoints") or []
            if frame_width is not None and frame_height is not None:
                common_set = set(_common_indices_in_frame(bwd_kps, fwd_kps, frame_width, frame_height))
            else:
                common_set = set(_common_indices(bwd_kps, fwd_kps))

            if len(common_set) < 4:
                continue

            for interp_id in sorted_ids:
                if not (backward_id < interp_id < forward_id):
                    continue
                if interp_id in already_rewritten:
                    continue
                fe = frame_data_map.get(interp_id)
                if fe is None:
                    continue
                weight = (interp_id - backward_id) / gap
                max_len = max(len(bwd_kps), len(fwd_kps), template_len)
                new_kps = []
                for i in range(max_len):
                    if i in common_set:
                        bx = float(bwd_kps[i][0])
                        by = float(bwd_kps[i][1])
                        fx = float(fwd_kps[i][0])
                        fy = float(fwd_kps[i][1])
                        new_kps.append([bx + (fx - bx) * weight, by + (fy - by) * weight])
                    else:
                        new_kps.append([0.0, 0.0])
                # Only update when the interpolated keypoints score higher than original
                before_score = score_map[interp_id]
                if frame_store is None:
                    continue
                video_frame = frame_store.get(interp_id)
                if video_frame is None:
                    continue
                new_score = _score_single_frame(new_kps, video_frame, template_image)
                applied = new_score > before_score

                if DEBUG_FLAG and debug_dir is not None and frame_store is not None:
                    bwd_frame = frame_store.get(backward_id)
                    fwd_frame = frame_store.get(forward_id)
                    if bwd_frame is not None and fwd_frame is not None:
                        step7_dir = debug_dir / "step7" / f"frame_{interp_id:04d}_bwd{backward_id}_fwd{forward_id}"
                        _write_step7_debug_images(
                            out_dir=step7_dir,
                            backward_id=backward_id,
                            forward_id=forward_id,
                            interp_id=interp_id,
                            weight=weight,
                            bwd_frame=bwd_frame,
                            fwd_frame=fwd_frame,
                            interp_frame=video_frame,
                            bwd_kps=bwd_kps,
                            fwd_kps=fwd_kps,
                            new_kps=new_kps,
                            template_image=template_image,
                            before_score=before_score,
                            new_score=new_score,
                            applied=applied,
                        )

                if not applied:
                    continue
                fe["keypoints"] = new_kps
                already_rewritten.add(interp_id)
                pass_count += 1

        print(f"  pass {threshold}: interpolated {pass_count} frame(s)")
        total_updated += pass_count

    print(f"Step 7: {total_updated} frame(s) interpolated across {len(STEP7_SCORE_THRESHOLDS)} passes (max_gap={STEP7_MAX_GAP})")
    return total_updated


# Pixel distance threshold for two keypoints across frames to be considered "the same"
SEGMENT_KP_PIXEL_THRESHOLD: float = 50.0
# Minimum common keypoints required to consider two adjacent frames in the same segment
SEGMENT_MIN_COMMON_KPS: int = 2


def _valid_kps(frame_entry: dict[str, Any]) -> dict[int, tuple[float, float]]:
    """Return {index: (x, y)} for all non-zero keypoints in a frame."""
    kps = frame_entry.get("keypoints") or []
    out: dict[int, tuple[float, float]] = {}
    for idx, kp in enumerate(kps):
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue
        x, y = float(kp[0]), float(kp[1])
        if x != 0.0 or y != 0.0:
            out[idx] = (x, y)
    return out


def _count_common_kps(
    a: dict[int, tuple[float, float]],
    b: dict[int, tuple[float, float]],
    threshold: float,
) -> int:
    """Count keypoint indices present in both frames whose positions are within threshold px."""
    count = 0
    for idx, (ax, ay) in a.items():
        if idx not in b:
            continue
        bx, by = b[idx]
        if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= threshold:
            count += 1
    return count


def _detect_segments(
    frames: list[dict[str, Any]],
) -> tuple[list[list[int]], list[int]]:
    """
    Step 2: Segment detection.

    Scans the ordered frame list and groups consecutive frames into segments.
    Two adjacent frames belong to the same segment when they share at least
    SEGMENT_MIN_COMMON_KPS keypoints whose positions differ by at most
    SEGMENT_KP_PIXEL_THRESHOLD pixels.

    Returns:
        segments  – list of segments, each segment is a list of frame IDs.
        empty_ids – list of frame IDs that carry no valid keypoints at all.
    """
    # Sort frames by frame_id so adjacency is meaningful
    id_frame: list[tuple[int, dict[str, Any]]] = []
    empty_ids: list[int] = []

    for fe in frames:
        if not isinstance(fe, dict):
            continue
        raw = fe.get("frame_id", fe.get("frame_number", -1))
        try:
            fid = int(raw)
        except Exception:
            fid = -1
        vkps = _valid_kps(fe)
        if not vkps:
            empty_ids.append(fid)
        else:
            id_frame.append((fid, fe))

    id_frame.sort(key=lambda t: t[0])

    segments: list[list[int]] = []
    if not id_frame:
        return segments, sorted(empty_ids)

    current_segment: list[int] = [id_frame[0][0]]
    prev_vkps = _valid_kps(id_frame[0][1])

    for i in range(1, len(id_frame)):
        fid, fe = id_frame[i]
        cur_vkps = _valid_kps(fe)
        common = _count_common_kps(prev_vkps, cur_vkps, SEGMENT_KP_PIXEL_THRESHOLD)
        if common >= SEGMENT_MIN_COMMON_KPS:
            current_segment.append(fid)
        else:
            segments.append(current_segment)
            current_segment = [fid]
        prev_vkps = cur_vkps

    segments.append(current_segment)
    return segments, sorted(empty_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only Step 8 keypoint adjustment.")
    parser.add_argument("--video-url", required=True, help="Path/URL of video (used for frame dimensions).")
    parser.add_argument("--unsorted-json", required=True, type=Path, help="Input miner JSON path.")
    args = parser.parse_args()

    raw_data = json.loads(args.unsorted_json.read_text())
    ordered_raw, ordered_frames = _extract_frames_container(raw_data)

    width, height = _get_video_dims(args.video_url)

    # ── Step 1: infer center-circle keypoint (kp[30] or kp[31]) when only 14,15 and one of 30/31 valid ──
    step1_updated = _run_step1_infer_center_circle(ordered_frames, width, height)
    if DEBUG_FLAG and step1_updated > 0:
        print(f"Step 1: inferred center-circle keypoint for {step1_updated} frame(s)")
    elif DEBUG_FLAG:
        print("Step 1: no frames matched center-circle inference condition")

    # ── Step 2: segment detection ──────────────────────────────────────────
    segments, empty_ids = _detect_segments(ordered_frames)
    total_frames_with_kps = sum(len(s) for s in segments)
    print(
        f"Step 2: {len(ordered_frames)} frames total  |  "
        f"{total_frames_with_kps} with keypoints  |  "
        f"{len(empty_ids)} empty"
    )
    print(f"Step 2: {len(segments)} segment(s) detected "
          f"(threshold: {SEGMENT_MIN_COMMON_KPS} common kps within {SEGMENT_KP_PIXEL_THRESHOLD}px)")
    for seg_idx, seg in enumerate(segments):
        print(f"  segment {seg_idx + 1:>3}: frames {seg[0]:>5} – {seg[-1]:>5}  ({len(seg)} frames)")
    if empty_ids:
        ranges: list[str] = []
        start = prev = empty_ids[0]
        for eid in empty_ids[1:]:
            if eid == prev + 1:
                prev = eid
            else:
                ranges.append(f"{start}–{prev}" if start != prev else str(start))
                start = prev = eid
        ranges.append(f"{start}–{prev}" if start != prev else str(start))
        print(f"  empty frames: {', '.join(ranges)}")
    print()

    scoring_debug_dir: Path | None = None
    if DEBUG_FLAG:
        scoring_debug_dir = Path("debug_frames") / "scoring_step_by_step" / args.unsorted_json.stem
        scoring_debug_dir.mkdir(parents=True, exist_ok=True)
    frame_store = _FrameStore(args.video_url)
    try:
        per_frame_scores, avg_score = _score_frames_from_input_keypoints(
            ordered_frames, width, height, debug_dir=scoring_debug_dir, frame_store=frame_store,
            video_url=args.video_url,
        )
        print(f"\nAverage input keypoint score: {avg_score:.4f}")

        _run_step7_interpolation(
            ordered_frames, per_frame_scores, segments,
            frame_store=frame_store,
            debug_dir=scoring_debug_dir,
            frame_width=width,
            frame_height=height,
        )
    finally:
        frame_store.close()

    if STEP8_ENABLED:
        _run_step8_adjustment(ordered_frames, width, height)
    else:
        print("Step 8: skipped (STEP8_ENABLED=False).")

    optimized_dir = Path("miner_responses_ordered_optimised")
    optimized_dir.mkdir(parents=True, exist_ok=True)
    optimized_path = optimized_dir / args.unsorted_json.name
    optimized_path.write_text(json.dumps(ordered_raw, indent=2))
    print(f"Wrote optimised miner JSON to {optimized_path}")


if __name__ == "__main__":
    main()
