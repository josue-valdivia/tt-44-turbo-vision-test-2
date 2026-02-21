from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

STEP6_FILL_MISSING_ENABLED = True
# If False, skip Step 8 keypoint adjustment.
STEP8_ENABLED = True
# Optional: process only specific frames. Keep empty to process all.
# ONLY_FRAMES: list[int] = list(range(395, 396))
# If True, write step-by-step scoring debug images.
DEBUG_FLAG = False

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


def _get_video_frame_cv2(video_url: str, frame_id: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_url)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        return frame if ok and frame is not None else None
    finally:
        cap.release()



def _score_frames_from_input_keypoints(
    frames: list[dict[str, Any]],
    default_width: int | None,
    default_height: int | None,
    *,
    debug_dir: Path | None = None,
    video_url: str | None = None,
) -> tuple[list[dict[str, float | int]], float]:
    """
    Score input keypoints using the exact same pipeline as keypoints_calculate_score.py:
      1. Build H from FOOTBALL_KEYPOINTS_CORRECTED -> observed frame keypoints.
      2. Warp football_pitch_template.png into frame space using H.
      3. Extract ground mask (threshold > 10) and expected line mask (threshold > 200).
      4. Extract predicted lines: top-hat + Gaussian blur + Canny + dilate, masked by ground.
      5. Score = overlap(expected_lines, predicted_lines) / expected_line_pixels.
    """
    print("Input keypoint scores (using FOOTBALL_KEYPOINTS_CORRECTED homography):")
    template_image = _challenge_template()
    per_frame: list[dict[str, float | int]] = []
    scores: list[float] = []
    only_frames = globals().get("ONLY_FRAMES")
    only_frames_set = set(int(x) for x in only_frames) if only_frames else None

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

        # Collect valid FOOTBALL_KEYPOINTS_CORRECTED -> frame correspondences
        valid_src: list[tuple[float, float]] = []
        valid_dst: list[tuple[float, float]] = []
        for idx, kp in enumerate(kps):
            if not isinstance(kp, list) or len(kp) < 2:
                continue
            x, y = float(kp[0]), float(kp[1])
            if abs(x) < 1e-6 and abs(y) < 1e-6:
                continue
            valid_src.append(FOOTBALL_KEYPOINTS_CORRECTED[idx])
            valid_dst.append((x, y))

        if len(valid_src) < 4:
            per_frame.append({"frame": frame_id, "score": 0.0})
            scores.append(0.0)
            continue

        # Read actual video frame (needed for edge detection and dimensions)
        video_frame = _get_video_frame_cv2(video_url or "", frame_id)
        if video_frame is None:
            per_frame.append({"frame": frame_id, "score": 0.0})
            scores.append(0.0)
            continue
        frame_height, frame_width = video_frame.shape[:2]

        # Warp template using H from FOOTBALL_KEYPOINTS_CORRECTED -> frame
        try:
            warped = _warp_template(
                template_image.copy(), valid_src, valid_dst, frame_width, frame_height
            )
        except (ValueError, _InvalidMask):
            per_frame.append({"frame": frame_id, "score": 0.0})
            scores.append(0.0)
            continue

        # Extract ground and expected line masks from warped template
        try:
            ground_mask, line_mask = _extract_masks(warped)
        except _InvalidMask:
            per_frame.append({"frame": frame_id, "score": 0.0})
            scores.append(0.0)
            continue

        # Extract predicted lines from the real frame using edge detection
        predicted_mask = _predicted_lines_mask(video_frame, ground_mask)

        # Score = overlap / expected_line_pixels  (same formula as keypoints_calculate_score.py)
        overlap = cv2.bitwise_and(line_mask, predicted_mask)
        pixels_on_lines = int(line_mask.sum())
        pixels_overlap = int(overlap.sum())
        frame_score = float(pixels_overlap) / float(pixels_on_lines + 1e-8)

        if DEBUG_FLAG and debug_dir is not None:
            _write_scoring_debug_images(
                frame_id=frame_id,
                out_dir=debug_dir,
                video_frame=video_frame,
                warped_template=warped,
                ground_mask=ground_mask,
                line_mask=line_mask,
                predicted_mask=predicted_mask,
                score=frame_score,
            )

        print(f"  frame {frame_id}: {frame_score:.4f}", end="\r", flush=True)
        per_frame.append({"frame": frame_id, "score": frame_score})
        scores.append(frame_score)

    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    return per_frame, float(avg_score)


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

        # Filter keypoints the same way project_image_using_keypoints does:
        # exclude only (0, 0) points (missing/default), keep everything else.
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

        # Use cv2.findHomography (same algorithm as keypoints_calculate_score.py)
        src_np = np.array(filtered_src, dtype=np.float32)
        dst_np = np.array(filtered_dst, dtype=np.float32)
        H_corrected, _ = cv2.findHomography(src_np, dst_np)
        if H_corrected is None:
            skipped_count += 1
            continue

        # Apply H to all FOOTBALL_KEYPOINTS (same as cv2.perspectiveTransform)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only Step 8 keypoint adjustment.")
    parser.add_argument("--video-url", required=True, help="Path/URL of video (used for frame dimensions).")
    parser.add_argument("--unsorted-json", required=True, type=Path, help="Input miner JSON path.")
    args = parser.parse_args()

    raw_data = json.loads(args.unsorted_json.read_text())
    ordered_raw, ordered_frames = _extract_frames_container(raw_data)

    width, height = _get_video_dims(args.video_url)
    scoring_debug_dir: Path | None = None
    if DEBUG_FLAG:
        scoring_debug_dir = Path("debug_frames") / "scoring_step_by_step" / args.unsorted_json.stem
        scoring_debug_dir.mkdir(parents=True, exist_ok=True)
    per_frame_scores, avg_score = _score_frames_from_input_keypoints(
        ordered_frames, width, height, debug_dir=scoring_debug_dir, video_url=args.video_url
    )
    print(f"\nAverage input keypoint score: {avg_score:.4f}")

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
