from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

NUM_THREADS: int = 8

STEP6_FILL_MISSING_ENABLED = True
STEP7_ENABLED = True
STEP7_SCORE_THRESHOLDS: list[float] = [0.3, 0.5]
STEP7_MAX_GAP: int = 20
STEP8_ENABLED = True

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


class _FrameStore:
    def __init__(self, source: str) -> None:
        self._cap = cv2.VideoCapture(source)
        self._last_id: int | None = None

    def get(self, frame_id: int) -> np.ndarray | None:
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

    def get_range(self, start: int, end: int) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for i in range(start, end):
            frame = self.get(i)
            if frame is not None:
                out[i] = frame
        return out

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
) -> tuple[int, float, str | None]:
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
    return (frame_id, best_score, best_label)


def _step5_chunk_worker(
    chunk: list[tuple[dict[str, Any], int]],
    video_url: str,
    template_image: np.ndarray,
) -> list[tuple[int, float, str | None]]:
    cap = _FrameStore(video_url)
    results: list[tuple[int, float, str | None]] = []
    try:
        for frame_entry, frame_id in chunk:
            video_frame = cap.get(frame_id)
            r = _step5_process_one_frame(frame_entry, frame_id, video_frame, template_image)
            results.append(r)
    finally:
        cap.close()
    return results


def _score_frames_from_input_keypoints(
    frames: list[dict[str, Any]],
    default_width: int | None,
    default_height: int | None,
    *,
    frame_store: "_FrameStore | None" = None,
    video_url: str | None = None,
) -> tuple[list[dict[str, float | int]], float]:
    template_image = _challenge_template()
    per_frame: list[dict[str, float | int]] = []
    scores: list[float] = []
    only_frames = globals().get("ONLY_FRAMES")
    only_frames_set = set(int(x) for x in only_frames) if only_frames else None
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
                executor.submit(_step5_chunk_worker, ch, video_url, template_image)
                for ch in chunks
            ]
            for fut in concurrent.futures.as_completed(futures):
                all_results.extend(fut.result())
        all_results.sort(key=lambda t: t[0])
        for frame_id, score, label in all_results:
            per_frame.append({"frame": frame_id, "score": score})
            scores.append(score)
    else:
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
            w1 = _weight_map_h1()
            w2 = _weight_map_h2()
            w3 = _weight_map_h3()
            H1 = _find_homography_weighted(valid_indices, valid_src, valid_dst, w1)
            H2 = _find_homography_weighted(valid_indices, valid_src, valid_dst, w2)
            H3 = _find_homography_weighted(valid_indices, valid_src, valid_dst, w3)

            score1 = _score_given_H(H1, template_image, video_frame) if H1 is not None else 0.0
            score2 = _score_given_H(H2, template_image, video_frame) if H2 is not None else 0.0
            score3 = _score_given_H(H3, template_image, video_frame) if H3 is not None else 0.0
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
            src_all = np.array(FOOTBALL_KEYPOINTS_CORRECTED, dtype=np.float32).reshape(1, -1, 2)
            projected = cv2.perspectiveTransform(src_all, best_H)[0]
            frame_entry["keypoints"] = [[float(projected[i][0]), float(projected[i][1])] for i in range(len(FOOTBALL_KEYPOINTS_CORRECTED))]
            per_frame.append({"frame": frame_id, "score": best_score})
            scores.append(best_score)

    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    return per_frame, float(avg_score)


def _score_single_frame(
    keypoints: list[list[float]],
    video_frame: np.ndarray,
    template_image: np.ndarray,
) -> float:
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
    return processed_count, skipped_count


def _common_indices_in_frame(
    a: list[Any],
    b: list[Any],
    frame_width: int,
    frame_height: int,
) -> list[int]:
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
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> int:
    if not STEP7_ENABLED:
        return 0
    template_image = _challenge_template()
    score_map: dict[int, float] = {int(item["frame"]): float(item["score"]) for item in per_frame_scores}
    sorted_ids = sorted(score_map.keys())
    if not sorted_ids:
        return 0
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
            continue

        segments_seen: dict[tuple[int, int], tuple[list[Any], list[Any], set[int]]] = {}
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
            if forward_id - backward_id > STEP7_MAX_GAP:
                continue
            bwd_fe = frame_data_map.get(backward_id)
            fwd_fe = frame_data_map.get(forward_id)
            if bwd_fe is None or fwd_fe is None:
                continue
            bwd_kps = bwd_fe.get("keypoints") or []
            fwd_kps = fwd_fe.get("keypoints") or []
            if frame_width is not None and frame_height is not None:
                common_set = set(_common_indices_in_frame(bwd_kps, fwd_kps, frame_width, frame_height))
            else:
                common_set = set(_common_indices(bwd_kps, fwd_kps))
            if len(common_set) < 4:
                continue
            key = (backward_id, forward_id)
            if key not in segments_seen:
                segments_seen[key] = (bwd_kps, fwd_kps, common_set)

        pass_count = 0
        already_rewritten: set[int] = set()
        for (backward_id, forward_id), (bwd_kps, fwd_kps, common_set) in segments_seen.items():
            gap = forward_id - backward_id
            if frame_store is not None:
                frame_cache = frame_store.get_range(backward_id + 1, forward_id)
            else:
                frame_cache = {}
            for interp_id in range(backward_id + 1, forward_id):
                if interp_id not in frame_data_map:
                    continue
                if interp_id in already_rewritten:
                    continue
                fe = frame_data_map.get(interp_id)
                if fe is None:
                    continue
                video_frame = frame_cache.get(interp_id)
                if video_frame is None:
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
                before_score = score_map[interp_id]
                new_score = _score_single_frame(new_kps, video_frame, template_image)
                if new_score <= before_score:
                    continue
                fe["keypoints"] = new_kps
                already_rewritten.add(interp_id)
                pass_count += 1
        total_updated += pass_count
    return total_updated


SEGMENT_KP_PIXEL_THRESHOLD: float = 50.0
SEGMENT_MIN_COMMON_KPS: int = 2


def _valid_kps(frame_entry: dict[str, Any]) -> dict[int, tuple[float, float]]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-url", required=True, type=str)
    parser.add_argument("--unsorted-json", required=True, type=Path)
    args = parser.parse_args()

    raw_data = json.loads(args.unsorted_json.read_text())
    ordered_raw, ordered_frames = _extract_frames_container(raw_data)
    segments, empty_ids = _detect_segments(ordered_frames)
    width, height = _get_video_dims(args.video_url)
    frame_store = _FrameStore(args.video_url)
    try:
        per_frame_scores, avg_score = _score_frames_from_input_keypoints(
            ordered_frames, width, height, frame_store=frame_store, video_url=args.video_url,
        )
        _run_step7_interpolation(
            ordered_frames, per_frame_scores, segments,
            frame_store=frame_store,
            frame_width=width,
            frame_height=height,
        )
    finally:
        frame_store.close()

    if STEP8_ENABLED:
        _run_step8_adjustment(ordered_frames, width, height)
    optimized_dir = Path("miner_responses_ordered_optimised")
    optimized_dir.mkdir(parents=True, exist_ok=True)
    optimized_path = optimized_dir / args.unsorted_json.name
    optimized_path.write_text(json.dumps(ordered_raw, indent=2))


if __name__ == "__main__":
    main()
