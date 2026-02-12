# cython: boundscheck=True
# cython: wraparound=True
from __future__ import annotations
import argparse
import functools
import json
import logging
from operator import truediv
import os
import sys
import threading
import time
import importlib.util
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Iterable, Sequence
import cv2
import numpy as np
_C3: float = 0
_P1: float = 30.0 * 60.0

def _p2(path: Path) -> float | None:
    try:
        st = path.stat()
        return getattr(st, 'st_birthtime', None) or float(st.st_mtime)
    except OSError:
        return None

def _p3() -> list[Path]:
    base = Path(__file__).parent
    paths = []
    kp_so = sorted(base.glob('keypoints_convert*.so'))
    if kp_so:
        paths.append(kp_so[0])
    elif (base / 'keypoints_convert.py').exists():
        paths.append(base / 'keypoints_convert.py')
    cy_so = sorted(base.glob('keypoints_cy_all*.so'))
    if cy_so:
        paths.append(cy_so[0])
    elif (base / 'keypoints_cy_all.pyx').exists():
        paths.append(base / 'keypoints_cy_all.pyx')
    return paths

def _manual_import_cy(name: str, base_dir: Path):
    so_candidates = sorted(base_dir.glob(f'{name}*.so'))
    if so_candidates:
        so_path = so_candidates[0]
        spec = importlib.util.spec_from_file_location(name, so_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                return (module, None)
            except Exception:
                return (None, 101.3)
        else:
            return (None, 101.2)
    pyx_path = base_dir / f'{name}.pyx'
    if pyx_path.is_file():
        try:
            try:
                import pyximport
            except Exception:
                return (None, 101.4)
            pyximport.install(setup_args={'include_dirs': [np.get_include()]}, language_level=3, build_in_temp=True)
            old_path = sys.path[:]
            if str(base_dir) not in sys.path:
                sys.path.insert(0, str(base_dir))
            try:
                module = __import__(name)
                sys.modules.pop(name, None)
                return (module, None)
            finally:
                sys.path = old_path
        except Exception:
            return (None, 101.4)
    return (None, 101.1)
_CY_MODULE_NAME = 'keypoints_cy_all'
_CY_BASE_DIR = Path(__file__).parent
_cy_module, _CY_LOAD_ERROR = _manual_import_cy(_CY_MODULE_NAME, _CY_BASE_DIR)
_adding_four_points_cy_candidate_logged = False
if _cy_module is not None:
    _seg_candidates_col_cy = getattr(_cy_module, 'segments_from_col_band', None)
    _seg_candidates_row_cy = getattr(_cy_module, 'segments_from_row_band', None)
    _seg_precompute_cy = getattr(_cy_module, 'precompute_segments_from_prefix', None)
    _normalize_keypoints_cy = getattr(_cy_module, 'normalize_keypoints', None)
    _search_horizontal_in_area_cy = getattr(_cy_module, 'search_horizontal_in_area_cy', None)
    _search_vertical_in_area_cy = getattr(_cy_module, 'search_vertical_in_area_cy', None)
    _search_horizontal_in_area_integral_cy = getattr(_cy_module, 'search_horizontal_in_area_integral_cy', None)
    _search_vertical_in_area_integral_cy = getattr(_cy_module, 'search_vertical_in_area_integral_cy', None)
    _sloping_line_white_count_cy = getattr(_cy_module, 'sloping_line_white_count_cy', None)
else:
    _seg_candidates_col_cy = None
    _seg_candidates_row_cy = None
    _seg_precompute_cy = None
    _normalize_keypoints_cy = None
    _search_horizontal_in_area_cy = None
    _search_vertical_in_area_cy = None
    _search_horizontal_in_area_integral_cy = None
    _search_vertical_in_area_integral_cy = None
    _sloping_line_white_count_cy = None
TV_KP_PROFILE: bool = False
# ONLY_FRAMES = list(range(2, 3))
PROCESSING_SCALE = 1.0
REFINE_EXPECTED_MASKS_AT_BOUNDARIES = False
STEP5_ENABLED = True
STEP6_ENABLED = True
STEP4_3_ENABLED = True
_KP_SCORE_CACHE: OrderedDict[tuple, float] = OrderedDict()
_KP_WARP_CACHE: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
_KP_PRED_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_KP_CACHE_MAX = 512
STEP7_ENABLED = False
ADDING_FOUR_POINTS_ENABLED = True
ADD4_PAIR_0_1_9_13 = True
ADD4_PAIR_13_17_24_25 = True
ADD4_PAIR_15_16_19_20 = True
ADD4_PAIR_22_23_25_26_27 = True
ADD4_PAIR22_TOP_YY = 3
ADD4_PAIR22_TOP_YY2 = 3
ADD4_PAIR22_EARLY_STOP_SCORE = 0.95
ADD4_USE_POLYGON_MASKS = True
ADD4_POLYGON_PREDICTED_DILATE = True
ADD4_POLYGON_SCORE_EDGE_RATE = False
ADD4_POLYGON_USE_JSON = True
ADD4_POLYGON_MASKS: dict[str, dict[str, list[list[tuple[float, float]]]]] = {'pair_0_1_9_13': {'ground': [[(0, 0), (888, 0), (888, 900), (0, 900)]], 'line_add': [[(0, 0), (888, 0), (888, 5), (527, 5), (527, 900), (524, 900), (524, 5), (5, 5), (5, 138), (166, 138), (166, 900), (163, 900), (163, 141), (5, 141), (5, 900), (0, 900)]], 'line_sub': []}, 'pair_13_17_24_25': {'ground': [[(522, 0), (1051, 0), (1051, 900), (522, 900)]], 'line_add': [[(522, 0), (1051, 0), (1051, 900), (1046, 900), (1046, 141), (888, 141), (888, 900), (885, 900), (885, 138), (1046, 138), (1046, 5), (527, 5), (527, 900), (522, 900)]], 'line_sub': []}, 'pair_15_16_19_20': {'ground': [[(522, 410), (888, 410), (900, 410), (900, 681), (522, 681)]], 'line_add': [[(522, 410), (522, 681), (900, 681), (900, 676), (527, 676), (527, 410)], [(885, 410), (888, 410), (888, 542), (885, 542)], [(527, 432), (538, 431), (548, 429), (564, 424), (575, 417), (585, 410), (582, 408), (574, 414), (563, 421), (548, 426), (537, 428), (527, 429)]], 'line_sub': []}, 'pair_22_23_25_26_27': {'ground': [[(888, 138), (1051, 138), (1051, 432), (888, 432)]], 'line_add': [[(888, 138), (1051, 138), (1051, 432), (995, 432), (995, 248), (1046, 248), (1046, 141), (888, 141)]], 'line_sub': [[(998, 251), (1046, 251), (1046, 429), (998, 429)]]}}
_POLYGON_JSON_PATH = Path(__file__).parent / 'polygon.json'
_polygon_masks_from_json_cache: dict[str, list[list[tuple[float, float]]]] | None = None

def _get_polygon_masks(group_name: str) -> dict[str, list[list[tuple[float, float]]]] | None:
    global _polygon_masks_from_json_cache
    if ADD4_POLYGON_USE_JSON:
        if _polygon_masks_from_json_cache is None:
            if not _POLYGON_JSON_PATH.is_file():
                return None
            with open(_POLYGON_JSON_PATH, encoding='utf-8') as f:
                data = json.load(f)

            def _to_tuples(rings: list) -> list[list[tuple[float, float]]]:
                return [[(float(p[0]), float(p[1])) for p in ring] for ring in rings]
            _polygon_masks_from_json_cache = {'ground': _to_tuples(data.get('ground', [])), 'line_add': _to_tuples(data.get('line_add', [])), 'line_sub': _to_tuples(data.get('line_sub', []))}
        return _polygon_masks_from_json_cache
    return ADD4_POLYGON_MASKS.get(group_name)
PREV_RELATIVE_FLAG = True
BORDER_50PX_REMOVE_FLAG = True
SIMILARITY_MIN_SCORE_THRESHOLD = 0.3
STEP5_SCORE_THRESHOLD = 0.9
KEYPOINT_H_CONVERT_FLAG = True
FORCE_DECISION_RIGHT = False
STEP4_1_LINE_REFINEMENT_ENABLED = True
STEP4_1_LINE_WEIGHT = 8
STEP4_EARLY_TERMINATE_THRESHOLD = 20.0
STEP4_MAX_CANDIDATES = 50
BLACKLISTS: tuple[tuple[int, int, int, int], ...] = ((23, 24, 27, 28), (7, 8, 3, 4), (2, 10, 1, 14), (18, 26, 14, 25), (5, 13, 6, 17), (21, 29, 17, 30), (10, 11, 2, 3), (10, 11, 2, 7), (12, 13, 4, 5), (12, 13, 5, 8), (18, 19, 26, 27), (18, 19, 26, 23), (20, 21, 24, 29), (20, 21, 28, 29), (8, 4, 5, 13), (3, 7, 2, 10), (23, 27, 18, 26), (24, 28, 21, 29))
STEP8_EDGE_CACHE_MAX = 512
STEP3_PATTERN_CACHE_MAX = 512
_FRAME_CACHE: OrderedDict[int, np.ndarray] = OrderedDict()
_FRAME_CACHE_MAX = 64
def _frame_cache_get(frame_store, frame_id: int) -> np.ndarray:
    frame_id = int(frame_id)
    if frame_id in _FRAME_CACHE:
        _FRAME_CACHE.move_to_end(frame_id)
        return _FRAME_CACHE[frame_id]
    frame = frame_store.get_frame(frame_id)
    _FRAME_CACHE[frame_id] = frame
    while len(_FRAME_CACHE) > _FRAME_CACHE_MAX:
        _FRAME_CACHE.popitem(last=False)
    return frame

def _frame_cache_clear() -> None:
    _FRAME_CACHE.clear()

def _count_valid_keypoints(kps: list[list[float]] | None) -> int:
    if not kps:
        return 0
    try:
        arr = np.array(kps, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return sum((1 for pt in kps if pt and len(pt) >= 2 and (not (abs(pt[0]) < 1e-06 and abs(pt[1]) < 1e-06))))
        valid_mask = np.abs(arr[:, :2]).max(axis=1) > 1e-06
        return int(np.sum(valid_mask))
    except (ValueError, TypeError):
        return sum((1 for pt in kps if pt and len(pt) >= 2 and (not (abs(pt[0]) < 1e-06 and abs(pt[1]) < 1e-06))))

def _near_edges(x: float, y: float, W: int, H: int, t: int=50) -> set[str]:
    edges = set()
    if x <= t:
        edges.add('left')
    if x >= W - t:
        edges.add('right')
    if y <= t:
        edges.add('top')
    if y >= H - t:
        edges.add('bottom')
    return edges

def _both_points_same_direction(A: tuple[float, float], B: tuple[float, float], W: int, H: int, t: int=100) -> bool:
    edges_A = _near_edges(A[0], A[1], W, H, t)
    edges_B = _near_edges(B[0], B[1], W, H, t)
    if not edges_A or not edges_B:
        return False
    return not edges_A.isdisjoint(edges_B)
if _cy_module is not None:
    _connected_by_segment_cy = getattr(_cy_module, 'connected_by_segment', None)
    _step3_filter_labels_cy = getattr(_cy_module, 'filter_labels', None)
    _step3_conn_constraints_cy = getattr(_cy_module, 'filter_connection_constraints', None)
    _step3_conn_label_constraints_cy = getattr(_cy_module, 'filter_connection_label_constraints', None)
else:
    _connected_by_segment_cy = None
    _step3_filter_labels_cy = None
    _step3_conn_constraints_cy = None
    _step3_conn_label_constraints_cy = None

def _step8_edges_cache_put(cache: OrderedDict[int, np.ndarray], key: int, value: np.ndarray, max_size: int) -> None:
    if max_size <= 0:
        return
    if key in cache:
        cache.pop(key, None)
    cache[key] = value
    if len(cache) > max_size:
        cache.popitem(last=False)

def parse_miner_prediction(miner_run: Any) -> dict[int, dict]:
    predicted_frames = ((miner_run.predictions or {}).get('frames') if miner_run.predictions else None) or []
    miner_annotations = {}
    for predicted_frame in predicted_frames:
        frame_number = predicted_frame.get('frame_id', predicted_frame.get('frame_number', -1))
        bboxes = predicted_frame.get('boxes') or predicted_frame.get('bboxes') or []
        keypoints = predicted_frame.get('keypoints') or []
        labels = predicted_frame.get('labels') or predicted_frame.get('keypoints_labels')
        keypoints_labeled = predicted_frame.get('keypoints_labeled')
        miner_annotations[int(frame_number)] = {'bboxes': bboxes, 'keypoints': keypoints, 'labels': labels, 'keypoints_labeled': keypoints_labeled}
    return miner_annotations

def download_video_cached(url: str, _frame_numbers: list[int] | None=None):
    import urllib.request
    import tempfile
    import shutil
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme in ('http', 'https'):
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            with urllib.request.urlopen(url) as resp, open(tmp_path, 'wb') as out:
                shutil.copyfileobj(resp, out)
        except Exception as e:
            try:
                tmp_path.unlink()
            except Exception:
                pass
            raise RuntimeError(f'Failed to download video: {e}')
        return (tmp_path, FrameStore(str(tmp_path)))
    else:
        if not Path(url).exists():
            raise RuntimeError(f'Video path does not exist: {url}')
        return (None, FrameStore(url))

def extract_mask_of_ground_lines_in_image(image: np.ndarray, ground_mask: np.ndarray, blur_ksize: int=5, canny_low: int=30, canny_high: int=100, use_tophat: bool=True, dilate_kernel_size: int=3, dilate_iterations: int=3, cached_edges: np.ndarray | None=None) -> np.ndarray:
    if cached_edges is not None:
        edges = cached_edges
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if use_tophat:
            gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _kernel_rect_31())
        if blur_ksize and blur_ksize % 2 == 1:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        edges = cv2.Canny(gray, canny_low, canny_high)
    edges_on_ground = cv2.bitwise_and(edges, edges, mask=ground_mask)
    if dilate_kernel_size > 1:
        if int(dilate_kernel_size) == 3:
            dilate_kernel = _kernel_rect_3()
        else:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
        edges_on_ground = cv2.dilate(edges_on_ground, dilate_kernel, iterations=dilate_iterations)
    return (edges_on_ground > 0).astype(np.uint8)

def compute_frame_canny_edges(frame_bgr: np.ndarray, *, blur_ksize: int=5, canny_low: int=30, canny_high: int=100, use_tophat: bool=True) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if use_tophat:
        try:
            kernel = _kernel_rect_31()
        except Exception:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        gray = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    if blur_ksize and blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return cv2.Canny(gray, canny_low, canny_high)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
_CURRENT_PROGRESS: str = ''
TV_AF_EVAL_PARALLEL: bool = True
# Thread pool size for adding_four_points and group eval. Uses threads (not processes), so CPU
# utilization is often 40–60% due to Python's GIL; NumPy/OpenCV can release GIL and use more.
# Default: CPU count (capped at 32). Override with env TV_AF_EVAL_MAX_WORKERS if desired.
def _tv_af_eval_max_workers() -> int:
    env = os.environ.get('TV_AF_EVAL_MAX_WORKERS', '')
    if env.strip().isdigit():
        return max(1, min(64, int(env)))
    return min(32, (os.cpu_count() or 8))
TV_AF_EVAL_MAX_WORKERS: int = _tv_af_eval_max_workers()

def _kp_prof_enabled() -> bool:
    return bool(TV_KP_PROFILE)

class _KPProfiler:

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._t0 = time.perf_counter() if self.enabled else 0.0
        self._frame_count = 0
        self.totals_ms: dict[str, float] = {}

    def _add(self, key: str, ms: float) -> None:
        self.totals_ms[key] = self.totals_ms.get(key, 0.0) + float(ms)

    def begin_frame(self) -> float:
        if not self.enabled:
            return 0.0
        self._frame_count += 1
        return time.perf_counter()

    def end_frame(self, *, frame_id: int, t_frame0: float, parts_ms: dict[str, float]) -> None:
        if not self.enabled:
            return
        total_ms = (time.perf_counter() - t_frame0) * 1000.0
        self._add('frame_total', total_ms)
        for k, v in parts_ms.items():
            self._add(k, v)
    def summary(self, *, label: str) -> None:
        if not self.enabled:
            return
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        frames = max(1, self._frame_count)
        print(f'[tv][kp_profile] {label} frames={self._frame_count} elapsed_ms={elapsed_ms:.2f} avg_frame_ms={self.totals_ms.get("frame_total", 0.0) / frames:.2f}')
        tot = self.totals_ms.get('step5_fallback_add4', 0.0)
        avg = tot / frames
        print(f'[tv][kp_profile] {label} total_step5_fallback_add4_ms={tot:.2f} avg_step5_fallback_add4_ms={avg:.2f}')

def _step4_1_compute_border_line(*, frame: np.ndarray, ordered_kps: list[list[float]], frame_number: int, cached_edges: np.ndarray | None=None) -> dict[str, Any]:
    result: dict[str, Any] = {'frame_number': int(frame_number), 'has_h': False, 'p5': None, 'p29': None, 'Ey': None, 'Fy': None, 'best_aay': None, 'best_bby': None, 'best_white_rate': None, 'kkp5': None, 'kkp29': None}
    try:
        H_img, W_img = frame.shape[:2]
        filtered_src: list[tuple[float, float]] = []
        filtered_dst: list[tuple[float, float]] = []
        for src_pt, kp in zip(FOOTBALL_KEYPOINTS_CORRECTED, ordered_kps, strict=True):
            if not kp or len(kp) < 2:
                continue
            dx = float(kp[0])
            dy = float(kp[1])
            if abs(dx) < 1e-06 and abs(dy) < 1e-06:
                continue
            filtered_src.append((float(src_pt[0]), float(src_pt[1])))
            filtered_dst.append((dx, dy))
        if len(filtered_src) < 4:
            return result
        H_mat, _ = cv2.findHomography(np.array(filtered_src, dtype=np.float32), np.array(filtered_dst, dtype=np.float32))
        if H_mat is None:
            return result
        result['has_h'] = True
        result['H'] = H_mat.tolist()
        src_5_29 = np.array([[FOOTBALL_KEYPOINTS_CORRECTED[5]], [FOOTBALL_KEYPOINTS_CORRECTED[29]]], dtype=np.float32)
        dst_5_29 = cv2.perspectiveTransform(src_5_29, H_mat).reshape(-1, 2)
        x1, y1 = (float(dst_5_29[0][0]), float(dst_5_29[0][1]))
        x2, y2 = (float(dst_5_29[1][0]), float(dst_5_29[1][1]))
        result['p5'] = [int(round(x1)), int(round(y1))]
        result['p29'] = [int(round(x2)), int(round(y2))]
        dx_line = x2 - x1
        dy_line = y2 - y1
        if abs(dx_line) < 1e-09:
            result['skipped'] = 'red line vertical'
            return result
        Ey = y1 + (0.0 - x1) * (dy_line / dx_line)
        Fy = y1 + (float(W_img - 1) - x1) * (dy_line / dx_line)
        result['Ey'] = float(Ey)
        result['Fy'] = float(Fy)
        red_line_y_min = min(Ey, Fy)
        red_line_y_max = max(Ey, Fy)
        if red_line_y_max < 0 or red_line_y_min > H_img - 1:
            result['skipped'] = 'red line outside image range'
            return result
        if cached_edges is not None:
            edges = cached_edges
        else:
            edges = compute_frame_canny_edges(frame)
        try:
            dil_k = _kernel_rect_3()
        except Exception:
            dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, dil_k, iterations=3)
        edge_only = None
        edge_only_dilated = None
        LINE_WIDTH_PX_41 = 10
        half_width_41 = LINE_WIDTH_PX_41 // 2
        LINE_SAMPLE_MAX_41 = 256
        edges_dilated_flat = edges_dilated.ravel() if _sloping_line_white_count_cy is not None else None

        def _border_line_white_rate(y_left: float, y_right: float) -> float:
            ax, ay = (0.0, y_left)
            bx, by = (float(W_img - 1), y_right)
            if _sloping_line_white_count_cy is not None and edges_dilated_flat is not None:
                try:
                    white, total = _sloping_line_white_count_cy(edges_dilated_flat, W_img, H_img, ax, ay, bx, by, half_width_41, LINE_SAMPLE_MAX_41)
                    return white / total if total > 0 else 0.0
                except Exception:
                    pass
            L = float(np.hypot(bx - ax, by - ay))
            if L < 1.0:
                return 0.0
            n_steps = min(max(1, int(L) + 1), LINE_SAMPLE_MAX_41)
            t = np.linspace(0, 1, n_steps, dtype=np.float64)
            px = ax + t * (bx - ax)
            py = ay + t * (by - ay)
            perp_x = -(float(by) - float(ay)) / L
            perp_y = (float(bx) - float(ax)) / L
            qx_list: list[np.ndarray] = []
            qy_list: list[np.ndarray] = []
            for k in range(-half_width_41, half_width_41 + 1):
                qx_list.append(np.round(px + k * perp_x).astype(np.int32))
                qy_list.append(np.round(py + k * perp_y).astype(np.int32))
            qx_flat = np.concatenate(qx_list)
            qy_flat = np.concatenate(qy_list)
            valid = (qx_flat >= 0) & (qx_flat < W_img) & (qy_flat >= 0) & (qy_flat < H_img)
            qx_flat = qx_flat[valid]
            qy_flat = qy_flat[valid]
            if qx_flat.size == 0:
                return 0.0
            linear = np.unique(qy_flat.astype(np.int64) * W_img + qx_flat.astype(np.int64))
            qy_u = linear // W_img
            qx_u = linear % W_img
            total = qx_u.size
            white = int(np.count_nonzero(edges_dilated[qy_u, qx_u]))
            return white / total if total else 0.0
        a_min = int(np.floor(Ey - 100.0))
        a_max = int(np.ceil(Ey + 100.0))
        b_min = int(np.floor(Fy - 100.0))
        b_max = int(np.ceil(Fy + 100.0))
        best_rate = -1.0
        best_aay: int | None = None
        best_bby: int | None = None
        STEP_COARSE_41 = 4
        REFINE_RADIUS_41 = 4
        STEP_FINE_41 = 2
        GREEN_RATE_EARLY_EXIT_41 = 0.95
        done = False
        for aay in range(a_min, a_max + 1, STEP_COARSE_41):
            if done:
                break
            for bby in range(b_min, b_max + 1, STEP_COARSE_41):
                rate = _border_line_white_rate(float(aay), float(bby))
                if rate > best_rate:
                    best_rate = rate
                    best_aay = aay
                    best_bby = bby
                    if best_rate >= GREEN_RATE_EARLY_EXIT_41:
                        done = True
                        break
        if best_aay is not None and best_bby is not None and (not done):
            a_ref_min = max(a_min, best_aay - REFINE_RADIUS_41)
            a_ref_max = min(a_max, best_aay + REFINE_RADIUS_41)
            b_ref_min = max(b_min, best_bby - REFINE_RADIUS_41)
            b_ref_max = min(b_max, best_bby + REFINE_RADIUS_41)
            for aay in range(a_ref_min, a_ref_max + 1, STEP_FINE_41):
                if done:
                    break
                for bby in range(b_ref_min, b_ref_max + 1, STEP_FINE_41):
                    rate = _border_line_white_rate(float(aay), float(bby))
                    if rate > best_rate:
                        best_rate = rate
                        best_aay = aay
                        best_bby = bby
                        if best_rate >= GREEN_RATE_EARLY_EXIT_41:
                            done = True
                            break
        if best_aay is None or best_bby is None:
            best_aay = int(round(Ey))
            best_bby = int(round(Fy))
            if best_rate < 0.0:
                best_rate = _border_line_white_rate(float(best_aay), float(best_bby))
        if best_aay is not None and best_bby is not None:
            pA = (0, best_aay)
            pB = (W_img - 1, best_bby)
            green_thickness = 3
            result['best_aay'] = best_aay
            result['best_bby'] = best_bby
            result['best_white_rate'] = best_rate

            def _line_line_intersection_41(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> tuple[float, float] | None:
                v1x = bx - ax
                v1y = by - ay
                v2x = dx - cx
                v2y = dy - cy
                det = v1x * v2y - v1y * v2x
                if abs(det) < 1e-12:
                    return None
                t = ((cx - ax) * v2y - (cy - ay) * v2x) / det
                return (ax + t * v1x, ay + t * v1y)
            green_ax, green_ay = (0.0, float(best_aay))
            green_bx, green_by = (float(W_img - 1), float(best_bby))
            src_0_4_24_28 = np.array([[FOOTBALL_KEYPOINTS_CORRECTED[0]], [FOOTBALL_KEYPOINTS_CORRECTED[4]], [FOOTBALL_KEYPOINTS_CORRECTED[24]], [FOOTBALL_KEYPOINTS_CORRECTED[28]]], dtype=np.float32)
            proj_0_4_24_28 = cv2.perspectiveTransform(src_0_4_24_28, H_mat).reshape(-1, 2)
            cx0, cy0 = (float(proj_0_4_24_28[0][0]), float(proj_0_4_24_28[0][1]))
            cx4, cy4 = (float(proj_0_4_24_28[1][0]), float(proj_0_4_24_28[1][1]))
            cx24, cy24 = (float(proj_0_4_24_28[2][0]), float(proj_0_4_24_28[2][1]))
            cx28, cy28 = (float(proj_0_4_24_28[3][0]), float(proj_0_4_24_28[3][1]))
            if abs(cx4 - cx0) >= 1e-09 or abs(cy4 - cy0) >= 1e-09:
                pt = _line_line_intersection_41(green_ax, green_ay, green_bx, green_by, cx0, cy0, cx4, cy4)
                if pt is not None:
                    result['kkp5'] = [float(pt[0]), float(pt[1])]
            if abs(cx28 - cx24) >= 1e-09 or abs(cy28 - cy24) >= 1e-09:
                pt = _line_line_intersection_41(green_ax, green_ay, green_bx, green_by, cx24, cy24, cx28, cy28)
                if pt is not None:
                    result['kkp29'] = [float(pt[0]), float(pt[1])]
    except Exception as exc:
        logger.error('Failed Step 4.1 border line compute for frame %s: %s', frame_number, exc)
        result['error'] = str(exc)
    return result

def _step4_9_select_h_and_keypoints(*, input_kps: list[list[float]], step4_1: dict[str, Any] | None, step4_3: dict[str, Any] | None, frame: np.ndarray, frame_number: int, decision: str | None, cached_edges: np.ndarray | None=None) -> tuple[list[list[float]] | None, float]:
    H_img, W_img = frame.shape[:2]
    template = FOOTBALL_KEYPOINTS_CORRECTED
    n_tpl = len(template)
    if not input_kps or len(input_kps) < n_tpl:
        return (None, 0.0)

    def _build_weighted_correspondences(kps: list[list[float]], weight_43: int, weight_41: int, use_right: bool) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        src_list: list[tuple[float, float]] = []
        dst_list: list[tuple[float, float]] = []
        idx_43 = (4, 12) if not use_right else (28, 20)
        idx_41 = (5, 29)
        for i in range(n_tpl):
            if i >= len(kps) or not kps[i] or len(kps[i]) < 2:
                continue
            x, y = (float(kps[i][0]), float(kps[i][1]))
            if abs(x) < 1e-06 and abs(y) < 1e-06:
                continue
            src_pt = (float(template[i][0]), float(template[i][1]))
            dst_pt = (x, y)
            w = 1
            if i in idx_43:
                w = weight_43
            elif i in idx_41:
                w = weight_41
            for _ in range(w):
                src_list.append(src_pt)
                dst_list.append(dst_pt)
        return (src_list, dst_list)

    def _homography_from_kps(kps: list[list[float]]) -> np.ndarray | None:
        filtered_src: list[tuple[float, float]] = []
        filtered_dst: list[tuple[float, float]] = []
        for i in range(min(n_tpl, len(kps))):
            if not kps[i] or len(kps[i]) < 2:
                continue
            x, y = (float(kps[i][0]), float(kps[i][1]))
            if abs(x) < 1e-06 and abs(y) < 1e-06:
                continue
            filtered_src.append((float(template[i][0]), float(template[i][1])))
            filtered_dst.append((x, y))
        if len(filtered_src) < 4:
            return None
        H, _ = cv2.findHomography(np.array(filtered_src, dtype=np.float32), np.array(filtered_dst, dtype=np.float32))
        return H

    def _score_h(H: np.ndarray | None) -> float:
        if H is None:
            return 0.0
        try:
            tpl_pts = np.array([[float(p[0]), float(p[1])] for p in template], dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(tpl_pts, H).reshape(-1, 2)
            frame_kps_tuples = [(float(proj[i][0]), float(proj[i][1])) for i in range(len(proj))]
            template_image = challenge_template()
            return evaluate_keypoints_for_frame(template_keypoints=template, frame_keypoints=frame_kps_tuples, frame=frame, floor_markings_template=template_image, frame_number=frame_number, cached_edges=cached_edges, processing_scale=PROCESSING_SCALE)
        except Exception:
            return 0.0

    def _project_and_valid_only(H: np.ndarray) -> list[list[float]]:
        tpl_pts = np.array([[float(p[0]), float(p[1])] for p in template], dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(tpl_pts, H).reshape(-1, 2)
        out: list[list[float]] = [[0.0, 0.0]] * n_tpl
        for i in range(n_tpl):
            x, y = (float(proj[i][0]), float(proj[i][1]))
            if 0 <= x < W_img and 0 <= y < H_img:
                out[i] = [x, y]
        return out
    use_right = decision == 'right'
    kps1 = [list(kp) if kp else [0.0, 0.0] for kp in input_kps[:n_tpl]]
    while len(kps1) < n_tpl:
        kps1.append([0.0, 0.0])
    kps2 = [list(kp) if kp else [0.0, 0.0] for kp in kps1]
    if step4_3:
        if not use_right:
            if step4_3.get('kkp4') is not None:
                kps2[4] = list(step4_3['kkp4'])
            if step4_3.get('kkp12') is not None:
                kps2[12] = list(step4_3['kkp12'])
        else:
            if step4_3.get('kkp28') is not None:
                kps2[28] = list(step4_3['kkp28'])
            if step4_3.get('kkp20') is not None:
                kps2[20] = list(step4_3['kkp20'])
    kps3 = [list(kp) if kp else [0.0, 0.0] for kp in kps2]
    if step4_1:
        if step4_1.get('kkp5') is not None:
            kps3[5] = list(step4_1['kkp5'])
        if step4_1.get('kkp29') is not None:
            kps3[29] = list(step4_1['kkp29'])
    H1 = None
    if step4_1 and step4_1.get('H') is not None:
        try:
            H1 = np.array(step4_1['H'], dtype=np.float32)
        except (TypeError, ValueError):
            H1 = _homography_from_kps(kps1)
    if H1 is None:
        H1 = _homography_from_kps(kps1)
    src2, dst2 = _build_weighted_correspondences(kps2, weight_43=4, weight_41=1, use_right=use_right)
    H2 = None
    if len(src2) >= 4:
        H2, _ = cv2.findHomography(np.array(src2, dtype=np.float32), np.array(dst2, dtype=np.float32))
    src3, dst3 = _build_weighted_correspondences(kps3, weight_43=4, weight_41=8, use_right=use_right)
    H3 = None
    if len(src3) >= 4:
        H3, _ = cv2.findHomography(np.array(src3, dtype=np.float32), np.array(dst3, dtype=np.float32))
    if TV_AF_EVAL_PARALLEL:
        with ThreadPoolExecutor(max_workers=3) as ex:
            f1 = ex.submit(_score_h, H1)
            f2 = ex.submit(_score_h, H2)
            f3 = ex.submit(_score_h, H3)
            s1, s2, s3 = (f1.result(), f2.result(), f3.result())
    else:
        s1 = _score_h(H1)
        s2 = _score_h(H2)
        s3 = _score_h(H3)
    best_score = max(s1, s2, s3)
    if best_score <= 0.0:
        return (None, 0.0)
    if H1 is None and H2 is None and (H3 is None):
        return (None, 0.0)
    best_H: np.ndarray | None = None
    best_label = ''
    if best_score == s1 and H1 is not None:
        best_H = H1
        best_label = 'H1'
    elif best_score == s2 and H2 is not None:
        best_H = H2
        best_label = 'H2'
    elif best_score == s3 and H3 is not None:
        best_H = H3
        best_label = 'H3'
    if best_H is None:
        return (None, 0.0)
    out_kps = _project_and_valid_only(best_H)
    return (out_kps, best_score)

def _step4_3_debug_dilate_and_lines(*, frame: np.ndarray, ordered_kps: list[list[float]], frame_number: int, decision: str | None=None, cached_edges: np.ndarray | None=None) -> dict[str, Any]:
    result: dict[str, Any] = {'frame_number': int(frame_number), 'did_line_draw': False, 'kkp4': None, 'kkp12': None, 'kkp28': None, 'kkp20': None}
    try:
        H_img, W_img = frame.shape[:2]
        template_len = len(FOOTBALL_KEYPOINTS_CORRECTED)
        if not ordered_kps or len(ordered_kps) < template_len:
            return result
        use_right = decision == 'right'
        bottom_idx = 20 if use_right else 12
        idx_A = 28 if use_right else 4
        idx_C = 17 if use_right else 9
        idx_top = 25 if use_right else 1
        side_label = 'right' if use_right else 'left'
        max_idx_required = 28 if use_right else 12
        if len(ordered_kps) <= max_idx_required:
            return result
        if cached_edges is not None:
            edges = cached_edges
        else:
            edges = compute_frame_canny_edges(frame)
        try:
            dil_k = _kernel_rect_3()
        except Exception:
            dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, dil_k, iterations=3)
        edge_dilated_bgr = None
        filtered_src: list[tuple[float, float]] = []
        filtered_dst: list[tuple[float, float]] = []
        for src_pt, kp in zip(FOOTBALL_KEYPOINTS_CORRECTED, ordered_kps, strict=True):
            if not kp or len(kp) < 2:
                continue
            dx, dy = (float(kp[0]), float(kp[1]))
            if abs(dx) < 1e-06 and abs(dy) < 1e-06:
                continue
            filtered_src.append((float(src_pt[0]), float(src_pt[1])))
            filtered_dst.append((dx, dy))
        if len(filtered_src) < 4:
            return result
        H_mat, _ = cv2.findHomography(np.array(filtered_src, dtype=np.float32), np.array(filtered_dst, dtype=np.float32))
        if H_mat is None:
            return result
        all_proj = cv2.perspectiveTransform(np.array([[float(p[0]), float(p[1])] for p in FOOTBALL_KEYPOINTS_CORRECTED], dtype=np.float32).reshape(-1, 1, 2), H_mat)
        all_proj_list = [[float(all_proj[i][0][0]), float(all_proj[i][0][1])] for i in range(len(all_proj))]

        def _red_line_pos(idx: int):
            if idx < len(ordered_kps):
                kp = ordered_kps[idx]
                if kp and len(kp) >= 2 and (not (abs(kp[0]) < 1e-06 and abs(kp[1]) < 1e-06)):
                    return (int(round(float(kp[0]))), int(round(float(kp[1]))))
            if idx < len(all_proj_list):
                p = all_proj_list[idx]
                if abs(p[0]) >= 1e-06 or abs(p[1]) >= 1e-06:
                    return (int(round(p[0])), int(round(p[1])))
            return None
        pt_A = _red_line_pos(idx_A)
        if pt_A is None and idx_A < len(all_proj_list):
            p = all_proj_list[idx_A]
            pt_A = (int(round(p[0])), int(round(p[1])))
        pt_bottom = _red_line_pos(bottom_idx)
        if pt_bottom is None and bottom_idx < len(all_proj_list):
            p = all_proj_list[bottom_idx]
            pt_bottom = (int(round(p[0])), int(round(p[1])))
        pt_C = _red_line_pos(idx_C)
        if pt_C is None and idx_C < len(all_proj_list):
            p = all_proj_list[idx_C]
            pt_C = (int(round(p[0])), int(round(p[1])))
        if pt_A is None or pt_bottom is None or pt_C is None:
            return result
        x_A, y_A = (pt_A[0], pt_A[1])
        x_bottom, y_bottom = (pt_bottom[0], pt_bottom[1])
        x_C, y_C = (pt_C[0], pt_C[1])
        pt_top = _red_line_pos(idx_top)
        LINE_WIDTH_PX = 10
        STEP_COARSE = 8
        REFINE_RADIUS = 2
        LINE_SAMPLE_MAX = 128
        edges_dilated_flat = edges_dilated.ravel() if _sloping_line_white_count_cy is not None else None

        def _line_segment_white_rate(ax: int, ay: int, bx: int, by: int, half_width: int=5) -> float:
            if _sloping_line_white_count_cy is not None and edges_dilated_flat is not None:
                try:
                    white, total = _sloping_line_white_count_cy(edges_dilated_flat, W_img, H_img, float(ax), float(ay), float(bx), float(by), half_width, LINE_SAMPLE_MAX)
                    return white / total if total > 0 else 0.0
                except Exception:
                    pass
            L = float(np.hypot(float(bx - ax), float(by - ay)))
            if L < 1.0:
                return 0.0
            n_steps = min(max(1, int(L) + 1), LINE_SAMPLE_MAX)
            t = np.linspace(0, 1, n_steps, dtype=np.float64)
            px = ax + t * (bx - ax)
            py = ay + t * (by - ay)
            perp_x = -(float(by) - float(ay)) / L
            perp_y = (float(bx) - float(ax)) / L
            qx_list: list[np.ndarray] = []
            qy_list: list[np.ndarray] = []
            for k in range(-half_width, half_width + 1):
                qx_list.append(np.round(px + k * perp_x).astype(np.int32))
                qy_list.append(np.round(py + k * perp_y).astype(np.int32))
            qx_flat = np.concatenate(qx_list)
            qy_flat = np.concatenate(qy_list)
            valid = (qx_flat >= 0) & (qx_flat < W_img) & (qy_flat >= 0) & (qy_flat < H_img)
            qx_flat = qx_flat[valid]
            qy_flat = qy_flat[valid]
            if qx_flat.size == 0:
                return 0.0
            linear = np.unique(qy_flat.astype(np.int64) * W_img + qx_flat.astype(np.int64))
            qy_u = linear // W_img
            qx_u = linear % W_img
            total = qx_u.size
            white = int(np.count_nonzero(edges_dilated[qy_u, qx_u]))
            return white / total if total else 0.0
        half_width = LINE_WIDTH_PX // 2
        a_y_min = max(0, y_A - 30)
        a_y_max = min(H_img - 1, y_A + 30)
        b_y_min = max(0, y_bottom - 30)
        b_y_max = min(H_img - 1, y_bottom + 30)
        best_rate = -1.0
        best_a_y = y_A
        best_b_y = y_bottom
        coarse_candidates = 0
        a_y_coarse = list(range(a_y_min, a_y_max + 1, STEP_COARSE))
        if a_y_coarse and a_y_coarse[-1] != a_y_max:
            a_y_coarse.append(a_y_max)
        b_y_coarse = list(range(b_y_min, b_y_max + 1, STEP_COARSE))
        if b_y_coarse and b_y_coarse[-1] != b_y_max:
            b_y_coarse.append(b_y_max)
        for a_y in a_y_coarse:
            for b_y in b_y_coarse:
                coarse_candidates += 1
                rate = _line_segment_white_rate(x_A, a_y, x_bottom, b_y, half_width=half_width)
                if rate > best_rate:
                    best_rate = rate
                    best_a_y = a_y
                    best_b_y = b_y
        a_ref_min = max(a_y_min, best_a_y - REFINE_RADIUS)
        a_ref_max = min(a_y_max, best_a_y + REFINE_RADIUS)
        b_ref_min = max(b_y_min, best_b_y - REFINE_RADIUS)
        b_ref_max = min(b_y_max, best_b_y + REFINE_RADIUS)
        for a_y in range(a_ref_min, a_ref_max + 1):
            for b_y in range(b_ref_min, b_ref_max + 1):
                rate = _line_segment_white_rate(x_A, a_y, x_bottom, b_y, half_width=half_width)
                if rate > best_rate:
                    best_rate = rate
                    best_a_y = a_y
                    best_b_y = b_y
        best_a = (x_A, best_a_y)
        best_b = (x_bottom, best_b_y)
        c_y_min = max(0, y_C - 30)
        c_y_max = min(H_img - 1, y_C + 30)
        d_y_min = max(0, y_bottom - 30)
        d_y_max = min(H_img - 1, y_bottom + 30)
        c_y_coarse = list(range(c_y_min, c_y_max + 1, STEP_COARSE))
        if c_y_coarse and c_y_coarse[-1] != c_y_max:
            c_y_coarse.append(c_y_max)
        d_y_coarse = list(range(d_y_min, d_y_max + 1, STEP_COARSE))
        if d_y_coarse and d_y_coarse[-1] != d_y_max:
            d_y_coarse.append(d_y_max)
        best_cd_rate = -1.0
        best_c_y = y_C
        best_d_y = y_bottom
        cd_coarse_candidates = 0
        for c_y in c_y_coarse:
            for d_y in d_y_coarse:
                cd_coarse_candidates += 1
                rate = _line_segment_white_rate(x_C, c_y, x_bottom, d_y, half_width=half_width)
                if rate > best_cd_rate:
                    best_cd_rate = rate
                    best_c_y = c_y
                    best_d_y = d_y
        c_ref_min = max(c_y_min, best_c_y - REFINE_RADIUS)
        c_ref_max = min(c_y_max, best_c_y + REFINE_RADIUS)
        d_ref_min = max(d_y_min, best_d_y - REFINE_RADIUS)
        d_ref_max = min(d_y_max, best_d_y + REFINE_RADIUS)
        cd_refine_candidates = (c_ref_max - c_ref_min + 1) * (d_ref_max - d_ref_min + 1)
        for c_y in range(c_ref_min, c_ref_max + 1):
            for d_y in range(d_ref_min, d_ref_max + 1):
                rate = _line_segment_white_rate(x_C, c_y, x_bottom, d_y, half_width=half_width)
                if rate > best_cd_rate:
                    best_cd_rate = rate
                    best_c_y = c_y
                    best_d_y = d_y
        best_c = (x_C, best_c_y)
        best_d = (x_bottom, best_d_y)
        def _line_line_intersection(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> tuple[float, float] | None:
            v1x, v1y = (bx - ax, by - ay)
            v2x, v2y = (dx - cx, dy - cy)
            det = v1x * v2y - v1y * v2x
            if abs(det) < 1e-10:
                return None
            t = ((cx - ax) * v2y - (cy - ay) * v2x) / det
            return (float(ax + t * v1x), float(ay + t * v1y))
        ax, ay = (float(best_a[0]), float(best_a[1]))
        bx, by = (float(best_b[0]), float(best_b[1]))
        cx, cy = (float(best_c[0]), float(best_c[1]))
        dx, dy = (float(best_d[0]), float(best_d[1]))
        idx_E = 0 if not use_right else 24
        idx_F = 5 if not use_right else 29
        if idx_E < len(all_proj_list) and idx_F < len(all_proj_list):
            p_E = all_proj_list[idx_E]
            p_F = all_proj_list[idx_F]
            x_E, y_E = (float(p_E[0]), float(p_E[1]))
            x_F, y_F = (float(p_F[0]), float(p_F[1]))
            pt_E = (int(round(x_E)), int(round(y_E)))
            pt_F = (int(round(x_F)), int(round(y_F)))
            x_E_int, y_E_int = (int(round(x_E)), int(round(y_E)))
            x_F_int, y_F_int = (int(round(x_F)), int(round(y_F)))
            ef_a_y_min = y_E_int - 30
            ef_a_y_max = y_E_int + 30
            ef_b_y_min = y_F_int - 30
            ef_b_y_max = y_F_int + 30
            best_ef_rate = -1.0
            best_ef_a_y = y_E_int
            best_ef_b_y = y_F_int
            if abs(x_F_int - x_E_int) >= 1 or abs(y_F_int - y_E_int) >= 1:
                ef_a_y_coarse = list(range(ef_a_y_min, ef_a_y_max + 1, STEP_COARSE))
                if ef_a_y_coarse and ef_a_y_coarse[-1] != ef_a_y_max:
                    ef_a_y_coarse.append(ef_a_y_max)
                ef_b_y_coarse = list(range(ef_b_y_min, ef_b_y_max + 1, STEP_COARSE))
                if ef_b_y_coarse and ef_b_y_coarse[-1] != ef_b_y_max:
                    ef_b_y_coarse.append(ef_b_y_max)
                for ef_a_y in ef_a_y_coarse:
                    for ef_b_y in ef_b_y_coarse:
                        rate = _line_segment_white_rate(x_E_int, ef_a_y, x_F_int, ef_b_y, half_width=half_width)
                        if rate > best_ef_rate:
                            best_ef_rate = rate
                            best_ef_a_y = ef_a_y
                            best_ef_b_y = ef_b_y
                ef_ref_min_a = max(ef_a_y_min, best_ef_a_y - REFINE_RADIUS)
                ef_ref_max_a = min(ef_a_y_max, best_ef_a_y + REFINE_RADIUS)
                ef_ref_min_b = max(ef_b_y_min, best_ef_b_y - REFINE_RADIUS)
                ef_ref_max_b = min(ef_b_y_max, best_ef_b_y + REFINE_RADIUS)
                for ef_a_y in range(ef_ref_min_a, ef_ref_max_a + 1):
                    for ef_b_y in range(ef_ref_min_b, ef_ref_max_b + 1):
                        rate = _line_segment_white_rate(x_E_int, ef_a_y, x_F_int, ef_b_y, half_width=half_width)
                        if rate > best_ef_rate:
                            best_ef_rate = rate
                            best_ef_a_y = ef_a_y
                            best_ef_b_y = ef_b_y
            best_ef_a = (x_E_int, best_ef_a_y)
            best_ef_b = (x_F_int, best_ef_b_y)
            ef_ax, ef_ay = (float(best_ef_a[0]), float(best_ef_a[1]))
            ef_bx, ef_by = (float(best_ef_b[0]), float(best_ef_b[1]))
            pt_ef_ab = _line_line_intersection(ef_ax, ef_ay, ef_bx, ef_by, ax, ay, bx, by)
            if pt_ef_ab is not None:
                if use_right:
                    result['kkp28'] = [pt_ef_ab[0], pt_ef_ab[1]]
                else:
                    result['kkp4'] = [pt_ef_ab[0], pt_ef_ab[1]]
        pt_ab_cd = _line_line_intersection(ax, ay, bx, by, cx, cy, dx, dy)
        if pt_ab_cd is not None:
            if use_right:
                result['kkp20'] = [pt_ab_cd[0], pt_ab_cd[1]]
            else:
                result['kkp12'] = [pt_ab_cd[0], pt_ab_cd[1]]
        result['did_line_draw'] = True
    except Exception as e:
        logger.error('Step 4.3 failed for frame %s: %s', frame_number, e)
        result['error'] = str(e)
    return result

class InvalidMask(Exception):
    pass

def has_a_wide_line(mask: np.ndarray, max_aspect_ratio: float=1.0, debug_frame_id: int | None=None) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        aspect_ratio = min(w, h) / max(w, h)
        if aspect_ratio >= max_aspect_ratio:
            return True
    return False

def validate_mask_lines(mask: np.ndarray, debug_frame_id: int | None=None) -> None:
    if mask.sum() == 0:
        raise InvalidMask('No projected lines')
    if mask.sum() == mask.size:
        raise InvalidMask('Projected lines cover the entire image surface')
    if has_a_wide_line(mask=mask, debug_frame_id=debug_frame_id):
        raise InvalidMask('A projected line is too wide')

def validate_mask_ground(mask: np.ndarray) -> None:
    if cv2.countNonZero(mask) == 0:
        raise InvalidMask('No projected ground (empty mask)')
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) == 0:
        raise InvalidMask('No projected ground (empty mask)')
    x, y, w, h = cv2.boundingRect(pts)
    is_rect = cv2.countNonZero(mask) == w * h
    if is_rect:
        raise InvalidMask('Projected ground should not be rectangular')
    num_labels, _ = cv2.connectedComponents(mask)
    num_distinct_regions = num_labels - 1
    if num_distinct_regions > 1:
        raise InvalidMask(f'Projected ground should be a single object, detected {num_distinct_regions}')
    area_covered = mask.sum() / mask.size
    if area_covered >= 0.9:
        raise InvalidMask(f'Projected ground covers more than {area_covered:.2f}% of the image surface which is unrealistic')

def _is_bowtie(points: np.ndarray) -> bool:

    def segments_intersect(p1: tuple[float, float], p2: tuple[float, float], q1: tuple[float, float], q2: tuple[float, float]) -> bool:

        def ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
    pts = points.reshape(-1, 2)
    if len(pts) < 4:
        return False
    edges = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0])]
    return segments_intersect(*edges[0], *edges[2]) or segments_intersect(*edges[1], *edges[3])

def validate_projected_corners(source_keypoints: list[tuple[int, int]], homography_matrix: np.ndarray) -> None:
    if len(source_keypoints) <= max(INDEX_KEYPOINT_CORNER_BOTTOM_LEFT, INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT, INDEX_KEYPOINT_CORNER_TOP_LEFT, INDEX_KEYPOINT_CORNER_TOP_RIGHT):
        return
    src_corners = np.array([source_keypoints[INDEX_KEYPOINT_CORNER_BOTTOM_LEFT], source_keypoints[INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT], source_keypoints[INDEX_KEYPOINT_CORNER_TOP_RIGHT], source_keypoints[INDEX_KEYPOINT_CORNER_TOP_LEFT]], dtype=np.float32)[None, :, :]
    warped_corners = cv2.perspectiveTransform(src_corners, homography_matrix)[0]
    if _is_bowtie(warped_corners):
        raise InvalidMask('Projection twisted!')

def project_image_using_keypoints(image: np.ndarray, source_keypoints: list[tuple[int, int]], destination_keypoints: list[tuple[float, float]], destination_width: int, destination_height: int, inverse: bool=False, return_h: bool=False) -> np.ndarray:
    filtered_src: list[tuple[int, int]] = []
    filtered_dst: list[tuple[float, float]] = []
    for src_pt, dst_pt in zip(source_keypoints, destination_keypoints, strict=True):
        if dst_pt[0] == 0.0 and dst_pt[1] == 0.0:
            continue
        filtered_src.append(src_pt)
        filtered_dst.append(dst_pt)
    if len(filtered_src) < 4:
        raise ValueError('At least 4 valid keypoints are required for homography.')
    source_points = np.array(filtered_src, dtype=np.float32)
    destination_points = np.array(filtered_dst, dtype=np.float32)
    if inverse:
        H_inv, _ = cv2.findHomography(destination_points, source_points)
        projected = cv2.warpPerspective(image, H_inv, (destination_width, destination_height))
        if return_h:
            return (projected, H_inv)
        return projected
    H, _ = cv2.findHomography(source_points, destination_points)
    if H is None:
        raise ValueError('Homography computation failed')
    projected_image = cv2.warpPerspective(image, H, (destination_width, destination_height))
    try:
        validate_projected_corners(source_keypoints=source_keypoints, homography_matrix=H)
    except InvalidMask:
        raise
    if return_h:
        return (projected_image, H)
    return projected_image

def extract_masks_for_ground_and_lines(image: np.ndarray, debug_frame_id: int | None=None) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_ground = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    _, mask_lines = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask_ground_binary = (mask_ground > 0).astype(np.uint8)
    mask_lines_binary = (mask_lines > 0).astype(np.uint8)
    validate_mask_ground(mask=mask_ground_binary)
    validate_mask_lines(mask=mask_lines_binary, debug_frame_id=debug_frame_id)
    return (mask_ground_binary, mask_lines_binary)

def _homography_from_keypoints(source_keypoints: list[tuple[int, int]], destination_keypoints: list[tuple[float, float]]) -> np.ndarray:
    filtered_src: list[tuple[int, int]] = []
    filtered_dst: list[tuple[float, float]] = []
    for src_pt, dst_pt in zip(source_keypoints, destination_keypoints, strict=True):
        if dst_pt[0] == 0.0 and dst_pt[1] == 0.0:
            continue
        filtered_src.append(src_pt)
        filtered_dst.append(dst_pt)
    if len(filtered_src) < 4:
        raise ValueError('At least 4 valid keypoints are required for homography.')
    source_points = np.array(filtered_src, dtype=np.float32)
    destination_points = np.array(filtered_dst, dtype=np.float32)
    H, _ = cv2.findHomography(source_points, destination_points)
    if H is None:
        raise ValueError('Homography computation failed')
    validate_projected_corners(source_keypoints=source_keypoints, homography_matrix=H)
    return H

def _polygon_masks_from_homography(homography_matrix: np.ndarray, frame_shape: tuple[int, int], polygon_def: dict[str, list[list[tuple[float, float]]]], *, debug_frame_id: int | None=None) -> tuple[np.ndarray, np.ndarray]:
    frame_height, frame_width = frame_shape
    mask_ground = np.zeros((frame_height, frame_width), dtype=np.uint8)
    mask_lines = np.zeros((frame_height, frame_width), dtype=np.uint8)

    def _warp_poly(poly: list[tuple[float, float]]) -> np.ndarray | None:
        if not poly or len(poly) < 3:
            return None
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, homography_matrix)
        if not np.isfinite(warped).all():
            return None
        return np.round(warped).astype(np.int32)
    for poly in polygon_def.get('ground', []):
        warped = _warp_poly(poly)
        if warped is not None:
            cv2.fillPoly(mask_ground, [warped], 1)
    for poly in polygon_def.get('line_add', []):
        warped = _warp_poly(poly)
        if warped is not None:
            cv2.fillPoly(mask_lines, [warped], 1)
    for poly in polygon_def.get('line_sub', []):
        warped = _warp_poly(poly)
        if warped is not None:
            cv2.fillPoly(mask_lines, [warped], 0)
    g_sum = int(mask_ground.sum())
    l_sum = int(mask_lines.sum())
    g_size = int(mask_ground.size)
    l_size = int(mask_lines.size)
    if g_sum == 0:
        raise InvalidMask('No projected ground (empty mask)')
    if g_sum == g_size:
        raise InvalidMask('Projected ground covers the entire image surface')
    if l_sum == 0:
        raise InvalidMask('No projected lines (empty mask)')
    if l_sum == l_size:
        raise InvalidMask('Projected lines cover the entire image surface')
    return (mask_ground, mask_lines)

def evaluate_keypoints_for_frame(template_keypoints: list[tuple[int, int]], frame_keypoints: list[tuple[int, int]], frame: np.ndarray, floor_markings_template: np.ndarray, frame_number: int | None=None, *, log_frame_number: bool=False, debug_dir: Path | None=None, cached_edges: np.ndarray | None=None, mask_polygons: dict[str, list[list[tuple[float, float]]]] | None=None, mask_debug_label: str | None=None, mask_debug_dir: Path | None=None, score_only: bool=False, log_context: dict | None=None, processing_scale: float=1.0) -> float:
    try:
        warped_template = None
        mask_ground_bin = None
        mask_lines_expected = None
        homography_matrix = None
        score_only = bool(score_only)
        use_cache = True

        def _early_return(value: float) -> float:
            return value
        frame_id_str = f'Frame {frame_number}' if frame_number is not None else 'Frame'
        frame_height, frame_width = frame.shape[:2]
        original_keypoints = frame_keypoints[:]
        if _normalize_keypoints_cy is not None:
            try:
                frame_keypoints = _normalize_keypoints_cy(original_keypoints, int(frame_width), int(frame_height))
            except Exception:
                frame_keypoints = None
        else:
            frame_keypoints = None
        if frame_keypoints is None:
            frame_keypoints = []
            for x, y in original_keypoints:
                xf, yf = (float(x), float(y))
                if (xf != 0.0 or yf != 0.0) and (xf < 0 or yf < 0 or xf >= frame_width or (yf >= frame_height)):
                    frame_keypoints.append((0, 0))
                else:
                    frame_keypoints.append((xf, yf))
        if not score_only:
            non_idx_set = {idx + 1 for idx, kpts in enumerate(frame_keypoints) if not (kpts[0] == 0 and kpts[1] == 0)}
            for blacklist in BLACKLISTS:
                if non_idx_set.issubset(blacklist):
                    if _both_points_same_direction(frame_keypoints[blacklist[0] - 1], frame_keypoints[blacklist[1] - 1], frame_width, frame_height):
                        return _early_return(0.0)
        cache_key: tuple = (int(frame_number) if frame_number is not None else -1, int(frame_width), int(frame_height), tuple(((float(x), float(y)) for x, y in frame_keypoints)))
        if mask_polygons is None:
            cache_key = cache_key + (round(processing_scale, 6),)
        if use_cache and cache_key in _KP_SCORE_CACHE:
            cached_score = float(_KP_SCORE_CACHE[cache_key])
            _KP_SCORE_CACHE.move_to_end(cache_key)
            return cached_score
        valid_keypoints = [(x, y) for x, y in frame_keypoints if not (x == 0 and y == 0)]
        if len(valid_keypoints) < 4:
            return _early_return(0.0)
        if not score_only:
            xs, ys = zip(*valid_keypoints)
            min_x_kp, max_x_kp = (min(xs), max(xs))
            min_y_kp, max_y_kp = (min(ys), max(ys))
            if max_x_kp < 0 or max_y_kp < 0 or min_x_kp >= frame_width or (min_y_kp >= frame_height):
                return _early_return(0.0)
            if max_x_kp - min_x_kp > 2 * frame_width or max_y_kp - min_y_kp > 2 * frame_height:
                return _early_return(0.0)
        cached_warp = _KP_WARP_CACHE.get(cache_key) if use_cache else None
        if cached_warp is not None:
            _KP_WARP_CACHE.move_to_end(cache_key)
            warped_template, mask_ground_bin, mask_lines_expected = cached_warp
        if warped_template is None and homography_matrix is None:
            try:
                if mask_polygons is not None:
                    filtered_src: list[tuple[int, int]] = []
                    filtered_dst: list[tuple[float, float]] = []
                    for src_pt, dst_pt in zip(template_keypoints, frame_keypoints, strict=True):
                        if dst_pt[0] == 0.0 and dst_pt[1] == 0.0:
                            continue
                        filtered_src.append(src_pt)
                        filtered_dst.append(dst_pt)
                    if len(filtered_src) < 4:
                        raise ValueError('At least 4 valid keypoints are required for homography.')
                    source_points = np.array(filtered_src, dtype=np.float32)
                    destination_points = np.array(filtered_dst, dtype=np.float32)
                    H, _ = cv2.findHomography(source_points, destination_points)
                    if H is None:
                        raise ValueError('Homography computation failed')
                    validate_projected_corners(source_keypoints=template_keypoints, homography_matrix=H)
                    homography_matrix = H
                else:
                    if floor_markings_template is None:
                        floor_markings_template = challenge_template()
                    if mask_polygons is not None:
                        warped_template, homography_matrix = project_image_using_keypoints(image=floor_markings_template, source_keypoints=template_keypoints, destination_keypoints=frame_keypoints, destination_width=frame_width, destination_height=frame_height, return_h=True)
                    elif 0 < processing_scale < 1.0:
                        scaled_w = max(1, int(frame_width * processing_scale))
                        scaled_h = max(1, int(frame_height * processing_scale))
                        frame_kp_scaled = [(float(x) * processing_scale, float(y) * processing_scale) for x, y in frame_keypoints]
                        warped_template, homography_matrix = project_image_using_keypoints(image=floor_markings_template, source_keypoints=template_keypoints, destination_keypoints=frame_kp_scaled, destination_width=scaled_w, destination_height=scaled_h, return_h=True)
                    else:
                        warped_template, homography_matrix = project_image_using_keypoints(image=floor_markings_template, source_keypoints=template_keypoints, destination_keypoints=frame_keypoints, destination_width=frame_width, destination_height=frame_height, return_h=True)
            except ValueError as e:
                return _early_return(0.0)
            except InvalidMask as e:
                return _early_return(0.0)
        if mask_ground_bin is None or mask_lines_expected is None:
            try:
                if mask_polygons is not None:
                    if homography_matrix is None:
                        raise ValueError('Homography not available for polygon masks')
                    mask_ground_bin, mask_lines_expected = _polygon_masks_from_homography(homography_matrix, (frame_height, frame_width), mask_polygons, debug_frame_id=frame_number)
                else:
                    if warped_template is None:
                        raise ValueError('Non-polygon path requires warped_template from project_image_using_keypoints')
                    mask_ground_bin, mask_lines_expected = extract_masks_for_ground_and_lines(warped_template, debug_frame_id=frame_number)
                    if 0 < processing_scale < 1.0:
                        mask_ground_bin = cv2.resize(mask_ground_bin, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                        mask_ground_bin = (mask_ground_bin > 0).astype(np.uint8)
                        mask_lines_expected = cv2.resize(mask_lines_expected, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                        mask_lines_expected = (mask_lines_expected > 0).astype(np.uint8)
                        if REFINE_EXPECTED_MASKS_AT_BOUNDARIES:
                            tpl = floor_markings_template if floor_markings_template is not None else challenge_template()
                            if tpl is not None:
                                warped_full, _ = project_image_using_keypoints(image=tpl, source_keypoints=template_keypoints, destination_keypoints=frame_keypoints, destination_width=frame_width, destination_height=frame_height, return_h=True)
                                mask_ground_bin, mask_lines_expected = extract_masks_for_ground_and_lines(warped_full, debug_frame_id=frame_number)
            except InvalidMask as e:
                return _early_return(0.0)
            if use_cache:
                _KP_WARP_CACHE[cache_key] = (warped_template, mask_ground_bin, mask_lines_expected)
                _KP_WARP_CACHE.move_to_end(cache_key)
                if len(_KP_WARP_CACHE) > _KP_CACHE_MAX:
                    _KP_WARP_CACHE.popitem(last=False)
        if mask_polygons is not None:
            edges = cached_edges
            if edges is None:
                edges = compute_frame_canny_edges(frame)
            if ADD4_POLYGON_SCORE_EDGE_RATE:
                edges_in_line = cv2.bitwise_and(edges, edges, mask=mask_lines_expected)
                mask_lines_predicted = (edges_in_line > 0).astype(np.uint8)
            else:
                mask_lines_predicted = extract_mask_of_ground_lines_in_image(image=frame, ground_mask=mask_ground_bin, cached_edges=edges)
        else:
            mask_lines_predicted = _KP_PRED_CACHE.get(cache_key) if use_cache else None
            if mask_lines_predicted is not None:
                _KP_PRED_CACHE.move_to_end(cache_key)
            if mask_lines_predicted is None:
                mask_lines_predicted = extract_mask_of_ground_lines_in_image(image=frame, ground_mask=mask_ground_bin, cached_edges=cached_edges)
                if use_cache:
                    _KP_PRED_CACHE[cache_key] = mask_lines_predicted
                    _KP_PRED_CACHE.move_to_end(cache_key)
                    if len(_KP_PRED_CACHE) > _KP_CACHE_MAX:
                        _KP_PRED_CACHE.popitem(last=False)
        pixels_overlapping_result = cv2.bitwise_and(mask_lines_expected, mask_lines_predicted)
        if not score_only:
            pts = cv2.findNonZero(mask_lines_expected)
            if pts is None or len(pts) == 0:
                return _early_return(0.0)
            else:
                min_x, min_y, w_box, h_box = cv2.boundingRect(pts)
                max_x = int(min_x + w_box - 1)
                max_y = int(min_y + h_box - 1)
                bbox = (int(min_x), int(min_y), int(max_x), int(max_y))
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = frame_height * frame_width
            if bbox_area / frame_area < 0.2:
                return _early_return(0.0)
            valid_keypoints = [(x, y) for x, y in frame_keypoints if not (x == 0 and y == 0)]
            if not valid_keypoints:
                return _early_return(0.0)
            xs, ys = zip(*valid_keypoints)
            min_x_kp, max_x_kp = (min(xs), max(xs))
            min_y_kp, max_y_kp = (min(ys), max(ys))
            if max_x_kp < 0 or max_y_kp < 0 or min_x_kp >= frame_width or (min_y_kp >= frame_height):
                return _early_return(0.0)
            if max_x_kp - min_x_kp > 2 * frame_width or max_y_kp - min_y_kp > 2 * frame_height:
                return _early_return(0.0)
            inv_expected = cv2.bitwise_not(mask_lines_expected)
            pixels_rest = cv2.bitwise_and(inv_expected, mask_lines_predicted).sum()
            total_pixels = cv2.bitwise_or(mask_lines_expected, mask_lines_predicted).sum()
            if total_pixels == 0:
                return _early_return(0.0)
            if pixels_rest / total_pixels > 0.9:
                return _early_return(0.0)
        pixels_overlapping = pixels_overlapping_result.sum()
        pixels_on_lines = mask_lines_expected.sum()
        overlap_ratio = float(pixels_overlapping) / float(pixels_on_lines + 1e-08)
        if use_cache:
            _KP_SCORE_CACHE[cache_key] = float(overlap_ratio)
            _KP_SCORE_CACHE.move_to_end(cache_key)
            if len(_KP_SCORE_CACHE) > _KP_CACHE_MAX:
                _KP_SCORE_CACHE.popitem(last=False)
        return overlap_ratio
    except Exception as e:
        pass
    return _early_return(0.0)

def _step5_validate_ordered_keypoints(*, ordered_kps: list[list[float]], frame: np.ndarray, template_image: np.ndarray, template_keypoints: list[tuple[int, int]], frame_number: int, best_meta: dict[str, Any], debug_label: str | None=None, cached_edges: np.ndarray | None=None) -> tuple[bool, str | None, list[list[float]]]:
    validation_passed = False
    validation_error: str | None = None
    try:
        if len(ordered_kps) < 4:
            raise ValueError('At least 4 valid keypoints are required.')
        if len(ordered_kps) != len(template_keypoints):
            raise ValueError(f'Keypoint count mismatch (expected {len(template_keypoints)}, got {len(ordered_kps)})')
        frame_height, frame_width = frame.shape[:2]
        frame_keypoints_tuples: list[tuple[float, float]] = []
        clamped_count = 0
        for idx, kp in enumerate(ordered_kps):
            if kp and len(kp) >= 2:
                x, y = (float(kp[0]), float(kp[1]))
                if (x, y) != (0, 0) and (x < 0 or y < 0 or x >= frame_width or (y >= frame_height)):
                    frame_keypoints_tuples.append((0, 0))
                    clamped_count += 1
                else:
                    frame_keypoints_tuples.append((x, y))
            else:
                frame_keypoints_tuples.append((0, 0))
        score = best_meta.get('score', 0.0)
        if score > 0.0:
            validation_passed = True
            if clamped_count > 0:
                for idx, clamped_kp in enumerate(frame_keypoints_tuples):
                    if idx < len(ordered_kps):
                        ordered_kps[idx] = [float(clamped_kp[0]), float(clamped_kp[1])]
                best_meta['reordered_keypoints'] = ordered_kps
        else:
            validation_error = f'Score validation failed: score={score:.6f}'
    except ValueError as e:
        validation_error = str(e)
    except Exception as e:
        validation_error = f'Unexpected error: {str(e)}'
    return (validation_passed, validation_error, ordered_kps)

def _step6_fill_keypoints_from_homography(*, ordered_kps: list[list[float]], frame: np.ndarray, frame_number: int) -> list[list[float]]:
    frame_height, frame_width = frame.shape[:2]
    valid_src_points: list[tuple[int, int]] = []
    valid_dst_points: list[tuple[int, int]] = []
    valid_indices: list[int] = []
    for idx, kp in enumerate(ordered_kps):
        if kp and len(kp) >= 2 and (not (abs(kp[0]) < 1e-06 and abs(kp[1]) < 1e-06)):
            x = float(kp[0])
            y = float(kp[1])
            if 0 <= x < frame_width and 0 <= y < frame_height:
                valid_src_points.append(FOOTBALL_KEYPOINTS_CORRECTED[idx])
                valid_dst_points.append((int(x), int(y)))
                valid_indices.append(idx)
    if len(valid_src_points) < 4:
        return ordered_kps
    source_points = np.array(valid_src_points, dtype=np.float32)
    destination_points = np.array(valid_dst_points, dtype=np.float32)
    H, _ = cv2.findHomography(source_points, destination_points)
    if H is None:
        return ordered_kps
    all_template_points = np.array(FOOTBALL_KEYPOINTS_CORRECTED, dtype=np.float32).reshape(-1, 1, 2)
    projected_points = cv2.perspectiveTransform(all_template_points, H)
    projected_points = projected_points.reshape(-1, 2)
    updated_kps = [list(kp) if kp else [0.0, 0.0] for kp in ordered_kps]
    num_kps = min(len(FOOTBALL_KEYPOINTS_CORRECTED), len(updated_kps))
    valid_indices_set = set(valid_indices)
    indices_to_process = np.array([i for i in range(num_kps) if i not in valid_indices_set], dtype=np.int32)
    if len(indices_to_process) > 0:
        proj_x_arr = projected_points[indices_to_process, 0]
        proj_y_arr = projected_points[indices_to_process, 1]
        in_bounds_mask = (proj_x_arr >= 0) & (proj_y_arr >= 0) & (proj_x_arr < frame_width) & (proj_y_arr < frame_height)
        filled_count = 0
        out_of_bounds_count = 0
        for i, idx in enumerate(indices_to_process):
            if in_bounds_mask[i]:
                updated_kps[idx] = [float(proj_x_arr[i]), float(proj_y_arr[i])]
                filled_count += 1
            else:
                updated_kps[idx] = [0.0, 0.0]
                out_of_bounds_count += 1
    else:
        filled_count = 0
        out_of_bounds_count = 0
    return updated_kps

def _step1_build_connections(*, frame: np.ndarray, kps: list[list[float]] | list[Any], labels: list[int], frame_number: int, frame_width: int, frame_height: int, cached_edges: np.ndarray | None=None) -> dict[str, Any]:
    orig_frame_width = frame_width
    orig_frame_height = frame_height
    orig_kps = list(kps) if kps else []
    scale = float(PROCESSING_SCALE)
    if scale < 1.0 and scale > 0:
        new_w = max(1, int(frame_width * scale))
        new_h = max(1, int(frame_height * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if cached_edges is not None:
            cached_edges = cv2.resize(cached_edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        kps = [[float(p[0]) * scale, float(p[1]) * scale] if p and len(p) >= 2 else p or [0.0, 0.0] for p in kps or []]
        frame_width, frame_height = (new_w, new_h)
    ground_mask = np.ones(frame.shape[:2], dtype=np.uint8)
    mask_pred = extract_mask_of_ground_lines_in_image(image=frame, ground_mask=ground_mask, cached_edges=cached_edges, dilate_iterations=2)
    m_closed = (mask_pred > 0).astype(np.uint8) * 255
    close_ksize = 3
    if close_ksize and close_ksize > 1:
        k = _kernel_ellipse_5() if close_ksize == 5 else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m_closed = cv2.morphologyEx(m_closed, cv2.MORPH_CLOSE, k, iterations=1)
    good_details: list[tuple[int, int, float, int]] = []
    valid_idx = [idx_v for idx_v, pt in enumerate(kps or []) if pt and len(pt) >= 2 and (not (pt[0] == 0 and pt[1] == 0))]
    pairs = list(combinations(valid_idx, 2))

    def _check_pair(pair: tuple[int, int]) -> tuple[int, int, float, int] | None:
        i, j = pair
        p1 = kps[i]
        p2 = kps[j]
        ok, hit_ratio, longest = _connected_by_segment(m_closed, tuple(p1), tuple(p2), sample_radius=3, close_ksize=0, sample_step=2)
        return (i, j, hit_ratio, longest) if ok else None
    if len(pairs) > 1:
        max_workers = min(len(pairs), TV_AF_EVAL_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for result in ex.map(_check_pair, pairs):
                if result is not None:
                    good_details.append(result)
    else:
        for pair in pairs:
            result = _check_pair(pair)
            if result is not None:
                good_details.append(result)
    pair_to_len: dict[tuple[int, int], float] = {}
    for i, j, _, _ in good_details:
        p1 = kps[i]
        p2 = kps[j]
        if not p1 or not p2 or len(p1) < 2 or (len(p2) < 2):
            continue
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        pair_to_len[tuple(sorted((int(i), int(j))))] = float(np.hypot(dx, dy))
    edges_to_remove: set[tuple[int, int]] = set()
    nodes = {a for pair in pair_to_len.keys() for a in pair}
    for a, b, c in combinations(sorted(nodes), 3):
        tri_edges = [tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))]
        if all((e in pair_to_len for e in tri_edges)):
            longest_edge = max(tri_edges, key=lambda e: pair_to_len[e])
            edges_to_remove.add(longest_edge)
    filtered_details = [(i, j, hr, lg) for i, j, hr, lg in good_details if tuple(sorted((i, j))) not in edges_to_remove]
    frame_connections = sorted([[int(i), int(j)] for i, j, _, _ in filtered_details])
    return {'frame_id': int(frame_number), 'frame_size': [int(orig_frame_width), int(orig_frame_height)], 'keypoints': [None if pt is None else [float(x) for x in pt] for pt in orig_kps], 'labels': labels, 'connections': frame_connections}

def _step2_build_four_point_groups(*, frame: np.ndarray, step1_entry: dict[str, Any], frame_number: int) -> dict[str, Any]:
    keypoints = step1_entry['keypoints']
    labels = step1_entry['labels']
    connections: Iterable[list[int]] = step1_entry['connections'] or []
    frame_size = step1_entry['frame_size']
    frame_width = frame_size[0] if frame_size and len(frame_size) >= 1 else 1920
    frame_height = frame_size[1] if frame_size and len(frame_size) >= 2 else 1080
    border_margin = 50.0
    valid_nodes = {idx_v for idx_v, pt in enumerate(keypoints) if pt is not None and len(pt) >= 2 and (not (abs(pt[0]) < 1e-06 and abs(pt[1]) < 1e-06)) and (float(pt[0]) >= border_margin) and (float(pt[0]) < frame_width - border_margin) and (float(pt[1]) >= border_margin) and (float(pt[1]) < frame_height - border_margin)}
    edge_set: set[tuple[int, int]] = {tuple(sorted((int(a), int(b)))) for a, b in connections if int(a) in valid_nodes and int(b) in valid_nodes}
    candidate_nodes = sorted(valid_nodes)
    deg: dict[int, int] = {int(i): 0 for i in candidate_nodes}
    for a, b in edge_set:
        deg[int(a)] = deg.get(int(a), 0) + 1
        deg[int(b)] = deg.get(int(b), 0) + 1
    _hypot = np.hypot
    _abs = abs

    def _dist_from_line(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        dx = bx - ax
        dy = by - ay
        denom = _hypot(dx, dy)
        if denom < 1e-06:
            return float('inf')
        num = _abs(dy * px - dx * py + bx * ay - by * ax)
        return num / denom

    def _has_collinear_triplet_20px(a: int, b: int, c: int, d: int, pts: list) -> bool:
        TOL = 20.0
        pa = pts[a]
        pb = pts[b]
        pc = pts[c]
        pd = pts[d]
        ax, ay = (float(pa[0]), float(pa[1]))
        bx, by = (float(pb[0]), float(pb[1]))
        cx, cy = (float(pc[0]), float(pc[1]))
        dx, dy = (float(pd[0]), float(pd[1]))
        if _dist_from_line(cx, cy, ax, ay, bx, by) <= TOL:
            return True
        if _dist_from_line(ax, ay, bx, by, cx, cy) <= TOL:
            return True
        if _dist_from_line(bx, by, ax, ay, cx, cy) <= TOL:
            return True
        if _dist_from_line(dx, dy, ax, ay, bx, by) <= TOL:
            return True
        if _dist_from_line(ax, ay, bx, by, dx, dy) <= TOL:
            return True
        if _dist_from_line(bx, by, ax, ay, dx, dy) <= TOL:
            return True
        if _dist_from_line(dx, dy, ax, ay, cx, cy) <= TOL:
            return True
        if _dist_from_line(ax, ay, cx, cy, dx, dy) <= TOL:
            return True
        if _dist_from_line(cx, cy, ax, ay, dx, dy) <= TOL:
            return True
        if _dist_from_line(dx, dy, bx, by, cx, cy) <= TOL:
            return True
        if _dist_from_line(bx, by, cx, cy, dx, dy) <= TOL:
            return True
        if _dist_from_line(cx, cy, bx, by, dx, dy) <= TOL:
            return True
        return False

    def _has_severe_collinearity_5px(a: int, b: int, c: int, d: int, pts: list) -> bool:
        TOL = 5.0
        pa = pts[a]
        pb = pts[b]
        pc = pts[c]
        pd = pts[d]
        ax, ay = (float(pa[0]), float(pa[1]))
        bx, by = (float(pb[0]), float(pb[1]))
        cx, cy = (float(pc[0]), float(pc[1]))
        dx, dy = (float(pd[0]), float(pd[1]))
        if _dist_from_line(cx, cy, ax, ay, bx, by) <= TOL:
            return True
        if _dist_from_line(dx, dy, ax, ay, bx, by) <= TOL:
            return True
        if _dist_from_line(dx, dy, ax, ay, cx, cy) <= TOL:
            return True
        if _dist_from_line(dx, dy, bx, by, cx, cy) <= TOL:
            return True
        return False

    def _spread_score_min_pair_distance(a: int, b: int, c: int, d: int, pts: list) -> float:
        pa = pts[a]
        pb = pts[b]
        pc = pts[c]
        pd = pts[d]
        ax, ay = (float(pa[0]), float(pa[1]))
        bx, by = (float(pb[0]), float(pb[1]))
        cx, cy = (float(pc[0]), float(pc[1]))
        dx, dy = (float(pd[0]), float(pd[1]))
        d_ab = _hypot(ax - bx, ay - by)
        d_ac = _hypot(ax - cx, ay - cy)
        d_ad = _hypot(ax - dx, ay - dy)
        d_bc = _hypot(bx - cx, by - cy)
        d_bd = _hypot(bx - dx, by - dy)
        d_cd = _hypot(cx - dx, cy - dy)
        m = min(d_ab, d_ac, d_ad, d_bc, d_bd, d_cd)
        return float(m) if m != float('inf') else 0.0
    total_combos = 0
    skipped_collinear = 0
    matching_groups: list[tuple[int, list[int]]] = []
    for combo in combinations(candidate_nodes, 4):
        total_combos += 1
        a, b, c, d = (combo[0], combo[1], combo[2], combo[3])
        if _has_collinear_triplet_20px(int(a), int(b), int(c), int(d), keypoints):
            skipped_collinear += 1
            continue
        points_with_conn_count = int(deg.get(int(a), 0) > 0) + int(deg.get(int(b), 0) > 0) + int(deg.get(int(c), 0) > 0) + int(deg.get(int(d), 0) > 0)
        matching_groups.append((points_with_conn_count, [int(a), int(b), int(c), int(d)]))
    def _calculate_group_difference(group1: list[int], group2: list[int]) -> int:
        set1 = set(group1)
        set2 = set(group2)
        return len(set1.symmetric_difference(set2))
    four_point_groups: list[list[int]] = []
    if matching_groups:
        matching_groups_scored: list[tuple[int, float, list[int]]] = []
        for conn_count, group in matching_groups:
            a, b, c, d = (int(group[0]), int(group[1]), int(group[2]), int(group[3]))
            spread = _spread_score_min_pair_distance(a, b, c, d, keypoints)
            matching_groups_scored.append((int(conn_count), float(spread), group))
        matching_groups_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected_groups: list[tuple[int, list[int]]] = []
        for conn_count, _spread, group in matching_groups_scored:
            if len(selected_groups) >= 3:
                break
            is_different = True
            if selected_groups:
                for _, selected_group in selected_groups:
                    diff = _calculate_group_difference(group, selected_group)
                    if diff < 2:
                        is_different = False
                        break
            if is_different or len(selected_groups) == 0:
                selected_groups.append((conn_count, group))
                four_point_groups.append(group)
    else:
        fallback_groups: list[list[int]] = []
        for combo in combinations(candidate_nodes, 4):
            a, b, c, d = (combo[0], combo[1], combo[2], combo[3])
            if _has_severe_collinearity_5px(int(a), int(b), int(c), int(d), keypoints):
                continue
            points_with_conn_count = int(deg.get(int(a), 0) > 0) + int(deg.get(int(b), 0) > 0) + int(deg.get(int(c), 0) > 0) + int(deg.get(int(d), 0) > 0)
            fallback_groups.append((points_with_conn_count, [int(a), int(b), int(c), int(d)]))
        if fallback_groups:
            fallback_groups_scored: list[tuple[int, float, list[int]]] = []
            for conn_count, group in fallback_groups:
                a, b, c, d = (int(group[0]), int(group[1]), int(group[2]), int(group[3]))
                spread = _spread_score_min_pair_distance(a, b, c, d, keypoints)
                fallback_groups_scored.append((int(conn_count), float(spread), group))
            fallback_groups_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            selected_fallback_groups: list[tuple[int, list[int]]] = []
            for conn_count, _spread, group in fallback_groups_scored:
                if len(selected_fallback_groups) >= 3:
                    break
                is_different = True
                if selected_fallback_groups:
                    for _, selected_group in selected_fallback_groups:
                        diff = _calculate_group_difference(group, selected_group)
                        if diff < 2:
                            is_different = False
                            break
                if is_different or len(selected_fallback_groups) == 0:
                    selected_fallback_groups.append((conn_count, group))
                    four_point_groups.append(group)
    return {'frame_id': step1_entry['frame_id'], 'frame_size': step1_entry['frame_size'], 'connections': sorted([list(edge) for edge in edge_set]), 'keypoints': keypoints, 'labels': labels, 'four_points': four_point_groups}

def _step3_build_ordered_candidates(*, step2_entry: dict[str, Any], template_patterns: dict[tuple[bool, ...], list[list[int]]], template_patterns_allowed_left: dict[tuple[bool, ...], list[list[int]]] | None=None, template_patterns_allowed_right: dict[tuple[bool, ...], list[list[int]]] | None=None, template_labels: list[int], frame_number: int) -> dict[str, Any]:
    frame_connections: list[list[int]] = list(step2_entry.get('connections') or [])
    frame_four_points: list[list[int]] = list(step2_entry.get('four_points') or [])
    frame_keypoints = step2_entry.get('keypoints') or []
    frame_size = step2_entry.get('frame_size') or []
    frame_labels = step2_entry.get('labels') or []
    frame_edge_set = _edge_set_from_connections(frame_connections)
    cx, _ = _connection_centroid(frame_connections, frame_keypoints)
    width = frame_size[0] if frame_size and len(frame_size) >= 1 else None
    allowed_left = set(range(0, 17)) | {30, 31}
    allowed_right = set(range(13, 32))
    allowed_set: set[int] | None = None
    decision: str | None = None
    if FORCE_DECISION_RIGHT:
        allowed_set = allowed_right
        decision = 'right'

    def _edge_direction_rule(target_labels: set[int]) -> bool:
        nonlocal allowed_set, decision
        for edge in frame_connections:
            if len(edge) < 2:
                continue
            a, b = (int(edge[0]), int(edge[1]))
            if a < 0 or b < 0 or a >= len(frame_labels) or (b >= len(frame_labels)):
                continue
            lab_a = frame_labels[a]
            lab_b = frame_labels[b]
            if {int(lab_a or -1), int(lab_b or -1)} != target_labels:
                continue
            kp_a = frame_keypoints[a] if a < len(frame_keypoints) else None
            kp_b = frame_keypoints[b] if b < len(frame_keypoints) else None
            if not kp_a or not kp_b or len(kp_a) < 2 or (len(kp_b) < 2):
                continue
            x1, y1 = (float(kp_a[0]), float(kp_a[1]))
            x2, y2 = (float(kp_b[0]), float(kp_b[1]))
            if abs(x1) < 1e-06 and abs(y1) < 1e-06 or (abs(x2) < 1e-06 and abs(y2) < 1e-06):
                continue
            same_sign = x1 < x2 and y1 < y2 or (x1 > x2 and y1 > y2)
            if same_sign:
                allowed_set = allowed_right
                decision = 'right'
            else:
                allowed_set = allowed_left
                decision = 'left'
            return True
        return False
    if not FORCE_DECISION_RIGHT:
        if decision is None:
            _edge_direction_rule({1, 2})
        if decision is None:
            _edge_direction_rule({2, 2})
        if decision is None:
            _edge_direction_rule({3, 3})
    if not FORCE_DECISION_RIGHT and decision is None and (width is not None):
        xs_label23: list[float] = []
        for idx_pt, kp in enumerate(frame_keypoints or []):
            if not kp or len(kp) < 2:
                continue
            x_val, y_val = (float(kp[0]), float(kp[1]))
            if abs(x_val) < 1e-06 and abs(y_val) < 1e-06:
                continue
            lab_val = frame_labels[idx_pt] if idx_pt < len(frame_labels) else None
            if lab_val is not None and int(lab_val) in (2, 3):
                xs_label23.append(x_val)
        if xs_label23:
            cx_label23 = sum(xs_label23) / len(xs_label23)
            if cx_label23 <= width / 2.0:
                allowed_set = allowed_left
                decision = 'left'
            else:
                allowed_set = allowed_right
                decision = 'right'
        elif cx is not None:
            if cx <= width / 2.0:
                allowed_set = allowed_left
                decision = 'left'
            else:
                allowed_set = allowed_right
                decision = 'right'
    constraints: dict[int, set[int]] = {}
    valid_pts: list[tuple[int, float, Any]] = []
    for idx_pt, kp in enumerate(frame_keypoints or []):
        if not kp or len(kp) < 2:
            continue
        x_val, y_val = (float(kp[0]), float(kp[1]))
        if abs(x_val) < 1e-06 and abs(y_val) < 1e-06:
            continue
        lab_val = frame_labels[idx_pt] if idx_pt < len(frame_labels) else None
        valid_pts.append((idx_pt, y_val, lab_val))
    if valid_pts:
        ys = [y for _, y, _ in valid_pts]
        y_min = min(ys)
        y_max = max(ys)
        min_pts = [(idx_v, lab_v) for idx_v, y, lab_v in valid_pts if abs(y - y_min) < 1e-06]
        max_pts = [(idx_v, lab_v) for idx_v, y, lab_v in valid_pts if abs(y - y_max) < 1e-06]
        if len(min_pts) == 1 and int(min_pts[0][1] or 0) == 1:
            constraints[min_pts[0][0]] = {0, 24}
        if len(max_pts) == 1 and int(max_pts[0][1] or 0) == 1:
            constraints.setdefault(max_pts[0][0], set()).update({5, 29})

    def _add_constraint(idx: int, allowed: set[int]) -> None:
        if idx < 0:
            return
        prev = constraints.get(idx)
        if prev is None:
            constraints[idx] = set(allowed)
        else:
            merged = set(prev) & set(allowed)
            constraints[idx] = merged if merged else set(allowed)
    labeled_points: list[tuple[int, float, float, int]] = []
    for idx_pt, kp in enumerate(frame_keypoints or []):
        if not kp or len(kp) < 2:
            continue
        x_val, y_val = (float(kp[0]), float(kp[1]))
        if abs(x_val) < 1e-06 and abs(y_val) < 1e-06:
            continue
        lab_val = frame_labels[idx_pt] if idx_pt < len(frame_labels) else None
        if lab_val is None:
            continue
        labeled_points.append((idx_pt, x_val, y_val, int(lab_val)))
    lab5_pts = [(idx, y) for idx, _x, y, lab in labeled_points if lab == 5]
    if len(lab5_pts) == 2:
        lab5_pts_sorted = sorted(lab5_pts, key=lambda t: t[1])
        _add_constraint(lab5_pts_sorted[0][0], {13})
        _add_constraint(lab5_pts_sorted[1][0], {16})
    elif len(lab5_pts) == 1:
        a_idx, a_y = lab5_pts[0]
        ys_lab4_6 = [y for _idx, _x, y, lab in labeled_points if lab in (4, 6)]
        any_bigger = any((y > a_y for y in ys_lab4_6))
        any_smaller = any((y < a_y for y in ys_lab4_6))
        if any_bigger and (not any_smaller):
            _add_constraint(a_idx, {13})
        elif any_smaller and (not any_bigger):
            _add_constraint(a_idx, {16})
    lab4_pts = [(idx, x) for idx, x, _y, lab in labeled_points if lab == 4]
    if len(lab4_pts) == 2:
        lab4_sorted = sorted(lab4_pts, key=lambda t: t[1])
        _add_constraint(lab4_sorted[0][0], {30})
        _add_constraint(lab4_sorted[1][0], {31})
    elif len(lab4_pts) == 1:
        b_idx, b_x = lab4_pts[0]
        lab6_pts = [(idx, x, y) for idx, x, y, lab in labeled_points if lab == 6]
        if len(lab6_pts) == 0:
            if decision == 'right':
                _add_constraint(b_idx, {31})
            elif decision == 'left':
                _add_constraint(b_idx, {30})
        elif len(lab6_pts) >= 1:
            lab6_x_coords = [x for _idx, x, _y in lab6_pts]
            lab6_center_x = sum(lab6_x_coords) / len(lab6_x_coords)
            if lab6_center_x < b_x:
                _add_constraint(b_idx, {31})
            elif lab6_center_x > b_x:
                _add_constraint(b_idx, {30})
            elif decision is not None:
                if decision == 'right':
                    _add_constraint(b_idx, {31})
                elif decision == 'left':
                    _add_constraint(b_idx, {30})
    lab1_pts = [(idx, x, y) for idx, x, y, lab in labeled_points if lab == 1]
    lab2_pts = [(idx, x, y) for idx, x, y, lab in labeled_points if lab == 2]
    lab3_pts = [(idx, x, y) for idx, x, y, lab in labeled_points if lab == 3]
    if len(lab1_pts) == 0 and len(lab2_pts) == 0 and (len(lab3_pts) == 1):
        lab3_idx, _lab3_x, _lab3_y = lab3_pts[0]
        if decision == 'left':
            _add_constraint(lab3_idx, {9, 12})
        elif decision == 'right':
            _add_constraint(lab3_idx, {17, 20})
    lab6_pts = [(idx, y) for idx, _x, y, lab in labeled_points if lab == 6]
    if len(lab6_pts) == 2:
        lab6_sorted = sorted(lab6_pts, key=lambda t: t[1])
        _add_constraint(lab6_sorted[0][0], {14})
        _add_constraint(lab6_sorted[1][0], {15})

    def _expanded_patterns(base: tuple[bool, ...]) -> set[tuple[bool, ...]]:
        base_list = list(base)
        indices_false = [i for i, v in enumerate(base_list) if not v]
        patterns_out: set[tuple[bool, ...]] = set()
        triangles = [(0, 1, 3), (0, 2, 4), (1, 2, 5), (3, 4, 5)]
        for mask in range(1 << len(indices_false)):
            edges = base_list[:]
            for bit, idx_edge in enumerate(indices_false):
                if mask >> bit & 1:
                    edges[idx_edge] = True
            bad = False
            for a, b, c in triangles:
                if edges[a] and edges[b] and edges[c]:
                    bad = True
                    break
            if not bad:
                patterns_out.add(tuple(edges))
        return patterns_out
    frame_labels_int = [int(v) if v is not None else -1 for v in frame_labels]
    template_labels_int = [int(v) if v is not None else -1 for v in template_labels]
    constraints_mask: list[int | None] = [None] * len(frame_labels_int)
    for idx_q, allowed in constraints.items():
        if idx_q is None or idx_q < 0 or idx_q >= len(constraints_mask):
            continue
        mask = 0
        for ci in allowed:
            if ci is not None and 0 <= int(ci) < 32:
                mask |= 1 << int(ci)
        constraints_mask[int(idx_q)] = mask

    def _labels_match(cand: list[int], quad: list[int]) -> bool:
        if len(cand) < 4 or len(quad) < 4:
            return False
        for ci, qi in zip(cand[:4], quad[:4]):
            if qi is None or qi < 0 or qi >= len(frame_labels_int):
                return False
            if ci is None or ci < 0 or ci >= len(template_labels_int):
                return False
            mask = constraints_mask[qi]
            if mask is not None and (not mask >> ci & 1):
                return False
            if template_labels_int[ci] != frame_labels_int[qi]:
                return False
        return True
    frame_labels_arr = np.asarray(frame_labels_int, dtype=np.int32)
    template_labels_arr = np.asarray(template_labels_int, dtype=np.int32)
    constraints_mask_arr = np.asarray([m if m is not None else -1 for m in constraints_mask], dtype=np.int64)
    template_adj_set: dict[int, set[int]] = {i: set(nbrs) for i, nbrs in enumerate(KEYPOINT_CONNECTIONS)}
    connection_graph: dict[int, set[int]] = {}
    for edge in frame_connections:
        if len(edge) < 2:
            continue
        a, b = (int(edge[0]), int(edge[1]))
        connection_graph.setdefault(a, set()).add(b)
        connection_graph.setdefault(b, set()).add(a)
    frame_edges_arr = np.asarray(frame_connections, dtype=np.int32).reshape(-1, 2)
    frame_reach3 = _build_frame_reach3_bitset(connection_graph, len(frame_labels_int)) if _step3_conn_constraints_cy is not None else None
    template_adj_mask, template_reach2_mask = _get_template_reach_bitmasks()
    frame_adj_mask = _build_frame_adj_bitset(connection_graph, len(frame_labels_int)) if _step3_conn_label_constraints_cy is not None else None
    template_neighbor_label_mask = _get_template_neighbor_label_mask()
    label1_ys: list[tuple[int, float]] = []
    for idx_kp, kp in enumerate(frame_keypoints):
        if idx_kp < len(frame_labels) and frame_labels[idx_kp] == 1:
            if kp and len(kp) >= 2:
                kp_y = float(kp[1])
                if abs(kp_y) > 1e-06:
                    label1_ys.append((idx_kp, kp_y))
    if label1_ys:
        label1_ys.sort(key=lambda x: x[1])
        label1_min_idx = label1_ys[0][0]
        label1_max_idx = label1_ys[-1][0]
    else:
        label1_min_idx = None
        label1_max_idx = None

    def _check_connection_label_constraints(cand: list[int], quad: list[int]) -> bool:
        if len(cand) < 4 or len(quad) < 4:
            return False
        frame_to_template: dict[int, int] = {}
        for ci, qi in zip(cand[:4], quad[:4]):
            if qi is not None and ci is not None:
                frame_to_template[int(qi)] = int(ci)
        for frame_point in quad[:4]:
            if frame_point not in frame_to_template:
                continue
            template_point = frame_to_template[frame_point]
            connected_frame_points = connection_graph.get(frame_point, set())
            connected_labels = set()
            connected_label1_points: list[int] = []
            for connected_frame_pt in connected_frame_points:
                if 0 <= connected_frame_pt < len(frame_labels):
                    label = frame_labels[connected_frame_pt]
                    if label > 0:
                        connected_labels.add(int(label))
                        if int(label) == 1:
                            connected_label1_points.append(connected_frame_pt)
            for label1_frame_pt in connected_label1_points:
                if label1_frame_pt < 0 or label1_frame_pt >= len(frame_keypoints):
                    continue
                label1_kp = frame_keypoints[label1_frame_pt]
                if not label1_kp or len(label1_kp) < 2:
                    continue
                if label1_min_idx is None or label1_max_idx is None:
                    continue
                if decision == 'right':
                    if label1_frame_pt == label1_min_idx:
                        required_template = 24
                    elif label1_frame_pt == label1_max_idx:
                        required_template = 29
                    else:
                        continue
                elif decision == 'left':
                    if label1_frame_pt == label1_min_idx:
                        required_template = 0
                    elif label1_frame_pt == label1_max_idx:
                        required_template = 5
                    else:
                        continue
                else:
                    continue
                if required_template not in template_adj_set.get(template_point, set()):
                    return False
            if connected_labels:
                template_neighbors = template_adj_set.get(template_point, set())
                template_neighbor_labels = set()
                for neighbor_template_pt in template_neighbors:
                    if 0 <= neighbor_template_pt < len(KEYPOINT_LABELS):
                        neighbor_label = KEYPOINT_LABELS[neighbor_template_pt]
                        if neighbor_label > 0:
                            template_neighbor_labels.add(int(neighbor_label))
                missing_labels = connected_labels - template_neighbor_labels
                if missing_labels:
                    return False
        return True

    def _check_connection_constraints(cand: list[int], quad: list[int]) -> bool:
        if len(cand) < 4 or len(quad) < 4:
            return False
        frame_to_template: dict[int, int] = {}
        quad_set = set(quad[:4])
        for ci, qi in zip(cand[:4], quad[:4]):
            if qi is not None and ci is not None:
                frame_to_template[int(qi)] = int(ci)
        quad_connections: dict[int, set[int]] = {q: set() for q in quad[:4]}

        def find_paths(start: int, target_set: set[int], max_depth: int=3) -> set[int]:
            if start not in connection_graph:
                return set()
            visited = {start}
            queue = deque([(start, 0)])
            reachable = set()
            while queue:
                node, depth = queue.popleft()
                if node in target_set and node != start:
                    reachable.add(node)
                if depth < max_depth and node in connection_graph:
                    for neighbor in connection_graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))
            return reachable
        for q in quad[:4]:
            if q in connection_graph:
                quad_connections[q].update(find_paths(q, quad_set, max_depth=3))
        for q1 in quad[:4]:
            if q1 not in frame_to_template:
                continue
            template_q1 = frame_to_template[q1]
            neighbors_q1 = template_adj_set.get(template_q1, set())
            for q2 in quad_connections.get(q1, set()):
                if q2 == q1 or q2 not in frame_to_template:
                    continue
                template_q2 = frame_to_template[q2]
                if template_q2 in neighbors_q1:
                    continue
                found_path = False
                for neighbor in neighbors_q1:
                    if template_q2 in template_adj_set.get(neighbor, set()):
                        found_path = True
                        break
                if not found_path:
                    for neighbor in neighbors_q1:
                        for neighbor2 in template_adj_set.get(neighbor, set()):
                            if template_q2 in template_adj_set.get(neighbor2, set()):
                                found_path = True
                                break
                        if found_path:
                            break
                if not found_path:
                    return False
        for edge in frame_connections:
            if len(edge) < 2:
                continue
            a, b = (int(edge[0]), int(edge[1]))
            if a in frame_to_template and b in frame_to_template:
                template_a = frame_to_template[a]
                template_b = frame_to_template[b]
                if template_b not in template_adj_set.get(template_a, set()):
                    return False
        return True
    if allowed_set is not None and decision in ('left', 'right'):
        if decision == 'left' and template_patterns_allowed_left is not None:
            patterns_src = template_patterns_allowed_left
            patterns_src_key = 'left'
        elif decision == 'right' and template_patterns_allowed_right is not None:
            patterns_src = template_patterns_allowed_right
            patterns_src_key = 'right'
        else:
            patterns_src = template_patterns
            patterns_src_key = 'all'
    else:
        patterns_src = template_patterns
        patterns_src_key = 'all'
    expanded_patterns_cache: dict[tuple[bool, ...], tuple[tuple[bool, ...], ...]] = {}
    mapped: list[dict[str, Any]] = []

    def _cand_arr_from_candidates(cands: list[tuple[int, ...]] | list[list[int]]) -> np.ndarray:
        arr = np.asarray(cands, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] < 4:
            arr = np.asarray([c[:4] for c in cands], dtype=np.int32)
        elif arr.shape[1] > 4:
            arr = arr[:, :4]
        return arr

    def _process_one_quad(quad: list[int]) -> dict[str, Any]:
        pattern = _pattern_for_quad(quad, frame_edge_set)
        pattern_key = expanded_patterns_cache.get(pattern)
        if pattern_key is None:
            pattern_set = _expanded_patterns(pattern)
            pattern_key = tuple(sorted(pattern_set))
            expanded_patterns_cache[pattern] = pattern_key
        candidates_all: list[tuple[int, ...]] | None = None
        candidates: list[tuple[int, ...]] = []
        cand_arr_cached: np.ndarray | None = None
        cand_arr: np.ndarray | None = None
        cand_all_len = cand_unique_len = 0
        cached = False
        try:
            candidates, cand_arr_cached, cand_all_len, cand_unique_len = _get_step3_candidates_cached(cache_key=patterns_src_key, pattern_key=pattern_key, patterns_src=patterns_src)
            cached = True
        except Exception:
            pattern_set = _expanded_patterns(pattern)
            candidates_all = []
            for pat in pattern_set:
                candidates_all.extend(patterns_src.get(pat, []))
            candidates = list(candidates_all)
            cand_all_len = len(candidates_all)
            cand_unique_len = len(candidates)
        if not cached and candidates_all is not None:
            if len(pattern_key) == 1:
                candidates = list(candidates_all)
            else:
                seen_cand: set[tuple[int, ...]] = set()
                candidates = []
                for cand in candidates_all:
                    tup = cand if isinstance(cand, tuple) else tuple(cand)
                    if tup in seen_cand:
                        continue
                    seen_cand.add(tup)
                    candidates.append(cand)
        if patterns_src is template_patterns:
            if allowed_set is not None:
                candidates = [cand for cand in candidates if all((node in allowed_set for node in cand))]
        if cached and cand_arr_cached is not None:
            if patterns_src_key != 'all' or allowed_set is None:
                cand_arr = cand_arr_cached
        if template_labels:
            if _step3_filter_labels_cy is not None and candidates:
                quad_arr = np.asarray(quad[:4], dtype=np.int32)
                if cand_arr is None:
                    cand_arr = _cand_arr_from_candidates(candidates)
                keep_idx = _step3_filter_labels_cy(cand_arr, quad_arr, frame_labels_arr, template_labels_arr, constraints_mask_arr)
                candidates = [candidates[i] for i in keep_idx]
                if cand_arr is not None:
                    cand_arr = cand_arr[keep_idx]
            else:
                candidates = [cand for cand in candidates if _labels_match(cand, quad)]
                cand_arr = None
            before_label_conn = len(candidates)
            use_cy_connlabel = False
            quad4 = quad[:4]
            if _step3_conn_label_constraints_cy is not None and frame_adj_mask is not None and candidates and (len(quad4) == 4):
                if all((q is not None and q >= 0 for q in quad4)):
                    use_cy_connlabel = max(quad4) < len(frame_adj_mask)
            if use_cy_connlabel:
                quad_arr = np.asarray(quad4, dtype=np.int32)
                if cand_arr is None:
                    cand_arr = _cand_arr_from_candidates(candidates)
                decision_flag = -1
                if decision == 'left':
                    decision_flag = 0
                elif decision == 'right':
                    decision_flag = 1
                keep_idx = _step3_conn_label_constraints_cy(cand_arr, quad_arr, frame_labels_arr, frame_adj_mask, template_adj_mask, template_neighbor_label_mask, int(label1_min_idx) if label1_min_idx is not None else -1, int(label1_max_idx) if label1_max_idx is not None else -1, decision_flag)
                candidates = [candidates[i] for i in keep_idx]
                if cand_arr is not None:
                    cand_arr = cand_arr[keep_idx]
            else:
                candidates = [cand for cand in candidates if _check_connection_label_constraints(cand, quad)]
                cand_arr = None
            before_conn = len(candidates)
            quad4 = quad[:4]
            use_cy_conn = False
            if _step3_conn_constraints_cy is not None and frame_reach3 is not None and candidates and (len(quad4) == 4):
                if all((q is not None and q >= 0 for q in quad4)):
                    use_cy_conn = max(quad4) < len(frame_reach3)
            if use_cy_conn:
                quad_arr = np.asarray(quad[:4], dtype=np.int32)
                if cand_arr is None:
                    cand_arr = _cand_arr_from_candidates(candidates)
                keep_idx = _step3_conn_constraints_cy(cand_arr, quad_arr, frame_edges_arr, frame_reach3, template_adj_mask, template_reach2_mask)
                candidates = [candidates[i] for i in keep_idx]
                if cand_arr is not None:
                    cand_arr = cand_arr[keep_idx]
            else:
                candidates = [cand for cand in candidates if _check_connection_constraints(cand, quad)]
                cand_arr = None
            filtered: list[list[int]] = []
            for cand in candidates:
                cand_seq = list(cand) if isinstance(cand, tuple) else cand
                if _validate_y_ordering_partial(cand_seq, quad, frame_keypoints):
                    filtered.append(cand_seq)
            candidates = filtered
        return {'four_points': quad, 'candidates': candidates, 'candidates_count': len(candidates)}
    if len(frame_four_points) > 1:
        step3_workers = min(len(frame_four_points), TV_AF_EVAL_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=step3_workers) as ex:
            mapped = list(ex.map(_process_one_quad, frame_four_points))
    else:
        for quad in frame_four_points:
            entry = _process_one_quad(quad)
            mapped.append(entry)
    out = {'frame_id': step2_entry['frame_id'], 'connections': list(frame_connections), 'keypoints': frame_keypoints, 'frame_size': frame_size, 'labels': frame_labels, 'decision': decision, 'matches': mapped}
    return out

def _step4_pick_best_candidate(*, matches: list[dict[str, Any]], orig_kps: list[list[float]], frame_keypoints: list, frame_labels: list, decision: str | None, template_pts: np.ndarray, frame_number: int) -> tuple[float, dict[str, Any] | None, list[int] | None]:
    best_avg = float('inf')
    best_meta: dict[str, Any] | None = None
    best_orig_idx_map: list[int] | None = None
    STEP4_GOOD_CANDIDATE_THRESHOLD = 50.0
    good_candidate_found = False
    for match_idx, match in enumerate(matches):
        if good_candidate_found:
            break
        match_best_avg = float('inf')
        match_best_meta: dict[str, Any] | None = None
        four_points = match.get('four_points') or []
        candidates = match.get('candidates') or []
        if len(four_points) < 4:
            continue
        dst_pts_list = []
        for kp_idx in four_points[:4]:
            if kp_idx is None or kp_idx < 0 or kp_idx >= len(frame_keypoints):
                dst_pts_list = []
                break
            kp = frame_keypoints[kp_idx]
            if not kp or len(kp) < 2:
                dst_pts_list = []
                break
            dst_pts_list.append((float(kp[0]), float(kp[1])))
        if len(dst_pts_list) < 4:
            continue
        dst_pts = np.asarray(dst_pts_list, dtype=np.float32).reshape(1, 4, 2)
        candidates_to_eval = candidates[:STEP4_MAX_CANDIDATES] if len(candidates) > STEP4_MAX_CANDIDATES else candidates
        total_candidates = len(candidates)
        eval_count = len(candidates_to_eval)
        for cand_idx, cand in enumerate(candidates_to_eval):
            if len(cand) < 4:
                continue
            src_pts = _FOOTBALL_KEYPOINTS_NP[np.asarray(cand[:4], dtype=np.int32)].reshape(1, 4, 2)
            H, _mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
            if not _is_valid_homography(H):
                continue
            projected_corners = cv2.perspectiveTransform(_FOOTBALL_CORNERS_NP, H)[0]
            if _is_bowtie(projected_corners):
                continue
            projected = cv2.perspectiveTransform(template_pts, H)
            avg_dist, _nearest_d, reordered, orig_idx_map = _avg_distance_to_projection(orig_kps, projected, orig_labels=frame_labels, template_labels=KEYPOINT_LABELS)
            is_valid_y_ordering, violation_msg = _validate_y_ordering(reordered, debug_frame_id=frame_number, debug_candidate=cand[:4] if len(cand) >= 4 else [])
            if not is_valid_y_ordering:
                continue
            if avg_dist < match_best_avg:
                match_best_avg = avg_dist
                match_best_meta = {'frame_id': int(frame_number), 'match_idx': match_idx, 'candidate_idx': cand_idx, 'candidate': [int(x) for x in cand], 'avg_distance': avg_dist, 'reordered_keypoints': reordered, 'decision': decision, 'added_four_point': False}
                match_best_orig_idx_map = orig_idx_map
            if match_best_avg < STEP4_EARLY_TERMINATE_THRESHOLD:
                good_candidate_found = True
                break
        if match_best_meta is not None and match_best_avg < best_avg:
            best_avg = match_best_avg
            best_meta = match_best_meta
            best_orig_idx_map = locals().get('match_best_orig_idx_map')
        if match_best_meta is not None and match_best_avg < STEP4_GOOD_CANDIDATE_THRESHOLD:
            good_candidate_found = True
            break
        elif match_idx == 0:
            if match_best_meta is None:
                pass
            elif match_best_avg >= STEP4_GOOD_CANDIDATE_THRESHOLD:
                pass
        elif match_idx == 1:
            if match_best_meta is None:
                pass
            elif match_best_avg >= STEP4_GOOD_CANDIDATE_THRESHOLD:
                pass
    if best_meta and best_avg > 60:
        best_meta = None
    return (best_avg, best_meta, best_orig_idx_map)

def _ordered_keypoints_to_labeled(ordered_kps: list[list[float]], template_len: int) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for slot_idx in range(min(len(ordered_kps), template_len)):
        kp = ordered_kps[slot_idx]
        if kp and len(kp) >= 2 and (not (abs(kp[0]) < 1e-06 and abs(kp[1]) < 1e-06)):
            label_val = KEYPOINT_LABELS[slot_idx] if slot_idx < len(KEYPOINT_LABELS) else 0
            if label_val > 0:
                label_str = f'kpv{label_val:02d}'
                labeled.append({'id': slot_idx, 'x': float(kp[0]), 'y': float(kp[1]), 'label': label_str})
    return labeled

def _try_process_similar_frame_fast_path(*, fn: int, progress_prefix: str, frame: np.ndarray, frame_store: Any, kps: list[list[float]] | list[Any], labels: list[int], labeled_cur: list[dict[str, Any]], template_len: int, ordered_frames: list[dict[str, Any]], step1_outputs: list[dict[str, Any]] | None, step2_outputs: list[dict[str, Any]] | None, step3_outputs: list[dict[str, Any]] | None, step5_outputs: list[dict[str, Any]] | None, best_entries: list[dict[str, Any]], original_keypoints: dict[int, list[list[float]]], template_image: np.ndarray, template_pts_corrected: np.ndarray, template_pts: np.ndarray, cached_edges: np.ndarray | None, prev_valid_labeled: list[dict[str, Any]] | None, prev_valid_original_index: list[int] | None, prev_best_meta: dict[str, Any] | None, prev_valid_frame_id: int | None, log_fn: Any, push_fn: Any) -> tuple[bool, dict[str, Any]]:
    H, W = frame.shape[:2]
    similar_frame = False
    matches: list[tuple[int, int, float]] = []
    common_matches: list[tuple[int, int, float]] = []
    labeled_ref: list[dict[str, Any]] | None = None
    best_meta_prev = prev_best_meta
    if prev_valid_frame_id is not None and prev_valid_frame_id != fn - 1:
        best_meta_prev = None
    if not PREV_RELATIVE_FLAG:
        best_meta_prev = None
    if best_meta_prev is not None:
        prev_ordered_kps = best_meta_prev.get('reordered_keypoints')
        if prev_ordered_kps and isinstance(prev_ordered_kps, list):
            labeled_ref = _ordered_keypoints_to_labeled(prev_ordered_kps, template_len)
    if labeled_ref is not None and best_meta_prev is not None:
        _avg_dist, _match_count, matches = _avg_distance_same_label(labeled_cur, labeled_ref)
        valid_cur = [idx_lab for idx_lab, item in enumerate(labeled_cur) if item.get('x') is not None and item.get('y') is not None and (item.get('label') is not None)]
        common_matches = [(cur_idx, prev_idx, dist) for cur_idx, prev_idx, dist in matches if dist != float('inf')]
        prev_score = best_meta_prev.get('score')
        prev_score_ok = prev_score is not None and float(prev_score) >= SIMILARITY_MIN_SCORE_THRESHOLD
        if len(common_matches) >= 3 and all((dist < 30.0 for _, _, dist in common_matches)) and prev_score_ok:
            similar_frame = True
    if not similar_frame:
        return (False, {})
    log_fn(f'{progress_prefix} - similar to prev, reusing order')
    ordered_kps = [[0.0, 0.0] for _ in range(template_len)]
    orig_idx_map_current = [-1 for _ in range(template_len)]
    fallback_used = False
    prev_ordered_keypoints = best_meta_prev.get('reordered_keypoints') if best_meta_prev else None
    mapped_count = 0
    skipped_count = 0
    if prev_ordered_keypoints is None or len(prev_ordered_keypoints) != template_len:
        pass
    elif not common_matches:
        pass
    else:

        def _label_val(lab: Any) -> int | None:
            if lab is None:
                return None
            digits = ''.join((ch for ch in str(lab) if ch.isdigit()))
            return int(digits) if digits else None

        def _coord_for_id(orig_idx: int) -> list[float] | None:
            if 0 <= orig_idx < len(kps):
                kp = kps[orig_idx]
                if kp and len(kp) >= 2:
                    return [float(kp[0]), float(kp[1])]
            for item in labeled_cur:
                if item.get('id') == orig_idx:
                    x = item.get('x')
                    y = item.get('y')
                    if x is not None and y is not None:
                        return [float(x), float(y)]
            return None
        prev_idx_to_slot: dict[int, int] = {}
        for slot in range(template_len):
            prev_kp = prev_ordered_keypoints[slot]
            if prev_kp and len(prev_kp) >= 2 and (not (abs(prev_kp[0]) < 1e-06 and abs(prev_kp[1]) < 1e-06)):
                prev_coord = [float(prev_kp[0]), float(prev_kp[1])]
                best_prev_idx = None
                best_coord_dist = float('inf')
                for prev_idx, prev_item in enumerate(labeled_ref or []):
                    px = prev_item.get('x')
                    py = prev_item.get('y')
                    if px is None or py is None:
                        continue
                    prev_item_coord = [float(px), float(py)]
                    coord_dist = float(np.hypot(prev_coord[0] - prev_item_coord[0], prev_coord[1] - prev_item_coord[1]))
                    if coord_dist < 1.0 and coord_dist < best_coord_dist:
                        best_coord_dist = coord_dist
                        best_prev_idx = prev_idx
                if best_prev_idx is not None:
                    prev_idx_to_slot[best_prev_idx] = slot
        used_slots = set()
        sorted_matches = sorted(common_matches, key=lambda x: x[2])
        for cur_idx, prev_idx, match_dist in sorted_matches:
            if cur_idx < 0 or cur_idx >= len(labeled_cur):
                continue
            cur_item = labeled_cur[cur_idx]
            cur_x = cur_item.get('x')
            cur_y = cur_item.get('y')
            cur_orig_id = cur_item.get('id')
            cur_label = _label_val(cur_item.get('label'))
            if cur_x is None or cur_y is None or cur_label is None:
                continue
            if prev_idx not in prev_idx_to_slot:
                continue
            target_slot = prev_idx_to_slot[prev_idx]
            if target_slot in used_slots:
                continue
            cur_coord = None
            if cur_orig_id is not None:
                cur_coord = _coord_for_id(int(cur_orig_id))
            if cur_coord is None:
                cur_coord = [float(cur_x), float(cur_y)]
            ordered_kps[target_slot] = cur_coord
            orig_idx_map_current[target_slot] = int(cur_orig_id) if cur_orig_id is not None else -1
            used_slots.add(target_slot)
            mapped_count += 1
    step1_entry = {'frame_id': int(fn), 'frame_size': [int(W), int(H)], 'keypoints': [None if pt is None else [float(x) for x in pt] for pt in kps or []], 'labels': labels, 'connections': []}
    step2_entry = {'frame_id': int(fn), 'frame_size': [int(W), int(H)], 'connections': [], 'keypoints': step1_entry['keypoints'], 'labels': labels, 'four_points': []}
    step3_entry = {'frame_id': int(fn), 'connections': [], 'keypoints': step1_entry['keypoints'], 'frame_size': [int(W), int(H)], 'labels': labels, 'decision': None, 'matches': []}
    push_fn(step1_outputs, step1_entry)
    push_fn(step2_outputs, step2_entry)
    push_fn(step3_outputs, step3_entry)
    prev_decision = best_meta_prev.get('decision') if best_meta_prev else None
    best_meta: dict[str, Any] = {'frame_id': int(fn), 'match_idx': None, 'candidate_idx': None, 'candidate': [], 'avg_distance': 0.0, 'reordered_keypoints': ordered_kps, 'decision': prev_decision, 'similar_prev_frame': True, 'added_four_point': False}
    best_entries.append(best_meta)
    found = None
    for fr in ordered_frames:
        fid = fr.get('frame_id')
        if fid is None:
            fid = fr.get('frame_number')
        if fid is None:
            continue
        if int(fid) == int(fn):
            found = fr
            break
    if found is None:
        return (True, {})
    found['keypoints'] = ordered_kps
    found.pop('original_index', None)
    found['added_four_point'] = False
    fallback_used = False
    best_meta['step4_1'] = _step4_1_compute_border_line(frame=frame, ordered_kps=ordered_kps, frame_number=int(fn))
    best_meta['step4_2'] = {}
    best_meta['reordered_keypoints'] = ordered_kps
    if STEP4_3_ENABLED:
        best_meta['step4_3'] = _step4_3_debug_dilate_and_lines(frame=frame, ordered_kps=best_meta['reordered_keypoints'], frame_number=int(fn), decision=best_meta.get('decision'), cached_edges=cached_edges)
    else:
        best_meta['step4_3'] = {}
    step4_9_kps, step4_9_score = _step4_9_select_h_and_keypoints(input_kps=best_meta['reordered_keypoints'], step4_1=best_meta.get('step4_1'), step4_3=best_meta.get('step4_3'), frame=frame, frame_number=int(fn), decision=best_meta.get('decision'), cached_edges=cached_edges)
    if step4_9_kps is not None:
        best_meta['reordered_keypoints'] = step4_9_kps
        best_meta['score'] = step4_9_score
        ordered_kps = step4_9_kps
    validation_passed = False
    validation_error = None
    if STEP5_ENABLED and (not best_meta.get('added_four_point')):
        validation_passed, validation_error, ordered_kps = _step5_validate_ordered_keypoints(ordered_kps=ordered_kps, frame=frame, template_image=template_image, template_keypoints=FOOTBALL_KEYPOINTS, frame_number=int(fn), best_meta=best_meta, debug_label='similar frame path', cached_edges=cached_edges)
    best_meta['validation_passed'] = validation_passed
    if validation_error:
        best_meta['validation_error'] = validation_error
    if STEP6_ENABLED and validation_passed and (not fallback_used) and (best_meta.get('added_four_point') != True):
        ordered_kps = _step6_fill_keypoints_from_homography(ordered_kps=ordered_kps, frame=frame, frame_number=int(fn))
        best_meta['reordered_keypoints'] = ordered_kps
    if not validation_passed:
        orig_kps_local = original_keypoints.get(int(fn)) or []
        ordered_kps = adding_four_points(orig_kps_local, frame_store, fn, template_len, cached_edges=cached_edges, frame_img=frame)
        best_meta['reordered_keypoints'] = ordered_kps
        best_meta['fallback'] = True
        best_meta['added_four_point'] = True
        fallback_used = True
        if not validation_error:
            validation_error = 'Validation failed, used adding_four_points fallback'
            best_meta['validation_error'] = validation_error
    found['keypoints'] = ordered_kps
    found.pop('original_index', None)
    if best_meta.get('added_four_point') == True:
        found['added_four_point'] = True
    else:
        found['added_four_point'] = False
    if best_meta.get('added_four_point') != True:
        push_fn(step5_outputs, best_meta.copy())
    else:
        best_meta_copy = best_meta.copy()
        best_meta_copy['validation_passed'] = False
        best_meta_copy['validation_error'] = validation_error or 'Added four points (fallback)'
        push_fn(step5_outputs, best_meta_copy)
    state_updates: dict[str, Any] = {}
    state_updates['prev_labeled'] = labeled_cur
    state_updates['prev_original_index'] = orig_idx_map_current if not fallback_used else None
    if not best_meta.get('fallback'):
        state_updates['prev_valid_labeled'] = labeled_cur
        state_updates['prev_valid_original_index'] = orig_idx_map_current
        state_updates['prev_best_meta'] = best_meta.copy()
        state_updates['prev_valid_frame_id'] = fn
    else:
        state_updates['prev_valid_labeled'] = None
        state_updates['prev_valid_original_index'] = None
        state_updates['prev_best_meta'] = None
        state_updates['prev_valid_frame_id'] = None
    log_fn(f'{progress_prefix} - done')
    return (True, state_updates)

def _step7_interpolate_problematic_frames(*, step5_outputs: list[dict[str, Any]] | None, ordered_frames: list[dict[str, Any]], template_len: int, out_step7: Path) -> None:
    if not STEP7_ENABLED:
        return
    if step5_outputs is None:
        return
    if len(step5_outputs) == 0:
        return
    step5_outputs_serializable_for_step7: list[dict[str, Any]] = []
    for entry in step5_outputs:
        entry_copy = entry.copy()
        avg_dist = entry_copy.get('avg_distance')
        if avg_dist is not None and (avg_dist == float('inf') or np.isinf(avg_dist)):
            entry_copy['avg_distance'] = 99999
        elif avg_dist is not None and isinstance(avg_dist, (int, float)):
            entry_copy['avg_distance'] = round(float(avg_dist), 2)
        score = entry_copy.get('score')
        if score is not None and isinstance(score, (int, float)):
            entry_copy['score'] = round(float(score), 2)
        step5_outputs_serializable_for_step7.append(entry_copy)
    if not step5_outputs_serializable_for_step7:
        return
    step7_entries: list[dict[str, Any]] = []
    frame_map: dict[int, dict[str, Any]] = {}
    for entry in step5_outputs_serializable_for_step7:
        frame_id = entry.get('frame_id')
        if frame_id is not None:
            frame_map[int(frame_id)] = entry
    sorted_frame_ids = sorted(frame_map.keys())
    problematic_frames: list[int] = []
    for frame_id in sorted_frame_ids:
        entry = frame_map[frame_id]
        score = entry.get('score')
        added_four_point = entry.get('added_four_point', False)
        if score is not None and score < 0.7 or added_four_point:
            problematic_frames.append(frame_id)
    already_rewritten: set[int] = set()

    def count_valid_keypoints(kps: list[list[float]]) -> int:
        return sum((1 for kp in kps if kp and len(kp) >= 2 and (not (abs(kp[0]) < 1e-06 and abs(kp[1]) < 1e-06))))
    for problem_frame_id in problematic_frames:
        forward_good_frame = None
        for frame_id in sorted_frame_ids:
            if frame_id > problem_frame_id:
                entry = frame_map[frame_id]
                score = entry.get('score')
                added_four_point = entry.get('added_four_point', False)
                if score is not None and score >= 0.5 and (added_four_point == False):
                    forward_good_frame = frame_id
                    break
        backward_good_frame = None
        for frame_id in reversed(sorted_frame_ids):
            if frame_id < problem_frame_id:
                entry = frame_map[frame_id]
                score = entry.get('score')
                added_four_point = entry.get('added_four_point', False)
                if score is not None and score >= 0.5 and (added_four_point == False):
                    backward_good_frame = frame_id
                    break
        if forward_good_frame is None or backward_good_frame is None:
            continue
        frame_distance = forward_good_frame - backward_good_frame
        if frame_distance > 20:
            continue
        forward_entry = frame_map[forward_good_frame]
        backward_entry = frame_map[backward_good_frame]
        forward_kps = forward_entry.get('reordered_keypoints', [])
        backward_kps = backward_entry.get('reordered_keypoints', [])
        forward_valid = count_valid_keypoints(forward_kps)
        backward_valid = count_valid_keypoints(backward_kps)
        if not (forward_valid >= 4 and backward_valid >= 4):
            continue
        frames_to_check = [fid for fid in sorted_frame_ids if backward_good_frame <= fid <= forward_good_frame]
        consecutive_invalid_count = 0
        has_invalid_sequence = False
        for check_fid in frames_to_check:
            if check_fid in frame_map:
                check_entry = frame_map[check_fid]
                check_kps = check_entry.get('reordered_keypoints', [])
                check_valid = count_valid_keypoints(check_kps)
                if check_valid < 4:
                    consecutive_invalid_count += 1
                    if consecutive_invalid_count >= 2:
                        has_invalid_sequence = True
                        break
                else:
                    consecutive_invalid_count = 0
            else:
                consecutive_invalid_count += 1
                if consecutive_invalid_count >= 2:
                    has_invalid_sequence = True
                    break
        if has_invalid_sequence:
            continue
        common_indices: list[int] = []
        min_len = min(len(forward_kps), len(backward_kps))
        for i in range(min_len):
            f_kp = forward_kps[i] if i < len(forward_kps) else None
            b_kp = backward_kps[i] if i < len(backward_kps) else None
            if f_kp and len(f_kp) >= 2 and (not (abs(f_kp[0]) < 1e-06 and abs(f_kp[1]) < 1e-06)) and b_kp and (len(b_kp) >= 2) and (not (abs(b_kp[0]) < 1e-06 and abs(b_kp[1]) < 1e-06)):
                common_indices.append(i)
        if len(common_indices) < 4:
            continue
        frames_to_interpolate = [fid for fid in sorted_frame_ids if backward_good_frame < fid < forward_good_frame and fid not in already_rewritten]
        for interp_frame_id in frames_to_interpolate:
            if interp_frame_id not in frame_map:
                continue
            already_rewritten.add(interp_frame_id)
            interp_entry = frame_map[interp_frame_id].copy()
            interp_kps = interp_entry.get('reordered_keypoints', [])
            if forward_good_frame != backward_good_frame:
                weight = (interp_frame_id - backward_good_frame) / (forward_good_frame - backward_good_frame)
            else:
                weight = 0.0
            max_len = max(len(forward_kps), len(backward_kps), template_len)
            interpolated_kps: list[list[float]] = []
            common_set = set(common_indices)
            for i in range(max_len):
                if i in common_set:
                    f_kp = forward_kps[i] if i < len(forward_kps) else [0.0, 0.0]
                    b_kp = backward_kps[i] if i < len(backward_kps) else [0.0, 0.0]
                    interp_x = b_kp[0] * (1.0 - weight) + f_kp[0] * weight
                    interp_y = b_kp[1] * (1.0 - weight) + f_kp[1] * weight
                    interpolated_kps.append([float(interp_x), float(interp_y)])
                else:
                    interpolated_kps.append([0.0, 0.0])
            step7_entries.append({'frame_id': interp_frame_id, 'backward_frame_id': backward_good_frame, 'forward_frame_id': forward_good_frame, 'interpolation_weight': round(weight, 4), 'common_keypoint_indices': common_indices, 'interpolated_keypoints': interpolated_kps, 'original_keypoints': interp_kps})
    step7_dict: dict[int, dict[str, Any]] = {}
    for entry in step7_entries:
        frame_id = entry.get('frame_id')
        if frame_id is not None:
            step7_dict[int(frame_id)] = entry
    step7_final = list(step7_dict.values())
    if not step7_final:
        print('Step 7: No frames needed interpolation (no problematic frames found or no valid good frames)')
        return
    step7_keypoints_map: dict[int, list[list[float]]] = {}
    for entry in step7_final:
        frame_id = entry.get('frame_id')
        interpolated_kps = entry.get('interpolated_keypoints')
        if frame_id is not None and interpolated_kps is not None:
            step7_keypoints_map[int(frame_id)] = interpolated_kps
    updated_count = 0
    for fr in ordered_frames:
        fid = fr.get('frame_id')
        if fid is None:
            fid = fr.get('frame_number')
        if fid is None:
            continue
        fid_int = int(fid)
        if fid_int in step7_keypoints_map:
            fr['keypoints'] = step7_keypoints_map[fid_int]
            updated_count += 1
_TEMPLATE_LOAD_ERROR: float | None = None

@functools.lru_cache(maxsize=1)
def challenge_template() -> np.ndarray:
    global _TEMPLATE_LOAD_ERROR
    template_path = Path(__file__).parent / 'football_pitch_template.png'
    if not template_path.exists():
        _TEMPLATE_LOAD_ERROR = 100.1
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    img = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if img is None:
        _TEMPLATE_LOAD_ERROR = 100.2
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    if img.sum() == 0:
        _TEMPLATE_LOAD_ERROR = 100.3
        return img
    _TEMPLATE_LOAD_ERROR = None
    return img

@functools.lru_cache(maxsize=1)
def _kernel_rect_31() -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))

@functools.lru_cache(maxsize=1)
def _kernel_rect_3() -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

@functools.lru_cache(maxsize=1)
def _kernel_ellipse_5() -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

@functools.lru_cache(maxsize=1)
def _get_shared_template_precomputations() -> tuple[list[list[int]], list[int], dict[tuple[bool, ...], list[list[int]]], dict[tuple[bool, ...], list[list[int]]], dict[tuple[bool, ...], list[list[int]]], np.ndarray, np.ndarray, int, np.ndarray]:
    template_adj = KEYPOINT_CONNECTIONS
    template_labels = KEYPOINT_LABELS
    template_patterns = _build_template_patterns(template_adj)
    allowed_left = set(range(0, 17)) | {30, 31}
    allowed_right = set(range(13, 32))
    template_patterns_allowed_left: dict[tuple[bool, ...], list[list[int]]] = {}
    template_patterns_allowed_right: dict[tuple[bool, ...], list[list[int]]] = {}
    for pat, cand_list in template_patterns.items():
        template_patterns_allowed_left[pat] = [cand for cand in cand_list if all((int(node) in allowed_left for node in cand))]
        template_patterns_allowed_right[pat] = [cand for cand in cand_list if all((int(node) in allowed_right for node in cand))]
    template_pts = _as_np_points(FOOTBALL_KEYPOINTS)
    template_pts_corrected = _as_np_points(FOOTBALL_KEYPOINTS_CORRECTED)
    template_len = len(FOOTBALL_KEYPOINTS)
    template_image = challenge_template()
    return (template_adj, template_labels, template_patterns, template_patterns_allowed_left, template_patterns_allowed_right, template_pts, template_pts_corrected, template_len, template_image)

@functools.lru_cache(maxsize=1)
def _get_all_template_points_np() -> np.ndarray:
    return np.array(FOOTBALL_KEYPOINTS, dtype=np.float32).reshape(-1, 1, 2)

class FrameStore:

    def __init__(self, source: str) -> None:
        self.cap = cv2.VideoCapture(source)
        self.video_path = source
        self._last_frame_id: int | None = None
        self._last_frame: np.ndarray | None = None

    def get_frame(self, frame_id: int) -> np.ndarray:
        if self._last_frame_id is not None and int(frame_id) == int(self._last_frame_id) + 1:
            ok, frame = self.cap.read()
        elif self._last_frame_id is None and int(frame_id) == 0:
            ok, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f'Could not read frame {frame_id}')
        self._last_frame_id = int(frame_id)
        self._last_frame = frame
        return frame

    def unlink(self) -> None:
        if self.cap:
            self.cap.release()

class SVRunOutput:

    def __init__(self, success: bool=True, latency_ms: float=0.0, predictions: Any | None=None, error: Any | None=None, model: Any | None=None) -> None:
        self.success = success
        self.latency_ms = latency_ms
        self.predictions = predictions
        self.error = error
        self.model = model
PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
FOOTBALL_KEYPOINTS: list[tuple[int, int]] = [(5, 5), (5, 140), (5, 250), (5, 430), (5, 540), (5, 675), (55, 250), (55, 430), (110, 340), (165, 140), (165, 270), (165, 410), (165, 540), (527, 5), (527, 253), (527, 433), (527, 675), (888, 140), (888, 270), (888, 410), (888, 540), (940, 340), (998, 250), (998, 430), (1045, 5), (1045, 140), (1045, 250), (1045, 430), (1045, 540), (1045, 675), (435, 340), (615, 340)]
FOOTBALL_KEYPOINTS_CORRECTED: list[tuple[int, int]] = [(2.5, 2.5), (2.5, 139.5), (2.5, 249.5), (2.5, 430.5), (2.5, 540.5), (2.5, 678), (54.5, 249.5), (54.5, 430.5), (110.5, 340.5), (164.5, 139.5), (164.5, 269), (164.5, 411), (164.5, 540.5), (525, 2.5), (525, 249.5), (525, 430.5), (525, 678), (886.5, 139.5), (886.5, 269), (886.5, 411), (886.5, 540.5), (940.5, 340.5), (998, 249.5), (998, 430.5), (1048, 2.5), (1048, 139.5), (1048, 249.5), (1048, 430.5), (1048, 540.5), (1048, 678), (434.5, 340), (615.5, 340)]
INDEX_KEYPOINT_CORNER_BOTTOM_LEFT = 5
INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT = 29
INDEX_KEYPOINT_CORNER_TOP_LEFT = 0
INDEX_KEYPOINT_CORNER_TOP_RIGHT = 24
FOOTBALL_KEYPOINTS_TUPLES: list[tuple[int, int]] = [(int(kp[0]), int(kp[1])) for kp in FOOTBALL_KEYPOINTS]
_FOOTBALL_KEYPOINTS_NP = np.array(FOOTBALL_KEYPOINTS, dtype=np.float32)
_FOOTBALL_CORNERS_NP = np.array([FOOTBALL_KEYPOINTS[INDEX_KEYPOINT_CORNER_BOTTOM_LEFT], FOOTBALL_KEYPOINTS[INDEX_KEYPOINT_CORNER_BOTTOM_RIGHT], FOOTBALL_KEYPOINTS[INDEX_KEYPOINT_CORNER_TOP_RIGHT], FOOTBALL_KEYPOINTS[INDEX_KEYPOINT_CORNER_TOP_LEFT]], dtype=np.float32).reshape(1, 4, 2)
KEYPOINT_CONNECTIONS = [[1, 13], [0, 2, 9], [1, 3, 6], [2, 4, 7], [3, 5, 12], [4, 16], [2, 7], [3, 6], [], [1, 10], [9, 11], [10, 12], [4, 11], [0, 14, 24], [13, 15], [14, 16], [5, 15, 29], [18, 25], [17, 19], [18, 20], [19, 28], [], [23, 26], [22, 27], [13, 25], [17, 24, 26], [22, 25, 27], [23, 26, 28], [20, 27, 29], [16, 28], [], []]
KEYPOINT_LABELS = [1, 2, 2, 2, 2, 1, 3, 3, 0, 3, 2, 2, 3, 5, 6, 6, 5, 3, 2, 2, 3, 0, 3, 3, 1, 2, 2, 2, 2, 1, 4, 4]
Y_ORDERING_RULES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [9, 10], [10, 11], [12, 13], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [22, 23], [24, 25], [25, 26], [27, 28], [28, 29]]

def _as_np_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array([[float(x), float(y)] for x, y in points], dtype=np.float32)
    return arr.reshape(-1, 1, 2)

def _validate_y_ordering(ordered_keypoints: list[list[float]], debug_frame_id: int | None=None, debug_candidate: list[int] | None=None) -> tuple[bool, str | None]:
    if not ordered_keypoints or len(ordered_keypoints) < 2:
        return (True, None)
    for rule_idx, rule_jdx in Y_ORDERING_RULES:
        if rule_idx >= len(ordered_keypoints) or rule_jdx >= len(ordered_keypoints):
            continue
        kp_i = ordered_keypoints[rule_idx]
        kp_j = ordered_keypoints[rule_jdx]
        if not kp_i or len(kp_i) < 2 or (abs(kp_i[0]) < 1e-06 and abs(kp_i[1]) < 1e-06):
            continue
        if not kp_j or len(kp_j) < 2 or (abs(kp_j[0]) < 1e-06 and abs(kp_j[1]) < 1e-06):
            continue
        y_i = float(kp_i[1])
        y_j = float(kp_j[1])
        if y_i >= y_j:
            violation_msg = f'y-ordering rule [{rule_idx}, {rule_jdx}] violated: y[{rule_idx}]={y_i:.2f} >= y[{rule_jdx}]={y_j:.2f}'
            return (False, violation_msg)
    return (True, None)

def _validate_y_ordering_partial(candidate: Sequence[int], quad: Sequence[int], frame_keypoints: Sequence[Sequence[float]]) -> bool:
    if len(candidate) < 4 or len(quad) < 4:
        return True
    template_to_frame = {}
    for i in range(4):
        template_idx = candidate[i]
        frame_idx = quad[i]
        template_to_frame[template_idx] = frame_idx
    for rule_i, rule_j in Y_ORDERING_RULES:
        if rule_i not in template_to_frame or rule_j not in template_to_frame:
            continue
        frame_idx_i = template_to_frame[rule_i]
        frame_idx_j = template_to_frame[rule_j]
        if frame_idx_i < 0 or frame_idx_i >= len(frame_keypoints):
            continue
        if frame_idx_j < 0 or frame_idx_j >= len(frame_keypoints):
            continue
        kp_i = frame_keypoints[frame_idx_i]
        kp_j = frame_keypoints[frame_idx_j]
        if not kp_i or len(kp_i) < 2 or (abs(kp_i[0]) < 1e-06 and abs(kp_i[1]) < 1e-06):
            continue
        if not kp_j or len(kp_j) < 2 or (abs(kp_j[0]) < 1e-06 and abs(kp_j[1]) < 1e-06):
            continue
        y_i = float(kp_i[1])
        y_j = float(kp_j[1])
        if y_i >= y_j:
            return False
    return True

def _load_miner_predictions(path: Path, *, frame_width: int | None=None, frame_height: int | None=None, border_margin_px: float=50.0, frame_store: Any | None=None) -> dict[int, dict[str, Any]]:
    data = json.loads(path.read_text())
    return _load_miner_predictions_from_obj(data, frame_width=frame_width, frame_height=frame_height, border_margin_px=border_margin_px, frame_store=frame_store)

def _load_miner_predictions_from_obj(data: Any, *, frame_width: int | None=None, frame_height: int | None=None, border_margin_px: float=50.0, frame_store: Any | None=None) -> dict[int, dict[str, Any]]:
    if isinstance(data, list):
        raw_frames = data
        payload = {'success': True, 'latency_ms': 0.0, 'predictions': {'frames': raw_frames}, 'error': None, 'model': None}
    else:
        raw_frames = data.get('frames') or (data.get('predictions') or {}).get('frames') or []
        payload = {'success': data.get('success', True), 'latency_ms': data.get('latency_ms', data.get('latency', 0.0) or 0.0), 'predictions': data.get('predictions'), 'error': data.get('error'), 'model': data.get('model')}
    labeled_lookup: dict[int, list[dict[str, Any]]] = {}
    for fr in raw_frames:
        fid = fr.get('frame_id')
        if fid is None:
            fid = fr.get('frame_number')
        if fid is None:
            continue
        labeled_lookup[int(fid)] = fr.get('keypoints_labeled') or []
    miner_run = SVRunOutput(**payload)
    parsed = parse_miner_prediction(miner_run=miner_run)
    template_len = len(FOOTBALL_KEYPOINTS)
    for frame_id, entry in parsed.items():
        kps = entry.get('keypoints')
        has_valid = bool(kps and any((pt and len(pt) >= 2 and (not (pt[0] == 0 and pt[1] == 0)) for pt in kps)))
        if has_valid:
            continue
        labeled = labeled_lookup.get(int(frame_id)) or []
        kps_filled = [[0.0, 0.0] for _ in range(template_len)]
        labels_filled = [0 for _ in range(template_len)]
        for item in labeled:
            idx = item.get('id')
            x = item.get('x')
            y = item.get('y')
            lab = item.get('label')
            if idx is None or x is None or y is None:
                continue
            if 0 <= int(idx) < template_len:
                kps_filled[int(idx)] = [float(x), float(y)]
                if lab:
                    digits = ''.join((ch for ch in str(lab) if ch.isdigit()))
                    val = int(digits) if digits else 0
                    labels_filled[int(idx)] = val if 0 <= val <= 6 else 0
        label1_indices = [i for i in range(len(labels_filled)) if labels_filled[i] == 1 and kps_filled[i] and (len(kps_filled[i]) >= 2) and (not (abs(kps_filled[i][0]) < 1e-06 and abs(kps_filled[i][1]) < 1e-06))]
        label2_indices = [j for j in range(len(labels_filled)) if labels_filled[j] == 2 and kps_filled[j] and (len(kps_filled[j]) >= 2) and (not (abs(kps_filled[j][0]) < 1e-06 and abs(kps_filled[j][1]) < 1e-06))]
        if label1_indices and label2_indices:
            for i in label1_indices:
                yi = float(kps_filled[i][1])
                for j in label2_indices:
                    yj = float(kps_filled[j][1])
                    if yi > yj:
                        labels_filled[i] = 2
                        break
        entry['keypoints'] = kps_filled
        entry['labels'] = labels_filled
    if BORDER_50PX_REMOVE_FLAG and frame_width is not None and (frame_height is not None):
        W = int(frame_width)
        H = int(frame_height)
        m = float(border_margin_px)

        def _in_border(x: float, y: float) -> bool:
            return bool(x < m or x >= float(W) - m or y < m or (y >= float(H) - m))
        removed_total = 0
        for frame_id, entry in parsed.items():
            kps = entry.get('keypoints') or []
            labels = entry.get('labels')
            removed_this = 0
            for i, pt in enumerate(kps):
                if not pt or len(pt) < 2:
                    continue
                try:
                    x = float(pt[0])
                    y = float(pt[1])
                except Exception:
                    continue
                if abs(x) < 1e-06 and abs(y) < 1e-06:
                    continue
                if _in_border(x, y):
                    kps[i] = [0.0, 0.0]
                    if isinstance(labels, list) and i < len(labels):
                        labels[i] = 0
                    removed_this += 1
            labeled = entry.get('keypoints_labeled')
            if isinstance(labeled, list):
                for item in labeled:
                    if not isinstance(item, dict):
                        continue
                    x = item.get('x')
                    y = item.get('y')
                    if x is None or y is None:
                        continue
                    try:
                        xf = float(x)
                        yf = float(y)
                    except Exception:
                        continue
                    if _in_border(xf, yf):
                        item['x'] = None
                        item['y'] = None
                        item['label'] = None
            if removed_this > 0:
                removed_total += removed_this
                entry['keypoints'] = kps
                if isinstance(labels, list):
                    entry['labels'] = labels
    return parsed

def _avg_distance_same_label(cur_labeled: list[dict[str, Any]], prev_labeled: list[dict[str, Any]]) -> tuple[float, int, list[tuple[int, int, float]]]:

    def _label_val(lab: Any) -> int | None:
        if lab is None:
            return None
        digits = ''.join((ch for ch in str(lab) if ch.isdigit()))
        return int(digits) if digits else None
    prev_by_label: dict[int, list[tuple[int, float, float]]] = {}
    for idx_prev, item in enumerate(prev_labeled or []):
        lx = item.get('x')
        ly = item.get('y')
        lab = _label_val(item.get('label'))
        if lab is None or lx is None or ly is None:
            continue
        prev_by_label.setdefault(lab, []).append((idx_prev, float(lx), float(ly)))
    cur_by_label: dict[int, list[tuple[int, float, float]]] = {}
    for idx_cur, item in enumerate(cur_labeled or []):
        cx = item.get('x')
        cy = item.get('y')
        lab = _label_val(item.get('label'))
        if lab is None or cx is None or cy is None:
            continue
        cur_by_label.setdefault(lab, []).append((idx_cur, float(cx), float(cy)))
    total = 0.0
    count = 0
    matches: list[tuple[int, int, float]] = []
    valid_cur = 0
    for label, cur_list in cur_by_label.items():
        prev_list = prev_by_label.get(label, [])
        if not prev_list:
            for cur_idx, _, _ in cur_list:
                matches.append((cur_idx, -1, float('inf')))
                total += float('inf')
                valid_cur += 1
            continue
        cur_remaining = cur_list.copy()
        prev_remaining = prev_list.copy()
        used_prev_indices = set()
        while cur_remaining and prev_remaining:
            min_dist = float('inf')
            best_cur_idx = None
            best_prev_idx = None
            best_cur_pos = None
            best_prev_pos = None
            for cur_idx, cx, cy in cur_remaining:
                for prev_idx, px, py in prev_remaining:
                    if prev_idx in used_prev_indices:
                        continue
                    dist = float(np.hypot(cx - px, cy - py))
                    if dist < min_dist:
                        min_dist = dist
                        best_cur_idx = cur_idx
                        best_prev_idx = prev_idx
                        best_cur_pos = (cx, cy)
                        best_prev_pos = (px, py)
            if best_cur_idx is not None and best_prev_idx is not None:
                matches.append((best_cur_idx, best_prev_idx, min_dist))
                total += min_dist
                count += 1
                valid_cur += 1
                cur_remaining = [(idx, x, y) for idx, x, y in cur_remaining if idx != best_cur_idx]
                used_prev_indices.add(best_prev_idx)
                prev_remaining = [(idx, x, y) for idx, x, y in prev_remaining if idx != best_prev_idx]
            else:
                break
        for cur_idx, _, _ in cur_remaining:
            matches.append((cur_idx, -1, float('inf')))
            total += float('inf')
            valid_cur += 1
    for idx_cur, item in enumerate(cur_labeled or []):
        cx = item.get('x')
        cy = item.get('y')
        lab = _label_val(item.get('label'))
        if cx is None or cy is None:
            continue
        if lab is None:
            if not any((m[0] == idx_cur for m in matches)):
                matches.append((idx_cur, -1, float('inf')))
                total += float('inf')
                valid_cur += 1
    effective_count = valid_cur
    if effective_count == 0:
        return (float('inf'), 0, matches)
    return (total / float(effective_count), count, matches)

def _connected_by_segment_py(mask: np.ndarray, p1: tuple[float, float], p2: tuple[float, float], sample_radius: int=5, close_ksize: int=5, min_hit_ratio: float=0.35, max_gap_px: int=20) -> tuple[bool, float, int]:
    m = (mask > 0).astype(np.uint8) * 255
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    H, W = m.shape[:2]
    L = int(np.hypot(x2 - x1, y2 - y1))
    if L < 2:
        return (False, 0.0, L)
    xs = np.linspace(x1, x2, L + 1)
    ys = np.linspace(y1, y2, L + 1)
    hits = []
    r = int(sample_radius)
    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))
        x0 = max(0, xi - r)
        x1b = min(W, xi + r + 1)
        y0 = max(0, yi - r)
        y1b = min(H, yi + r + 1)
        hits.append(np.any(m[y0:y1b, x0:x1b] > 0))
    hit_ratio = float(np.mean(hits))
    longest = cur = 0
    for h in hits:
        if not h:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    ok = hit_ratio >= min_hit_ratio and longest <= max_gap_px
    return (ok, hit_ratio, longest)

def _connected_by_segment(mask: np.ndarray, p1: tuple[float, float], p2: tuple[float, float], sample_radius: int=5, close_ksize: int=5, min_hit_ratio: float=0.35, max_gap_px: int=20, sample_step: int=1) -> tuple[bool, float, int]:
    if _connected_by_segment_cy is not None and close_ksize == 0:
        m = mask
        if m.dtype != np.uint8:
            m = (m > 0).astype(np.uint8) * 255
        if not m.flags['C_CONTIGUOUS']:
            m = np.ascontiguousarray(m)
        return _connected_by_segment_cy(m, p1, p2, int(sample_radius), 0, float(min_hit_ratio), int(max_gap_px), int(sample_step))
    return _connected_by_segment_py(mask, p1, p2, sample_radius=sample_radius, close_ksize=close_ksize, min_hit_ratio=min_hit_ratio, max_gap_px=max_gap_px)

def _connection_centroid(connections: Iterable[Iterable[int]], keypoints: list | None) -> tuple[float | None, float | None]:
    if not keypoints:
        return (None, None)
    xs: list[float] = []
    ys: list[float] = []
    for a, b in connections:
        for idx in (a, b):
            if idx is None or idx < 0 or idx >= len(keypoints):
                continue
            pt = keypoints[idx]
            if not pt or len(pt) < 2:
                continue
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
    if not xs or not ys:
        return (None, None)
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def _edge_set_from_connections(conns: Iterable[Iterable[int]]) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in conns}

def _edge_exists(u: int, v: int, edge_set: set[tuple[int, int]]) -> bool:
    return tuple(sorted((u, v))) in edge_set

def _pattern_for_quad(quad: list[int], edge_set: set[tuple[int, int]]) -> tuple[bool, ...]:
    return tuple((_edge_exists(quad[i], quad[j], edge_set) for i, j in PAIR_ORDER))

def _build_template_patterns(template_adj: list[list[int]]) -> dict[tuple[bool, ...], list[list[int]]]:
    template_edges = _edge_set_from_connections(((i, nbr) for i, nbrs in enumerate(template_adj) for nbr in nbrs))
    patterns: dict[tuple[bool, ...], list[list[int]]] = {}
    nodes = list(range(len(template_adj)))
    for quad in permutations(nodes, 4):
        quad_list = list(quad)
        pattern = _pattern_for_quad(quad_list, template_edges)
        patterns.setdefault(pattern, []).append(tuple(map(int, quad_list)))
    return patterns

def _build_frame_reach3_bitset(connection_graph: dict[int, set[int]], n_nodes: int) -> np.ndarray | None:
    if n_nodes <= 0 or n_nodes > 63:
        return None
    reach = np.zeros(n_nodes, dtype=np.uint64)
    for start in range(n_nodes):
        if start not in connection_graph:
            continue
        q = deque([(start, 0)])
        visited = {start}
        mask = 0
        while q:
            node, depth = q.popleft()
            if depth >= 3:
                continue
            for nb in connection_graph.get(node, set()):
                if nb in visited or nb < 0 or nb >= n_nodes:
                    continue
                visited.add(nb)
                mask |= 1 << nb
                q.append((nb, depth + 1))
        reach[start] = mask
    return reach

def _build_frame_adj_bitset(connection_graph: dict[int, set[int]], n_nodes: int) -> np.ndarray | None:
    if n_nodes <= 0 or n_nodes > 63:
        return None
    adj = np.zeros(n_nodes, dtype=np.uint64)
    for node, nbrs in connection_graph.items():
        if node < 0 or node >= n_nodes:
            continue
        mask = 0
        for nb in nbrs:
            if 0 <= int(nb) < 64:
                mask |= 1 << int(nb)
        adj[node] = mask
    return adj
_STEP3_PATTERN_CACHE: dict[str, OrderedDict[tuple[tuple[bool, ...], ...], tuple[list[tuple[int, ...]], np.ndarray, int, int]]] = {'all': OrderedDict(), 'left': OrderedDict(), 'right': OrderedDict()}
_STEP3_CACHE_LOCK = threading.Lock()

def _get_step3_candidates_cached(*, cache_key: str, pattern_key: tuple[tuple[bool, ...], ...], patterns_src: dict[tuple[bool, ...], list]) -> tuple[list[tuple[int, ...]], np.ndarray, int, int]:
    with _STEP3_CACHE_LOCK:
        cache = _STEP3_PATTERN_CACHE[cache_key]
        cached = cache.pop(pattern_key, None)
        if cached is not None:
            cache[pattern_key] = cached
            return cached
        candidates_all: list[tuple[int, ...]] = []
        for pat in pattern_key:
            candidates_all.extend(patterns_src.get(pat, []))
        if len(pattern_key) == 1:
            candidates = list(candidates_all)
            cand_unique_len = len(candidates)
        else:
            seen: set[tuple[int, ...]] = set()
            candidates = []
            for cand in candidates_all:
                tup = cand if isinstance(cand, tuple) else tuple(cand)
                if tup in seen:
                    continue
                seen.add(tup)
                candidates.append(tup)
            cand_unique_len = len(candidates)
        cand_all_len = len(candidates_all)
        cand_arr = np.asarray(candidates, dtype=np.int32)
        if cand_arr.ndim != 2 or cand_arr.shape[1] < 4:
            cand_arr = np.asarray([c[:4] for c in candidates], dtype=np.int32)
        elif cand_arr.shape[1] > 4:
            cand_arr = cand_arr[:, :4]
        out = (candidates, cand_arr, cand_all_len, cand_unique_len)
        cache[pattern_key] = out
        if len(cache) > STEP3_PATTERN_CACHE_MAX:
            cache.popitem(last=False)
        return out

@functools.lru_cache(maxsize=1)
def _get_template_reach_bitmasks() -> tuple[np.ndarray, np.ndarray]:
    n = len(KEYPOINT_CONNECTIONS)
    adj_mask = np.zeros(n, dtype=np.uint64)
    for i, nbrs in enumerate(KEYPOINT_CONNECTIONS):
        mask = 0
        for j in nbrs:
            if 0 <= int(j) < 64:
                mask |= 1 << int(j)
        adj_mask[i] = mask
    reach2 = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        mask = int(adj_mask[i])
        reach = int(mask)
        while mask:
            lsb = mask & -mask
            idx = int(lsb.bit_length() - 1)
            reach |= adj_mask[idx]
            mask &= mask - 1
        reach2[i] = reach
    return (adj_mask, reach2)

@functools.lru_cache(maxsize=1)
def _get_template_neighbor_label_mask() -> np.ndarray:
    n = len(KEYPOINT_CONNECTIONS)
    label_mask = np.zeros(n, dtype=np.uint64)
    for i, nbrs in enumerate(KEYPOINT_CONNECTIONS):
        mask = 0
        for j in nbrs:
            if 0 <= int(j) < len(KEYPOINT_LABELS):
                lab = int(KEYPOINT_LABELS[j])
                if lab > 0:
                    mask |= 1 << lab
        label_mask[i] = mask
    return label_mask

def _avg_distance_to_projection(orig_kps: list[list[float]], projected: np.ndarray, orig_labels: list[int] | None=None, template_labels: list[int] | None=None) -> tuple[float, list[int], list[list[float]], list[int]]:
    proj_pts = projected.reshape(-1, 2)
    nearest = [-1 for _ in proj_pts]
    reordered = [[0.0, 0.0] for _ in proj_pts]
    orig_idx_map = [-1 for _ in proj_pts]
    total = 0.0
    count = 0
    use_label_check = orig_labels is not None and template_labels is not None and (len(orig_labels) == len(orig_kps)) and (len(template_labels) == len(proj_pts))
    for orig_idx, orig in enumerate(orig_kps):
        if not orig or len(orig) < 2:
            continue
        if orig[0] == 0 and orig[1] == 0:
            continue
        orig_label = orig_labels[orig_idx] if use_label_check and orig_idx < len(orig_labels) else None
        o = np.array(orig[:2], dtype=np.float32)
        dists = np.linalg.norm(proj_pts - o, axis=1)
        if use_label_check and orig_label is not None:
            valid_slots = [j for j in range(len(proj_pts)) if template_labels[j] == orig_label]
            if not valid_slots:
                continue
            valid_dists = [dists[j] for j in valid_slots]
            min_valid_idx = int(np.argmin(valid_dists))
            j = valid_slots[min_valid_idx]
        else:
            j = int(np.argmin(dists))
        total += float(dists[j])
        count += 1
        if nearest[j] == -1 or float(dists[j]) < nearest[j]:
            nearest[j] = float(dists[j])
            reordered[j] = [float(orig[0]), float(orig[1])]
            orig_idx_map[j] = int(orig_idx)
    avg = total / count if count else float('inf')
    return (avg, nearest, reordered, orig_idx_map)

def _is_valid_homography(H: np.ndarray | None) -> bool:
    if H is None or H.shape != (3, 3):
        return False
    if H.dtype != np.float32 and H.dtype != np.float64:
        H = H.astype(np.float64)
    if not np.all(np.isfinite(H)):
        return False
    det = float(np.linalg.det(H))
    if abs(det) < 1e-08:
        return False
    return True

def adding_four_points(orig_kps: list[list[float]], frame_store, frame_id: int, template_len: int, cached_edges: np.ndarray | None=None, frame_img: np.ndarray | None=None) -> list[list[float]]:
    import os
    import time
    import json
    try:
        frame_id = int(frame_id)
    except Exception:
        pass
    _profile = False
    if frame_img is None:
        frame_img = frame_store.get_frame(frame_id)
    frame_height, frame_width = frame_img.shape[:2]
    result = [[0.0, 0.0] for _ in range(template_len)]
    if not ADDING_FOUR_POINTS_ENABLED:
        d = float(frame_width) / 20.0
        cx = float(frame_width) / 2.0
        cy = float(frame_height) / 2.0

        def _clamp(x: float, y: float) -> list[float]:
            return [min(max(float(x), 0.0), frame_width - 1.0), min(max(float(y), 0.0), frame_height - 1.0)]
        result[0] = (100.0, 100.0)
        result[5] = (100.0, 400.0)
        result[24] = (101.0, 100.0)
        result[29] = (101.0, 400.0)
        return result
    if cached_edges is None:
        gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        gray_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _kernel_rect_31())
        gray_blur = cv2.GaussianBlur(gray_tophat, (5, 5), 0)
        edges = cv2.Canny(gray_blur, 30, 100)
    else:
        edges = cached_edges
    binary_edges = (edges > 0).astype(np.uint8)
    ii = cv2.integral(binary_edges)
    row_prefix = (ii[1:, 1:] - ii[:-1, 1:]).astype(np.int32)
    col_prefix = (ii[1:, 1:] - ii[1:, :-1]).astype(np.int32)
    row_sums = row_prefix[:, -1] if frame_width > 0 else np.zeros((frame_height,), dtype=np.int32)
    col_sums = col_prefix[-1, :] if frame_height > 0 else np.zeros((frame_width,), dtype=np.int32)
    _edges_flat_cy = np.ascontiguousarray(binary_edges, dtype=np.uint8).ravel() if _search_horizontal_in_area_cy is not None or _search_vertical_in_area_cy is not None else None
    _integral_flat_cy = np.ascontiguousarray(ii, dtype=np.float64).ravel() if _search_horizontal_in_area_integral_cy is not None and _search_vertical_in_area_integral_cy is not None else None

    def _sum_rect(y0: int, y1: int, x0: int, x1: int) -> int:
        if y1 <= y0 or x1 <= x0:
            return 0
        return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])

    def _rolling_sum_1d(arr: np.ndarray, window: int) -> np.ndarray:
        n = int(arr.shape[0])
        if n < window or window <= 0:
            return np.zeros((0,), dtype=np.int64)
        c = np.cumsum(arr, dtype=np.int64)
        return c[window - 1:] - np.concatenate((np.array([0], dtype=np.int64), c[:-window]))
    SLOPING_LINE_WIDTH = 10
    SLOPING_LINE_HALF_WIDTH = SLOPING_LINE_WIDTH // 2
    SLOPING_LINE_SAMPLE_MAX = 128

    def _sloping_line_white_count(ax: float, ay: float, bx: float, by: float) -> tuple[int, int]:
        L = float(np.hypot(bx - ax, by - ay))
        if L < 1.0:
            return (0, 0)
        n_steps = min(max(1, int(L) + 1), SLOPING_LINE_SAMPLE_MAX)
        t = np.linspace(0, 1, n_steps, dtype=np.float64)
        px = ax + t * (bx - ax)
        py = ay + t * (by - ay)
        perp_x = -(by - ay) / L
        perp_y = (bx - ax) / L
        k = np.arange(-SLOPING_LINE_HALF_WIDTH, SLOPING_LINE_HALF_WIDTH + 1, dtype=np.float64)
        qx_flat = np.round(px[np.newaxis, :] + k[:, np.newaxis] * perp_x).astype(np.int32).ravel()
        qy_flat = np.round(py[np.newaxis, :] + k[:, np.newaxis] * perp_y).astype(np.int32).ravel()
        valid = (qx_flat >= 0) & (qx_flat < frame_width) & (qy_flat >= 0) & (qy_flat < frame_height)
        qx_flat = qx_flat[valid]
        qy_flat = qy_flat[valid]
        if qx_flat.size == 0:
            return (0, 0)
        linear = np.unique(np.asarray(qy_flat, dtype=np.int64) * frame_width + np.asarray(qx_flat, dtype=np.int64))
        total = int(linear.size)
        white = int(np.count_nonzero(np.take(binary_edges.ravel(), linear)))
        return (white, total)

    def _sloping_line_white_count_band(ax: float, ay: float, bx: float, by: float, half_width: int, edges_2d: np.ndarray) -> tuple[int, int]:
        L = float(np.hypot(bx - ax, by - ay))
        if L < 1.0:
            return (0, 0)
        h_edges, w_edges = edges_2d.shape[:2]
        n_steps = min(max(1, int(L) + 1), 128)
        t = np.linspace(0, 1, n_steps, dtype=np.float64)
        px = ax + t * (bx - ax)
        py = ay + t * (by - ay)
        perp_x = -(by - ay) / L
        perp_y = (bx - ax) / L
        k = np.arange(-half_width, half_width + 1, dtype=np.float64)
        qx_flat = np.round(px[np.newaxis, :] + k[:, np.newaxis] * perp_x).astype(np.int32).ravel()
        qy_flat = np.round(py[np.newaxis, :] + k[:, np.newaxis] * perp_y).astype(np.int32).ravel()
        valid = (qx_flat >= 0) & (qx_flat < w_edges) & (qy_flat >= 0) & (qy_flat < h_edges)
        qx_flat = qx_flat[valid]
        qy_flat = qy_flat[valid]
        if qx_flat.size == 0:
            return (0, 0)
        linear = np.unique(np.asarray(qy_flat, dtype=np.int64) * w_edges + np.asarray(qx_flat, dtype=np.int64))
        total = int(linear.size)
        white = int(np.count_nonzero(np.take(edges_2d.ravel(), linear)))
        return (white, total)

    def _xf_point(xo: float, yo: float, w: int, h: int, swap: bool, flip_x: bool, flip_y: bool):
        if swap:
            xt, yt = (yo, xo)
            wt, ht = (h, w)
        else:
            xt, yt = (xo, yo)
            wt, ht = (w, h)
        if flip_x:
            xt = wt - 1 - xt
        if flip_y:
            yt = ht - 1 - yt
        return (xt, yt, wt, ht)

    def _inv_xf_point(xt: float, yt: float, w: int, h: int, swap: bool, flip_x: bool, flip_y: bool):
        wt, ht = (h, w) if swap else (w, h)
        xo, yo = (xt, yt)
        if flip_y:
            yo = ht - 1 - yo
        if flip_x:
            xo = wt - 1 - xo
        if swap:
            xo, yo = (yo, xo)
        xo = min(max(float(xo), 0.0), float(w - 1))
        yo = min(max(float(yo), 0.0), float(h - 1))
        return (xo, yo)

    def _constraints_ok(xt: float, yt: float, wt: int, ht: int) -> bool:
        if xt * 883.0 / 522.0 < wt - 1:
            return False
        if xt > wt - 1:
            return False
        y_offset = (ht - 102) * 5.0 / 135.0
        y_minus_offset = yt - y_offset
        if not 0.1 * (ht - 1) < y_minus_offset < 0.8 * (ht - 1):
            return False
        if (ht - yt - 102) * 108.0 / 135.0 < 100:
            return False
        return True

    def _points_0_1_9_13_in_transformed(xt: float, yt: float, wt: int, ht: int):
        wt_f, ht_f = (float(wt - 1), float(ht))
        yy1, yy2 = (yt, yt)
        xx = xt
        dx = xx / 520.5
        dy = (ht_f - yy1 - 101.0) / 137.5
        dz = (yy1 - yy2) / max(wt_f, 1.0)
        p0 = (0.0, yy1 + 2.5 * dy)
        p1 = (0.0, ht_f - 101.0)
        p9 = (160.0 * dx, ht_f - 101.0 - 160.0 * dx * dz)
        p13 = (xx + 1.5 * dx, yy1 + 2.5 * dy - (xx + 1.5 * dx) * dz)
        return (p0, p1, p9, p13)

    def _points_13_17_24_25_in_transformed(xt: float, yt: float, wt: int, ht: int):
        wt_f, ht_f = (float(wt - 1), float(ht))
        yy1, yy2 = (yt, yt)
        dyy2 = (yy1 - yy2) * 5.0 / 518.0
        p25_y = ht_f - 101.0 - (yy1 - yy2 + dyy2) * 157.0 / 523.0
        dy = (p25_y - yy2) * 3.0 / 138.0
        p13 = (0.0, yy1 + dy)
        p17 = (wt_f * 361.0 / 518.0, ht_f - 101.0)
        p24 = (wt_f, yy2 + dy)
        p25 = (wt_f, p25_y)
        return (p13, p17, p24, p25)

    def _points_15_16_19_20_fallback(xt: float, yt: float, wt: int, ht: int):
        wt_f, ht_f = (float(wt - 1), float(ht))
        return ((wt_f * 0.5, ht_f * 0.5), (wt_f * 0.5, ht_f * 0.5), (wt_f * 0.5, ht_f * 0.5), (wt_f * 0.5, ht_f * 0.5))

    def _points_22_23_25_26_27_in_transformed(xt: float, yt: float, wt: int, ht: int, yy2: float | None=None):
        wt_f, ht_f = (float(wt - 1), float(ht))
        yy1, yy2 = (yt, yt)
        yy3 = yt
        dy = (yy3 - yy2) / 53.5
        p25 = (0.0, yy1 + 3.5 * dy)
        p26 = (wt_f * 110.0 / 290.0, yy2 + (yy1 - yy2) * 180.0 / 290.0 + 3.5 * dy)
        p27 = (wt_f, yy2 + 3.5 * dy)
        p22 = (wt_f * 110.0 / 290.0, yy2 + (yy1 - yy2) * 180.0 / 290.0 + 50.5 * dy)
        p23 = (wt_f, yy2 + 50.5 * dy)
        return (p22, p23, p25, p26, p27)

    def _fallback_points_default(frame_width: int, frame_height: int) -> dict[int, list[float]]:
        w = float(frame_width)
        h = float(frame_height)
        cy = h / 2.0
        x_left = (w - 1) * 3 / 184
        x_right = (w - 1) * 181 / 184
        return {22: [x_left, h - 102.0], 23: [x_right, h - 102.0], 26: [x_left, cy - 1.0], 27: [x_right, cy + 1.0]}

    def _get_add4_group_defs(frame_height: int):
        groups = [{'name': 'pair_0_1_9_13', 'enabled': ADD4_PAIR_0_1_9_13, 'indices': (0, 1, 9, 13), 'build_points': _points_0_1_9_13_in_transformed, 'fallback': None}, {'name': 'pair_13_17_24_25', 'enabled': ADD4_PAIR_13_17_24_25, 'indices': (13, 17, 24, 25), 'build_points': _points_13_17_24_25_in_transformed, 'fallback': None}, {'name': 'pair_15_16_19_20', 'enabled': ADD4_PAIR_15_16_19_20, 'indices': (15, 16, 19, 20), 'build_points': _points_15_16_19_20_fallback, 'fallback': None}, {'name': 'pair_22_23_25_26_27', 'enabled': ADD4_PAIR_22_23_25_26_27, 'indices': (22, 23, 25, 26, 27), 'build_points': _points_22_23_25_26_27_in_transformed, 'fallback': None}]
        enabled = [group for group in groups if group['enabled']]
        return enabled if enabled else [groups[0]]

    def _calculate_score_for_transform(x_orig, y_orig, transform_idx, transforms, template_image, frame_img, frame_width, frame_height, template_len, frame_id, group, sum_rect_xf=None, best_segment_y_fn=None, yy2_candidates_fn=None, frame_number_val=None, selection_meta: dict[str, float] | None=None):
        swap_axes, flip_x, flip_y = transforms[transform_idx]
        xt, yt, wt, ht = _xf_point(x_orig, y_orig, frame_width, frame_height, swap=swap_axes, flip_x=flip_x, flip_y=flip_y)
        if group.get('name') not in ('pair_0_1_9_13', 'pair_13_17_24_25', 'pair_15_16_19_20', 'pair_22_23_25_26_27'):
            if not _constraints_ok(xt, yt, wt, ht):
                return (0.0, None, 'rule-invalid', None)
        best_local_score = -1.0
        best_local_keypoints = None
        best_local_status = 'rule-invalid'
        best_local_error = None

        def _evaluate_points(points_t_local):
            if not points_t_local or len(points_t_local) != len(group['indices']):
                return (0.0, None, 'score-invalid', 'build_points returned wrong count')
            if group.get('rule_check') is not None and group.get('name') not in ('pair_0_1_9_13', 'pair_13_17_24_25', 'pair_15_16_19_20', 'pair_22_23_25_26_27'):
                try:
                    if not group['rule_check'](points_t_local, wt, ht):
                        return (0.0, None, 'rule-invalid', None)
                except Exception as e:
                    return (0.0, None, 'rule-invalid', f'rule_check failed: {e}')
            inv_xf = _inv_xf_point
            points = [inv_xf(float(pt[0]), float(pt[1]), frame_width, frame_height, swap_axes, flip_x, flip_y) for pt in points_t_local]
            test_keypoints = [(0.0, 0.0)] * template_len
            for idx, pt in zip(group['indices'], points, strict=True):
                test_keypoints[idx] = (float(pt[0]), float(pt[1]))
            frame_keypoints_tuples = test_keypoints
            score = 0.0
            error_message = None
            status = 'rule-valid'
            try:
                polygon_masks = None
                mask_label = None
                mask_debug_dir = None
                if not ADD4_USE_POLYGON_MASKS:
                    polygon_masks = _get_polygon_masks(str(group.get('name')))
                    if polygon_masks is None:
                        raise ValueError(f"Polygon masks missing for group '{group.get('name')}'")
                mask_label = f'{group.get('name')}_xf_{transform_idx}'
                score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS, frame_keypoints=frame_keypoints_tuples, frame=frame_img, floor_markings_template=template_image, frame_number=frame_number_val, log_frame_number=False, cached_edges=edges, mask_polygons=polygon_masks, mask_debug_label=mask_label, mask_debug_dir=mask_debug_dir, score_only=False, log_context={'transform_idx': transform_idx, 'pair_indices': list(group.get('indices', []))}, processing_scale=PROCESSING_SCALE)
                if score > 0.0:
                    status = 'score-valid'
                else:
                    status = 'score-invalid'
                    error_message = 'Score is 0.0 - check debug output above for specific reason (homography failed, mask validation failed, bounding box too small, lines outside expected, etc.)'
            except ValueError as e:
                status = 'score-invalid'
                error_message = f'ValueError: {str(e)}'
                score = 0.0
            except InvalidMask as e:
                status = 'score-invalid'
                error_message = f'InvalidMask: {str(e)}'
                score = 0.0
            except Exception as e:
                status = 'score-invalid'
                error_message = f'Exception ({type(e).__name__}): {str(e)}'
                score = 0.0
            return (score, test_keypoints, status, error_message)
        if group.get('uses_best_segment_y') and yy2_candidates_fn is not None:
            candidates = yy2_candidates_fn(yt, wt, ht, swap_axes, flip_x, flip_y)

            def _eval_candidate(yy2_val: float):
                try:
                    points_t = group['build_points'](xt, yt, wt, ht, yy2_val)
                except Exception as e:
                    return (0.0, None, 'score-invalid', f'build_points failed: {e}')
                return _evaluate_points(points_t)
            if TV_AF_EVAL_PARALLEL and (len(candidates) > 1):
                max_workers = max(1, int(TV_AF_EVAL_MAX_WORKERS))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(_eval_candidate, yy2_val): yy2_val for yy2_val, _ in candidates}
                    for fut in as_completed(futures):
                        score, kps, status, error = fut.result()
                        if status == 'rule-invalid':
                            continue
                        if score > best_local_score:
                            best_local_score = score
                            best_local_keypoints = kps
                            best_local_status = status
                            best_local_error = error
            else:
                for yy2_val, _count in candidates:
                    score, kps, status, error = _eval_candidate(yy2_val)
                    if status == 'rule-invalid':
                        continue
                    if score > best_local_score:
                        best_local_score = score
                        best_local_keypoints = kps
                        best_local_status = status
                        best_local_error = error
                        if score >= 1.0:
                            break
        else:
            try:
                points_t = group['build_points'](xt, yt, wt, ht)
            except Exception as e:
                return (0.0, None, 'score-invalid', f'build_points failed: {e}')
            score, kps, status, error = _evaluate_points(points_t)
            best_local_score = score
            best_local_keypoints = kps
            best_local_status = status
            best_local_error = error
        if best_local_score < 0.0:
            return (0.0, None, 'rule-invalid', None)
        return (best_local_score, best_local_keypoints, best_local_status, best_local_error)
    template_image = challenge_template()
    transforms = [(False, False, False), (False, True, False), (False, False, True), (False, True, True), (True, False, False), (True, True, False), (True, False, True), (True, True, True)]
    COARSE_STEP = 4
    REFINE_STEP = 2
    REFINE_RADIUS = 4
    MIN_SLOPE_PX = 2

    def _search_horizontal_in_area(y_lo: int, y_hi: int, max_slope: float) -> dict[str, float] | None:
        if y_hi <= y_lo + MIN_SLOPE_PX:
            return None
        if _search_horizontal_in_area_integral_cy is not None and _integral_flat_cy is not None:
            by0, by1, bwhite, btotal = _search_horizontal_in_area_integral_cy(_integral_flat_cy, frame_width, frame_height, y_lo, y_hi, max_slope, COARSE_STEP, REFINE_RADIUS, REFINE_STEP, MIN_SLOPE_PX, SLOPING_LINE_HALF_WIDTH, SLOPING_LINE_SAMPLE_MAX)
        elif _search_horizontal_in_area_cy is not None and _edges_flat_cy is not None:
            by0, by1, bwhite, btotal = _search_horizontal_in_area_cy(_edges_flat_cy, frame_width, frame_height, y_lo, y_hi, max_slope, COARSE_STEP, REFINE_RADIUS, REFINE_STEP, MIN_SLOPE_PX, SLOPING_LINE_HALF_WIDTH, SLOPING_LINE_SAMPLE_MAX)
        else:
            by0, by1, bwhite, btotal = (-1, -1, -1, -1)
        if by0 >= 0 and btotal > 0:
            mid_y = (by0 + by1) / 2.0
            return {'pos': mid_y, 'start': float(by0), 'count': float(bwhite), 'rate': bwhite / btotal, 'y0': float(by0), 'y1': float(by1)}
        used_cython = _search_horizontal_in_area_integral_cy is not None and _integral_flat_cy is not None or (_search_horizontal_in_area_cy is not None and _edges_flat_cy is not None)
        if used_cython:
            return None
        best_count = -1
        best_y0, best_y1 = (y_lo, y_lo + MIN_SLOPE_PX)
        best_white, best_total = (0, 0)
        max_slope_int = max(MIN_SLOPE_PX, int(max_slope))
        y0_coarse = list(range(y_lo, y_hi + 1, COARSE_STEP))
        if y0_coarse and y0_coarse[-1] != y_hi:
            y0_coarse.append(y_hi)
        for y0 in y0_coarse:
            for y1 in range(y_lo, y_hi + 1, COARSE_STEP):
                if abs(y1 - y0) < MIN_SLOPE_PX or abs(y1 - y0) > max_slope_int:
                    continue
                white, total = _sloping_line_white_count(0.0, float(y0), float(frame_width - 1), float(y1))
                if total > 0 and white > best_count:
                    best_count = white
                    best_y0, best_y1 = (y0, y1)
                    best_white, best_total = (white, total)
        if best_count < 0:
            return None
        y0_ref_min = max(y_lo, best_y0 - REFINE_RADIUS)
        y0_ref_max = min(y_hi, best_y0 + REFINE_RADIUS)
        y1_ref_min = max(y_lo, best_y1 - REFINE_RADIUS)
        y1_ref_max = min(y_hi, best_y1 + REFINE_RADIUS)
        for y0 in range(y0_ref_min, y0_ref_max + 1, REFINE_STEP):
            for y1 in range(y1_ref_min, y1_ref_max + 1, REFINE_STEP):
                if abs(y1 - y0) < MIN_SLOPE_PX or abs(y1 - y0) > max_slope_int:
                    continue
                white, total = _sloping_line_white_count(0.0, float(y0), float(frame_width - 1), float(y1))
                if total > 0 and white > best_count:
                    best_count = white
                    best_y0, best_y1 = (y0, y1)
                    best_white, best_total = (white, total)
        mid_y = (best_y0 + best_y1) / 2.0
        return {'pos': mid_y, 'start': float(best_y0), 'count': float(best_white), 'rate': best_white / best_total if best_total > 0 else 0.0, 'y0': float(best_y0), 'y1': float(best_y1)}

    def _search_vertical_in_area(x_lo: int, x_hi: int, max_slope: float) -> dict[str, float] | None:
        if x_hi <= x_lo + MIN_SLOPE_PX:
            return None
        if _search_vertical_in_area_integral_cy is not None and _integral_flat_cy is not None:
            bx0, bx1, bwhite, btotal = _search_vertical_in_area_integral_cy(_integral_flat_cy, frame_width, frame_height, x_lo, x_hi, max_slope, COARSE_STEP, REFINE_RADIUS, REFINE_STEP, MIN_SLOPE_PX, SLOPING_LINE_HALF_WIDTH, SLOPING_LINE_SAMPLE_MAX)
        elif _search_vertical_in_area_cy is not None and _edges_flat_cy is not None:
            bx0, bx1, bwhite, btotal = _search_vertical_in_area_cy(_edges_flat_cy, frame_width, frame_height, x_lo, x_hi, max_slope, COARSE_STEP, REFINE_RADIUS, REFINE_STEP, MIN_SLOPE_PX, SLOPING_LINE_HALF_WIDTH, SLOPING_LINE_SAMPLE_MAX)
        else:
            bx0, bx1, bwhite, btotal = (-1, -1, -1, -1)
        if bx0 >= 0 and btotal > 0:
            mid_x = (bx0 + bx1) / 2.0
            return {'pos': mid_x, 'start': float(bx0), 'count': float(bwhite), 'rate': bwhite / btotal, 'x0': float(bx0), 'x1': float(bx1)}
        used_cython = _search_vertical_in_area_integral_cy is not None and _integral_flat_cy is not None or (_search_vertical_in_area_cy is not None and _edges_flat_cy is not None)
        if used_cython:
            return None
        best_count = -1
        best_x0, best_x1 = (x_lo, x_lo + MIN_SLOPE_PX)
        best_white, best_total = (0, 0)
        max_slope_int = max(MIN_SLOPE_PX, int(max_slope))
        x0_coarse = list(range(x_lo, x_hi + 1, COARSE_STEP))
        if x0_coarse and x0_coarse[-1] != x_hi:
            x0_coarse.append(x_hi)
        for x0 in x0_coarse:
            for x1 in range(x_lo, x_hi + 1, COARSE_STEP):
                if abs(x1 - x0) < MIN_SLOPE_PX or abs(x1 - x0) > max_slope_int:
                    continue
                white, total = _sloping_line_white_count(float(x0), 0.0, float(x1), float(frame_height - 1))
                if total > 0 and white > best_count:
                    best_count = white
                    best_x0, best_x1 = (x0, x1)
                    best_white, best_total = (white, total)
        if best_count < 0:
            return None
        x0_ref_min = max(x_lo, best_x0 - REFINE_RADIUS)
        x0_ref_max = min(x_hi, best_x0 + REFINE_RADIUS)
        x1_ref_min = max(x_lo, best_x1 - REFINE_RADIUS)
        x1_ref_max = min(x_hi, best_x1 + REFINE_RADIUS)
        for x0 in range(x0_ref_min, x0_ref_max + 1, REFINE_STEP):
            for x1 in range(x1_ref_min, x1_ref_max + 1, REFINE_STEP):
                if abs(x1 - x0) < MIN_SLOPE_PX or abs(x1 - x0) > max_slope_int:
                    continue
                white, total = _sloping_line_white_count(float(x0), 0.0, float(x1), float(frame_height - 1))
                if total > 0 and white > best_count:
                    best_count = white
                    best_x0, best_x1 = (x0, x1)
                    best_white, best_total = (white, total)
        mid_x = (best_x0 + best_x1) / 2.0
        return {'pos': mid_x, 'start': float(best_x0), 'count': float(best_white), 'rate': best_white / best_total if best_total > 0 else 0.0, 'x0': float(best_x0), 'x1': float(best_x1)}
    H_REF = float(frame_height - 1) if frame_height > 0 else 0.0
    W_REF = float(frame_width - 1) if frame_width > 0 else 0.0
    max_slope_h = frame_height / 10.0 if frame_height > 0 else 0.0
    max_slope_w = frame_width / 10.0 if frame_width > 0 else 0.0
    h_10_80: dict[str, float] | None = None
    h_20_90: dict[str, float] | None = None
    v_20_90: dict[str, float] | None = None
    v_10_80: dict[str, float] | None = None
    global _adding_four_points_cy_candidate_logged
    if not _adding_four_points_cy_candidate_logged:
        _adding_four_points_cy_candidate_logged = True
    if frame_height > 0 and frame_width > 0:
        y_lo_10_80 = int(frame_height * 0.1)
        y_hi_10_80 = int(frame_height * 0.8)
        y_lo_20_90 = int(frame_height * 0.2)
        y_hi_20_90 = int(frame_height * 0.9)
        x_lo_10_80 = int(frame_width * 0.1)
        x_hi_10_80 = int(frame_width * 0.8)
        x_lo_20_90 = int(frame_width * 0.2)
        x_hi_20_90 = int(frame_width * 0.9)
        if TV_AF_EVAL_PARALLEL:
            with ThreadPoolExecutor(max_workers=TV_AF_EVAL_MAX_WORKERS) as ex:
                h_10_80_fut = ex.submit(_search_horizontal_in_area, y_lo_10_80, y_hi_10_80, max_slope_h)
                h_20_90_fut = ex.submit(_search_horizontal_in_area, y_lo_20_90, y_hi_20_90, max_slope_h)
                v_20_90_fut = ex.submit(_search_vertical_in_area, x_lo_20_90, x_hi_20_90, max_slope_w)
                v_10_80_fut = ex.submit(_search_vertical_in_area, x_lo_10_80, x_hi_10_80, max_slope_w)
                h_10_80 = h_10_80_fut.result()
                h_20_90 = h_20_90_fut.result()
                v_20_90 = v_20_90_fut.result()
                v_10_80 = v_10_80_fut.result()
        else:
            h_10_80 = _search_horizontal_in_area(y_lo_10_80, y_hi_10_80, max_slope_h)
            h_20_90 = _search_horizontal_in_area(y_lo_20_90, y_hi_20_90, max_slope_h)
            v_20_90 = _search_vertical_in_area(x_lo_20_90, x_hi_20_90, max_slope_w)
            v_10_80 = _search_vertical_in_area(x_lo_10_80, x_hi_10_80, max_slope_w)
    horizontal_candidates: list[dict[str, float]] = [c for c in [h_10_80, h_20_90] if c is not None]
    vertical_candidates: list[dict[str, float]] = [c for c in [v_20_90, v_10_80] if c is not None]
    best_line_for_transform: list[dict[str, float] | None] = [None] * 8
    if frame_width > 0 and frame_height > 0:
        if h_10_80 is not None:
            y0, y1 = (float(h_10_80['y0']), float(h_10_80['y1']))
            best_line_for_transform[0] = {'yy1': y0, 'yy2': y1, 'type': 'h', '_y0': y0, '_y1': y1}
            best_line_for_transform[1] = {'yy1': y1, 'yy2': y0, 'type': 'h', '_y0': y0, '_y1': y1}
        if h_20_90 is not None:
            y0, y1 = (float(h_20_90['y0']), float(h_20_90['y1']))
            best_line_for_transform[2] = {'yy1': H_REF - y0, 'yy2': H_REF - y1, 'type': 'h', '_y0': y0, '_y1': y1}
            best_line_for_transform[3] = {'yy1': H_REF - y1, 'yy2': H_REF - y0, 'type': 'h', '_y0': y0, '_y1': y1}
        if v_20_90 is not None:
            x0, x1 = (float(v_20_90['x0']), float(v_20_90['x1']))
            best_line_for_transform[4] = {'xx1': x0, 'xx2': x1, 'type': 'v', '_x0': x0, '_x1': x1}
            best_line_for_transform[5] = {'xx1': x1, 'xx2': x0, 'type': 'v', '_x0': x0, '_x1': x1}
        if v_10_80 is not None:
            x0, x1 = (float(v_10_80['x0']), float(v_10_80['x1']))
            best_line_for_transform[6] = {'xx1': W_REF - x0, 'xx2': W_REF - x1, 'type': 'v', '_x0': x0, '_x1': x1}
            best_line_for_transform[7] = {'xx1': W_REF - x1, 'xx2': W_REF - x0, 'type': 'v', '_x0': x0, '_x1': x1}
    pair01913_poly_scores: dict[int, float] = {}
    pair13172425_poly_scores: dict[int, float] = {}
    PAIR01_VERTICAL_BAND_HW = 2

    def _find_best_xx_vertical(yy1: float, yy2: float, wt: float, ht: float, edges_t: np.ndarray) -> float | None:
        x_left = wt * 520.5 / 880.0
        if wt - x_left < 3:
            return None
        xx_min = max(int(np.ceil(x_left)), 2)
        xx_max = min(int(wt) - 3, int(wt) - 1)
        if xx_max < xx_min:
            return None
        best_count = -1
        best_xx = x_left
        step = 4
        for xx_c in range(xx_min, xx_max + 1, step):
            y_at_line = yy1 + (yy2 - yy1) * xx_c / max(wt, 1.0)
            y_start = int(max(0, np.ceil(y_at_line)))
            y_end = int(ht)
            if y_end <= y_start + 2:
                continue
            x0 = max(0, xx_c - PAIR01_VERTICAL_BAND_HW)
            x1 = min(int(wt), xx_c + PAIR01_VERTICAL_BAND_HW + 1)
            count = int(edges_t[y_start:y_end, x0:x1].sum())
            if count > best_count:
                best_count = count
                best_xx = float(xx_c)
        xx_ref_min = max(xx_min, int(best_xx) - 4)
        xx_ref_max = min(xx_max, int(best_xx) + 4)
        for xx_c in range(xx_ref_min, xx_ref_max + 1, 2):
            y_at_line = yy1 + (yy2 - yy1) * xx_c / max(wt, 1.0)
            y_start = int(max(0, np.ceil(y_at_line)))
            y_end = int(ht)
            if y_end <= y_start + 2:
                continue
            x0 = max(0, xx_c - PAIR01_VERTICAL_BAND_HW)
            x1 = min(int(wt), xx_c + PAIR01_VERTICAL_BAND_HW + 1)
            count = int(edges_t[y_start:y_end, x0:x1].sum())
            if count > best_count:
                best_count = count
                best_xx = float(xx_c)
        return best_xx if best_count >= 0 else None

    def _pair01913_kps_for_transform(tf_idx: int) -> list[tuple[float, float]] | None:
        bl = best_line_for_transform[tf_idx] if tf_idx < len(best_line_for_transform) else None
        if bl is None:
            return None
        swap, flip_x, flip_y = transforms[tf_idx]
        W = float(frame_width - 1)
        H = float(frame_height - 1)
        wt = H if swap else W
        ht = W if swap else H
        if bl.get('type') == 'h':
            yy1, yy2 = (float(bl.get('yy1', 0)), float(bl.get('yy2', 0)))
        else:
            yy1, yy2 = (float(bl.get('xx1', 0)), float(bl.get('xx2', 0)))
        edges_t = _edges_xf(swap, flip_x, flip_y)
        xx = _find_best_xx_vertical(yy1, yy2, wt, ht, edges_t)
        if xx is None:
            xx = wt * 520.5 / 880.0
        dx = xx / 520.5
        dy = (ht - yy1 - 101.0) / 137.5
        dz = (yy1 - yy2) / max(wt, 1.0)
        p0 = (0.0, yy1 + 2.5 * dy)
        p1 = (0.0, ht - 101.0)
        p9 = (160.0 * dx, ht - 101.0 - 160.0 * dx * dz)
        p13 = (xx + 1.5 * dx, yy1 + 2.5 * dy - (xx + 1.5 * dx) * dz)
        pts_t = [p0, p1, p9, p13]
        return [_inv_xf_point(xt, yt, frame_width, frame_height, swap, flip_x, flip_y) for xt, yt in pts_t]

    def _pair13172425_kps_for_transform(tf_idx: int) -> list[tuple[float, float]] | None:
        bl = best_line_for_transform[tf_idx] if tf_idx < len(best_line_for_transform) else None
        if bl is None:
            return None
        swap, flip_x, flip_y = transforms[tf_idx]
        W = float(frame_width - 1)
        H = float(frame_height - 1)
        wt = H if swap else W
        ht = W if swap else H
        if bl.get('type') == 'h':
            yy1, yy2 = (float(bl.get('yy1', 0)), float(bl.get('yy2', 0)))
        else:
            yy1, yy2 = (float(bl.get('xx1', 0)), float(bl.get('xx2', 0)))
        dy = (ht - yy2 - 101.0) / 137.5
        p13 = (0.0, yy1 + 2.5 * dy)
        p17_y = (yy1 - yy2) * (1045.0 - 888.0) / (1045.0 - 527.0) + yy2 + 137.5 * dy
        p17 = (wt * (888.0 - 527.0) / (1045.0 - 527.0), p17_y)
        p24 = (wt, yy2 + 2.5 * dy)
        p25 = (wt, yy2 + 137.5 * dy)
        pts_t = [p13, p17, p24, p25]
        return [_inv_xf_point(xt, yt, frame_width, frame_height, swap, flip_x, flip_y) for xt, yt in pts_t]
    pair15161920_poly_scores: dict[int, float] = {}
    PAIR1516_VERTICAL_BAND_HW = 2
    PAIR1516_XX_RIGHT_RATIO = (995.0 - 886.5) / (995.0 - 527.0)
    PAIR1516_YY3_RATIO = 542.0 / 678.5
    PAIR1516_DY_DENOM = 678.5 - 410.0
    PAIR1516_DX_DENOM = 886.5 - 527.0

    def _find_best_xx_vertical_15161920(yy1: float, yy2: float, yy3: float, yy4: float, wt: float, ht: float, edges_t: np.ndarray) -> float | None:
        x_right = wt * PAIR1516_XX_RIGHT_RATIO
        if x_right < 3:
            return None
        xx_max = min(int(x_right), int(wt) - 1)
        xx_min = 2
        if xx_max < xx_min:
            return None
        best_count = -1
        best_xx = wt * 0.2
        step = 4
        for xx_c in range(xx_min, xx_max + 1, step):
            y_at_line = yy3 + (yy4 - yy3) * xx_c / max(wt, 1.0)
            y_start = int(max(0, np.ceil(y_at_line)))
            y_end = int(ht)
            if y_end <= y_start + 2:
                continue
            x0 = max(0, xx_c - PAIR1516_VERTICAL_BAND_HW)
            x1 = min(int(wt), xx_c + PAIR1516_VERTICAL_BAND_HW + 1)
            count = int(edges_t[y_start:y_end, x0:x1].sum())
            if count > best_count:
                best_count = count
                best_xx = float(xx_c)
        xx_ref_min = max(xx_min, int(best_xx) - 4)
        xx_ref_max = min(xx_max, int(best_xx) + 4)
        for xx_c in range(xx_ref_min, xx_ref_max + 1, 2):
            y_at_line = yy3 + (yy4 - yy3) * xx_c / max(wt, 1.0)
            y_start = int(max(0, np.ceil(y_at_line)))
            y_end = int(ht)
            if y_end <= y_start + 2:
                continue
            x0 = max(0, xx_c - PAIR1516_VERTICAL_BAND_HW)
            x1 = min(int(wt), xx_c + PAIR1516_VERTICAL_BAND_HW + 1)
            count = int(edges_t[y_start:y_end, x0:x1].sum())
            if count > best_count:
                best_count = count
                best_xx = float(xx_c)
        return best_xx if best_count >= 0 else None

    def _pair15161920_kps_for_transform(tf_idx: int) -> list[tuple[float, float]] | None:
        bl = best_line_for_transform[tf_idx] if tf_idx < len(best_line_for_transform) else None
        if bl is None:
            return None
        swap, flip_x, flip_y = transforms[tf_idx]
        W = float(frame_width - 1)
        H = float(frame_height - 1)
        wt = H if swap else W
        ht = W if swap else H
        if bl.get('type') == 'h':
            yy1, yy2 = (float(bl.get('yy1', 0)), float(bl.get('yy2', 0)))
        else:
            yy1, yy2 = (float(bl.get('xx1', 0)), float(bl.get('xx2', 0)))
        yy3 = yy1 + (ht - yy1) * PAIR1516_YY3_RATIO
        yy4 = yy2 + (ht - yy2) * PAIR1516_YY3_RATIO
        edges_t = _edges_xf(swap, flip_x, flip_y)
        xx = _find_best_xx_vertical_15161920(yy1, yy2, yy3, yy4, wt, ht, edges_t)
        if xx is None:
            xx = wt * PAIR1516_XX_RIGHT_RATIO * 0.5
        h = ht - (yy1 - (yy1 - yy2) * xx / max(wt, 1.0))
        dy = h / PAIR1516_DY_DENOM
        dx = (wt - xx) / PAIR1516_DX_DENOM
        p16 = (wt, yy2 + 3.5 * dy)
        p15 = (wt, yy2 + 245.5 * dy)
        p20 = (xx - 1.5 * dx, ht - 130.0 * dy)
        p19 = (xx - 1.5 * dx, ht)
        pts_t = [p15, p16, p19, p20]
        return [_inv_xf_point(xt, yt, frame_width, frame_height, swap, flip_x, flip_y) for xt, yt in pts_t]
    pair2223252627_poly_scores: dict[int, float] = {}
    PAIR22_LINE_BAND_HW = 4

    def _sloping_band_count_integral(ax: float, ay: float, bx: float, by: float, half_width: int, ii: np.ndarray, ht: int, wt: int) -> tuple[int, int]:
        L = float(np.hypot(bx - ax, by - ay))
        if L < 1.0:
            return (0, 0)
        y_min = max(0, int(min(ay, by)))
        y_max = min(ht - 1, int(max(ay, by)))
        if y_max < y_min:
            return (0, 0)
        half_x = half_width * abs(by - ay) / L
        white = 0
        total = 0
        dy = by - ay
        for y in range(y_min, y_max + 1):
            t = (y - ay) / dy if abs(dy) > 1e-09 else 0.5
            x_c = ax + (bx - ax) * t
            x_lo = max(0, int(x_c - half_x))
            x_hi = min(wt, int(x_c + half_x) + 1)
            if x_hi <= x_lo:
                continue
            white += int(ii[y + 1, x_hi] - ii[y, x_hi] - ii[y + 1, x_lo] + ii[y, x_lo])
            total += x_hi - x_lo
        return (white, total)

    def _find_best_yy3_parallel(yy1: float, yy2: float, wt: float, ht: float, edges_t: np.ndarray) -> float | None:
        x_left = wt * 110.0 / 290.0
        if wt - x_left < 2:
            return None
        yy3_lower = yy2 + (ht - yy2) * 52.5 / 160.5
        yy3_min = max(yy3_lower + 0.5, yy2 + 2, 0)
        yy3_max = min(ht - 2, ht)
        if yy3_max <= yy3_min:
            return (yy2 + yy3_max) / 2.0
        try:
            ii = cv2.integral(edges_t)
        except Exception:
            ii = None
        wt_i = int(wt)
        ht_i = int(ht)
        best_count = -1
        best_yy3 = yy2
        step = 8
        for yy3 in np.arange(yy3_min, yy3_max + 0.5, step):
            y_at_left = yy3 - (yy2 - yy1) * 180.0 / 290.0
            if ii is not None:
                white, total = _sloping_band_count_integral(x_left, y_at_left, wt, yy3, PAIR22_LINE_BAND_HW, ii, ht_i, wt_i)
            else:
                white, total = _sloping_line_white_count_band(x_left, y_at_left, wt, yy3, PAIR22_LINE_BAND_HW, edges_t)
            if total > 0 and white > best_count:
                best_count = white
                best_yy3 = float(yy3)
        yy3_ref_min = max(yy3_min, best_yy3 - 8)
        yy3_ref_max = min(yy3_max, best_yy3 + 8)
        step_ref = 4
        for yy3 in np.arange(yy3_ref_min, yy3_ref_max + 0.5, step_ref):
            y_at_left = yy3 - (yy2 - yy1) * 180.0 / 290.0
            if ii is not None:
                white, total = _sloping_band_count_integral(x_left, y_at_left, wt, yy3, PAIR22_LINE_BAND_HW, ii, ht_i, wt_i)
            else:
                white, total = _sloping_line_white_count_band(x_left, y_at_left, wt, yy3, PAIR22_LINE_BAND_HW, edges_t)
            if total > 0 and white > best_count:
                best_count = white
                best_yy3 = float(yy3)
        return best_yy3 if best_count >= 0 else None

    def _find_farthest_yy3_parallel(yy1: float, yy2: float, wt: float, ht: float, edges_t: np.ndarray) -> float | None:
        """Same valid range as _find_best_yy3_parallel; return yy3 farthest from best line (max |yy3 - yy2|)."""
        x_left = wt * 110.0 / 290.0
        if wt - x_left < 2:
            return None
        yy3_lower = yy2 + (ht - yy2) * 52.5 / 160.5
        yy3_min = max(yy3_lower + 0.5, yy2 + 2, 0)
        yy3_max = min(ht - 2, ht)
        if yy3_max <= yy3_min:
            return (yy2 + yy3_max) / 2.0
        if (yy3_max - yy2) >= (yy2 - yy3_min):
            return yy3_max
        return yy3_min

    def _pair2223252627_kps_for_transform(tf_idx: int, use_farthest_line: bool = False) -> list[tuple[float, float]] | None:
        bl = best_line_for_transform[tf_idx] if tf_idx < len(best_line_for_transform) else None
        if bl is None:
            return None
        swap, flip_x, flip_y = transforms[tf_idx]
        W = float(frame_width - 1)
        H = float(frame_height - 1)
        wt = H if swap else W
        ht = W if swap else H
        if bl.get('type') == 'h':
            yy1, yy2 = (float(bl.get('yy1', 0)), float(bl.get('yy2', 0)))
        else:
            yy1, yy2 = (float(bl.get('xx1', 0)), float(bl.get('xx2', 0)))
        edges_t = _edges_xf(swap, flip_x, flip_y)
        yy3 = _find_farthest_yy3_parallel(yy1, yy2, wt, ht, edges_t) if use_farthest_line else _find_best_yy3_parallel(yy1, yy2, wt, ht, edges_t)
        if yy3 is None:
            yy3 = yy2 + (ht - yy2) * 0.5
        dy = (yy3 - yy2) / 53.5
        p25 = (0.0, yy1 + 3.5 * dy)
        p26 = (wt * 110.0 / 290.0, yy2 + (yy1 - yy2) * 180.0 / 290.0 + 3.5 * dy)
        p27 = (wt, yy2 + 3.5 * dy)
        p22 = (wt * 110.0 / 290.0, yy2 + (yy1 - yy2) * 180.0 / 290.0 + 50.5 * dy)
        p23 = (wt, yy2 + 50.5 * dy)
        pts_t = [p22, p23, p25, p26, p27]
        return [_inv_xf_point(xt, yt, frame_width, frame_height, swap, flip_x, flip_y) for xt, yt in pts_t]
    edges_xf_cache: dict[tuple[bool, bool, bool], np.ndarray] = {}

    def _edges_xf(swap_axes: bool, flip_x: bool, flip_y: bool) -> np.ndarray:
        key = (swap_axes, flip_x, flip_y)
        cached = edges_xf_cache.get(key)
        if cached is not None:
            return cached
        edges_t = binary_edges
        if swap_axes:
            edges_t = edges_t.T
        if flip_y:
            edges_t = edges_t[::-1, :]
        if flip_x:
            edges_t = edges_t[:, ::-1]
        edges_xf_cache[key] = edges_t
        return edges_t
    seg_width = 3
    seg_step_precompute = 2
    seg_step_refine = 1
    x_pos_list = list(range(seg_width // 2, frame_width - seg_width // 2, seg_step_precompute))
    y_pos_list = list(range(seg_width // 2, frame_height - seg_width // 2, seg_step_precompute))
    x_pos_list_fine = list(range(seg_width // 2, frame_width - seg_width // 2, seg_step_refine))
    y_pos_list_fine = list(range(seg_width // 2, frame_height - seg_width // 2, seg_step_refine))

    def _col_band(y0: int, y1: int) -> np.ndarray:
        if y1 <= y0 or frame_width <= 0:
            return np.zeros((frame_width,), dtype=np.int64)
        if y0 <= 0:
            return col_prefix[y1 - 1].astype(np.int64)
        return (col_prefix[y1 - 1] - col_prefix[y0 - 1]).astype(np.int64)

    def _row_band(x0: int, x1: int) -> np.ndarray:
        if x1 <= x0 or frame_height <= 0:
            return np.zeros((frame_height,), dtype=np.int64)
        if x0 <= 0:
            return row_prefix[:, x1 - 1].astype(np.int64)
        return (row_prefix[:, x1 - 1] - row_prefix[:, x0 - 1]).astype(np.int64)

    def _segments_from_col_band(col_band: np.ndarray, total_pixels: int) -> list[dict[str, float]]:
        if total_pixels <= 0 or not x_pos_list:
            return []
        if _seg_candidates_col_cy is not None:
            try:
                col_band_u = np.ascontiguousarray(col_band, dtype=np.int64)
                return _seg_candidates_col_cy(col_band_u, int(seg_width), int(seg_step_precompute), int(total_pixels))
            except Exception:
                pass
        rolling = _rolling_sum_1d(col_band, seg_width)
        if not len(rolling):
            return []
        x_pos_arr = np.asarray(x_pos_list, dtype=np.int64)
        x_start = x_pos_arr - seg_width // 2
        white_counts = rolling[x_start].astype(np.int64)
        rates = white_counts.astype(np.float64) / float(total_pixels) if total_pixels > 0 else np.zeros_like(white_counts, dtype=np.float64)
        order = np.argsort(-rates, kind='mergesort')
        return [{'x': float(x_pos_arr[i]), 'rate': float(rates[i])} for i in order.tolist()]

    def _segments_from_row_band(row_band: np.ndarray, total_pixels: int) -> list[dict[str, float]]:
        if total_pixels <= 0 or not y_pos_list:
            return []
        if _seg_candidates_row_cy is not None:
            try:
                row_band_u = np.ascontiguousarray(row_band, dtype=np.int64)
                return _seg_candidates_row_cy(row_band_u, int(seg_width), int(seg_step_precompute), int(total_pixels))
            except Exception:
                pass
        rolling = _rolling_sum_1d(row_band, seg_width)
        if not len(rolling):
            return []
        y_pos_arr = np.asarray(y_pos_list, dtype=np.int64)
        y_start = y_pos_arr - seg_width // 2
        white_counts = rolling[y_start].astype(np.int64)
        rates = white_counts.astype(np.float64) / float(total_pixels) if total_pixels > 0 else np.zeros_like(white_counts, dtype=np.float64)
        order = np.argsort(-rates, kind='mergesort')
        return [{'y': float(y_pos_arr[i]), 'rate': float(rates[i])} for i in order.tolist()]

    def _segments_from_col_band_fine(col_band: np.ndarray, total_pixels: int) -> list[dict[str, float]]:
        if total_pixels <= 0 or not x_pos_list_fine:
            return []
        rolling = _rolling_sum_1d(col_band, seg_width)
        if not len(rolling):
            return []
        x_pos_arr = np.asarray(x_pos_list_fine, dtype=np.int64)
        x_start = x_pos_arr - seg_width // 2
        white_counts = rolling[x_start].astype(np.int64)
        rates = white_counts.astype(np.float64) / float(total_pixels) if total_pixels > 0 else np.zeros_like(white_counts, dtype=np.float64)
        order = np.argsort(-rates, kind='mergesort')
        return [{'x': float(x_pos_arr[i]), 'rate': float(rates[i])} for i in order.tolist()]

    def _segments_from_row_band_fine(row_band: np.ndarray, total_pixels: int) -> list[dict[str, float]]:
        if total_pixels <= 0 or not y_pos_list_fine:
            return []
        rolling = _rolling_sum_1d(row_band, seg_width)
        if not len(rolling):
            return []
        y_pos_arr = np.asarray(y_pos_list_fine, dtype=np.int64)
        y_start = y_pos_arr - seg_width // 2
        white_counts = rolling[y_start].astype(np.int64)
        rates = white_counts.astype(np.float64) / float(total_pixels) if total_pixels > 0 else np.zeros_like(white_counts, dtype=np.float64)
        order = np.argsort(-rates, kind='mergesort')
        return [{'y': float(y_pos_arr[i]), 'rate': float(rates[i])} for i in order.tolist()]

    def _segments_down_fine(h_y: float) -> list[dict[str, float]]:
        y_start = int(h_y)
        y_end = frame_height
        col_band = _col_band(y_start, y_end)
        total_pixels = seg_width * max(0, y_end - y_start)
        return _segments_from_col_band_fine(col_band, total_pixels)

    def _segments_up_fine(h_y: float) -> list[dict[str, float]]:
        y_start = 0
        y_end = int(h_y)
        col_band = _col_band(y_start, y_end)
        total_pixels = seg_width * max(0, y_end - y_start)
        return _segments_from_col_band_fine(col_band, total_pixels)

    def _segments_right_fine(v_x: float) -> list[dict[str, float]]:
        x_start = int(v_x)
        x_end = frame_width
        row_band = _row_band(x_start, x_end)
        total_pixels = seg_width * max(0, x_end - x_start)
        return _segments_from_row_band_fine(row_band, total_pixels)

    def _segments_left_fine(v_x: float) -> list[dict[str, float]]:
        x_start = 0
        x_end = int(v_x)
        row_band = _row_band(x_start, x_end)
        total_pixels = seg_width * max(0, x_end - x_start)
        return _segments_from_row_band_fine(row_band, total_pixels)
    seg_down_by_line: dict[int, list[dict[str, float]]] = {}
    seg_up_by_line: dict[int, list[dict[str, float]]] = {}
    seg_right_by_line: dict[int, list[dict[str, float]]] = {}
    seg_left_by_line: dict[int, list[dict[str, float]]] = {}
    t_seg_precompute_start = time.perf_counter() if _profile else 0.0

    def _segments_down_raw(h_y: float) -> list[dict[str, float]]:
        y_start = int(h_y)
        y_end = frame_height
        col_band = _col_band(y_start, y_end)
        total_pixels = seg_width * max(0, y_end - y_start)
        return _segments_from_col_band(col_band, total_pixels)

    def _segments_up_raw(h_y: float) -> list[dict[str, float]]:
        y_start = 0
        y_end = int(h_y)
        col_band = _col_band(y_start, y_end)
        total_pixels = seg_width * max(0, y_end - y_start)
        return _segments_from_col_band(col_band, total_pixels)

    def _segments_right_raw(v_x: float) -> list[dict[str, float]]:
        x_start = int(v_x)
        x_end = frame_width
        row_band = _row_band(x_start, x_end)
        total_pixels = seg_width * max(0, x_end - x_start)
        return _segments_from_row_band(row_band, total_pixels)

    def _segments_left_raw(v_x: float) -> list[dict[str, float]]:
        x_start = 0
        x_end = int(v_x)
        row_band = _row_band(x_start, x_end)
        total_pixels = seg_width * max(0, x_end - x_start)
        return _segments_from_row_band(row_band, total_pixels)
    seg_down_count = 0
    seg_up_count = 0
    seg_right_count = 0
    seg_left_count = 0
    used_cy_precompute = False
    if _seg_precompute_cy is not None:
        try:
            h_positions = np.asarray([int(line['pos']) for line in horizontal_candidates], dtype=np.int32)
            v_positions = np.asarray([int(line['pos']) for line in vertical_candidates], dtype=np.int32)
            seg_down_by_line, seg_up_by_line, seg_right_by_line, seg_left_by_line = _seg_precompute_cy(col_prefix, row_prefix, h_positions, v_positions, int(frame_height), int(frame_width), int(seg_width), int(seg_step_precompute))
            seg_down_count = sum((len(v) for v in seg_down_by_line.values()))
            seg_up_count = sum((len(v) for v in seg_up_by_line.values()))
            seg_right_count = sum((len(v) for v in seg_right_by_line.values()))
            seg_left_count = sum((len(v) for v in seg_left_by_line.values()))
            used_cy_precompute = True
        except Exception:
            used_cy_precompute = False
    if not used_cy_precompute and TV_AF_EVAL_PARALLEL:
        max_workers = max(1, int(TV_AF_EVAL_MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            all_futures = {ex.submit(_segments_down_raw, float(line['pos'])): ('down', int(line['pos'])) for line in horizontal_candidates}
            all_futures.update({ex.submit(_segments_up_raw, float(line['pos'])): ('up', int(line['pos'])) for line in horizontal_candidates})
            all_futures.update({ex.submit(_segments_right_raw, float(line['pos'])): ('right', int(line['pos'])) for line in vertical_candidates})
            all_futures.update({ex.submit(_segments_left_raw, float(line['pos'])): ('left', int(line['pos'])) for line in vertical_candidates})
            for fut in as_completed(all_futures):
                kind, key = all_futures[fut]
                segments = fut.result()
                if kind == 'down':
                    seg_down_by_line[key] = segments
                    seg_down_count += len(segments)
                elif kind == 'up':
                    seg_up_by_line[key] = segments
                    seg_up_count += len(segments)
                elif kind == 'right':
                    seg_right_by_line[key] = segments
                    seg_right_count += len(segments)
                else:
                    seg_left_by_line[key] = segments
                    seg_left_count += len(segments)
    elif not used_cy_precompute:
        for line in horizontal_candidates:
            y_key = int(line['pos'])
            if y_key not in seg_down_by_line:
                seg_down_by_line[y_key] = _segments_down_raw(float(line['pos']))
                seg_down_count += len(seg_down_by_line[y_key])
            if y_key not in seg_up_by_line:
                seg_up_by_line[y_key] = _segments_up_raw(float(line['pos']))
                seg_up_count += len(seg_up_by_line[y_key])
        for line in vertical_candidates:
            x_key = int(line['pos'])
            if x_key not in seg_right_by_line:
                seg_right_by_line[x_key] = _segments_right_raw(float(line['pos']))
                seg_right_count += len(seg_right_by_line[x_key])
            if x_key not in seg_left_by_line:
                seg_left_by_line[x_key] = _segments_left_raw(float(line['pos']))
                seg_left_count += len(seg_left_by_line[x_key])
    t_seg_precompute_end = time.perf_counter() if _profile else 0.0
    seg_precompute_ms = float((t_seg_precompute_end - t_seg_precompute_start) * 1000.0)

    def _evaluate_group(group, shared_executor=None):
        try:
            frame_number_val = int(frame_id)
        except Exception:
            frame_number_val = None
        t_group_start = time.perf_counter() if _profile else 0.0
        if ADD4_USE_POLYGON_MASKS and group.get('name') in ('pair_0_1_9_13', 'pair_13_17_24_25', 'pair_15_16_19_20', 'pair_22_23_25_26_27') and any(best_line_for_transform) and (frame_img is not None) and (template_image is not None):
            _kps_fn = {'pair_0_1_9_13': _pair01913_kps_for_transform, 'pair_13_17_24_25': _pair13172425_kps_for_transform, 'pair_15_16_19_20': _pair15161920_kps_for_transform, 'pair_22_23_25_26_27': _pair2223252627_kps_for_transform}.get(group.get('name'))
            _tpl_indices = {'pair_0_1_9_13': [0, 1, 9, 13], 'pair_13_17_24_25': [13, 17, 24, 25], 'pair_15_16_19_20': [15, 16, 19, 20], 'pair_22_23_25_26_27': [22, 23, 25, 26, 27]}.get(group.get('name'))
            if _kps_fn is not None and _tpl_indices is not None:
                t_tpl_start = time.perf_counter() if _profile else 0.0
                best_score = 0.0
                best_kps = None
                best_tf_idx = -1
                scores_list_tpl: list[tuple[float, float, float, int, list[tuple[float, float]]] | None] = [None] * 8
                mask_debug_dir = None

                def _eval_one_tf(tf_idx: int):
                    frame_kps = _kps_fn(tf_idx)
                    if frame_kps is None:
                        return (tf_idx, None)
                    test_keypoints = [(0.0, 0.0)] * template_len
                    for idx, pt in zip(_tpl_indices, frame_kps, strict=True):
                        test_keypoints[idx] = (float(pt[0]), float(pt[1]))
                    try:
                        score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS, frame_keypoints=test_keypoints, frame=frame_img, floor_markings_template=template_image, frame_number=frame_number_val, log_frame_number=False, cached_edges=edges, mask_polygons=None, mask_debug_label=f'{group.get('name')}_xf_{tf_idx}', mask_debug_dir=mask_debug_dir, score_only=False, log_context={'transform_idx': tf_idx, 'pair_indices': _tpl_indices}, processing_scale=PROCESSING_SCALE)
                    except Exception:
                        score = 0.0
                    return (tf_idx, (score, 0.0, 0.0, tf_idx, test_keypoints))

                is_pair_2223252627 = group.get('name') == 'pair_22_23_25_26_27'
                if is_pair_2223252627:
                    def _eval_one_tf_2223(tf_idx: int, use_farthest_line: bool):
                        frame_kps = _pair2223252627_kps_for_transform(tf_idx, use_farthest_line=use_farthest_line)
                        if frame_kps is None:
                            return (tf_idx, None)
                        test_keypoints = [(0.0, 0.0)] * template_len
                        for idx, pt in zip(_tpl_indices, frame_kps, strict=True):
                            test_keypoints[idx] = (float(pt[0]), float(pt[1]))
                        try:
                            score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS, frame_keypoints=test_keypoints, frame=frame_img, floor_markings_template=template_image, frame_number=frame_number_val, log_frame_number=False, cached_edges=edges, mask_polygons=None, mask_debug_label=None, mask_debug_dir=None, score_only=False, log_context={'transform_idx': tf_idx, 'pair_indices': _tpl_indices, 'use_farthest_line': use_farthest_line}, processing_scale=PROCESSING_SCALE)
                        except Exception:
                            score = 0.0
                        return (tf_idx, (score, 0.0, 0.0, tf_idx, test_keypoints))
                    if TV_AF_EVAL_PARALLEL and shared_executor is not None:
                        futures = {shared_executor.submit(_eval_one_tf_2223, i, uf): (i, uf) for i in range(8) for uf in (False, True)}
                        for fut in as_completed(futures):
                            tf_idx, res = fut.result()
                            if res is not None:
                                sc = res[0]
                                if scores_list_tpl[tf_idx] is None or sc > scores_list_tpl[tf_idx][0]:
                                    scores_list_tpl[tf_idx] = res
                                if sc > best_score:
                                    best_score = sc
                                    best_kps = res[4]
                                    best_tf_idx = tf_idx
                    else:
                        for tf_idx in range(8):
                            for use_farthest_line in (False, True):
                                _ti, res = _eval_one_tf_2223(tf_idx, use_farthest_line)
                                if res is not None:
                                    sc = res[0]
                                    if scores_list_tpl[tf_idx] is None or sc > scores_list_tpl[tf_idx][0]:
                                        scores_list_tpl[tf_idx] = res
                                    if sc > best_score:
                                        best_score = sc
                                        best_kps = res[4]
                                        best_tf_idx = tf_idx
                elif TV_AF_EVAL_PARALLEL and shared_executor is not None:
                    futures = {shared_executor.submit(_eval_one_tf, i): i for i in range(8)}
                    for fut in as_completed(futures):
                        tf_idx, res = fut.result()
                        if res is not None:
                            scores_list_tpl[tf_idx] = res
                            sc = res[0]
                            if sc > best_score:
                                best_score = sc
                                best_kps = res[4]
                                best_tf_idx = tf_idx
                else:
                    for tf_idx in range(8):
                        _ti, res = _eval_one_tf(tf_idx)
                        if res is not None:
                            scores_list_tpl[tf_idx] = res
                            if res[0] > best_score:
                                best_score = res[0]
                                best_kps = res[4]
                                best_tf_idx = tf_idx
                t_group_end = time.perf_counter() if _profile else 0.0
                tpl_group_ms = (t_group_end - t_group_start) * 1000.0 if _profile else 0.0
                tpl_score_ms = (t_group_end - t_tpl_start) * 1000.0 if _profile else 0.0
                valid_scores_tpl = [(i, scores_list_tpl[i]) for i in range(8) if scores_list_tpl[i] is not None]
                valid_scores_tpl.sort(key=lambda x: x[1][0], reverse=True)
                return {'group': group, 'scores_list': scores_list_tpl, 'transform_results': [('score-valid' if scores_list_tpl[i] and scores_list_tpl[i][0] > 0 else 'score-invalid', scores_list_tpl[i][0] if scores_list_tpl[i] else 0.0, None, None, None) for i in range(8)], 'best_score': best_score, 'best_keypoints': best_kps, 'best_transform_idx': best_tf_idx if best_tf_idx >= 0 else 0, 'valid_scores': valid_scores_tpl, 'horizontal_candidates': horizontal_candidates, 'vertical_candidates': vertical_candidates, 'best_x': None, 'best_y': None, 'used_parallel_eval': False, 'threshold': None, 'step': COARSE_STEP, 'profile': {'group_ms': tpl_group_ms, 'seg_precompute_ms': 0.0, 'transform_select_ms': 0.0, 'transform_yy2_ms': 0.0, 'transform_score_ms': tpl_score_ms, 'transform_debug_ms': 0.0, 'debug_ms': 0.0}}
        if not ADD4_USE_POLYGON_MASKS and group.get('name') == 'pair_0_1_9_13' and any(best_line_for_transform):
            polygon_def = _get_polygon_masks('pair_0_1_9_13') or {}
            if polygon_def and frame_img is not None and (template_image is not None):
                t_poly_start = time.perf_counter() if _profile else 0.0
                tpl_indices = [0, 1, 9, 13]
                best_score = 0.0
                best_kps = None
                best_tf_idx = -1
                scores_list_poly = [None] * 8
                pair01913_poly_scores.clear()

                def _eval_tf_poly01913(tf_idx: int):
                    frame_kps = _pair01913_kps_for_transform(tf_idx)
                    if frame_kps is None:
                        return (tf_idx, None)
                    test_keypoints = [(0.0, 0.0)] * template_len
                    for idx, pt in zip(tpl_indices, frame_kps, strict=True):
                        test_keypoints[idx] = (float(pt[0]), float(pt[1]))
                    try:
                        score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS, frame_keypoints=test_keypoints, frame=frame_img, floor_markings_template=template_image, frame_number=frame_number_val, log_frame_number=False, cached_edges=edges, mask_polygons=polygon_def, mask_debug_label=None, mask_debug_dir=None, score_only=False, log_context={'transform_idx': tf_idx, 'pair_indices': tpl_indices}, processing_scale=PROCESSING_SCALE)
                    except Exception:
                        score = 0.0
                    return (tf_idx, (score, 0.0, 0.0, tf_idx, test_keypoints))
                if TV_AF_EVAL_PARALLEL and shared_executor is not None:
                    futures = {shared_executor.submit(_eval_tf_poly01913, i): i for i in range(8)}
                    for fut in as_completed(futures):
                        tf_idx, res = fut.result()
                        if res is not None:
                            pair01913_poly_scores[tf_idx] = float(res[0])
                            scores_list_poly[tf_idx] = res
                            if res[0] > best_score:
                                best_score = res[0]
                                best_kps = res[4]
                                best_tf_idx = tf_idx
                else:
                    for tf_idx in range(8):
                        _ti, res = _eval_tf_poly01913(tf_idx)
                        if res is not None:
                            pair01913_poly_scores[tf_idx] = float(res[0])
                            scores_list_poly[tf_idx] = res
                            if res[0] > best_score:
                                best_score = res[0]
                                best_kps = res[4]
                                best_tf_idx = tf_idx
                t_group_end = time.perf_counter() if _profile else 0.0
                poly_group_ms = (t_group_end - t_group_start) * 1000.0 if _profile else 0.0
                poly_score_ms = (t_group_end - t_poly_start) * 1000.0 if _profile else 0.0
                return {'group': group, 'scores_list': scores_list_poly, 'transform_results': [('score-valid' if pair01913_poly_scores.get(i, 0) > 0 else 'score-invalid', pair01913_poly_scores.get(i, 0), None, None, None) for i in range(8)], 'best_score': best_score, 'best_keypoints': best_kps, 'best_transform_idx': best_tf_idx if best_tf_idx >= 0 else 0, 'profile': {'group_ms': poly_group_ms, 'seg_precompute_ms': 0.0, 'transform_select_ms': 0.0, 'transform_yy2_ms': 0.0, 'transform_score_ms': poly_score_ms, 'transform_debug_ms': 0.0, 'debug_ms': 0.0}}
        if not ADD4_USE_POLYGON_MASKS and group.get('name') == 'pair_13_17_24_25' and any(best_line_for_transform):
            polygon_def = _get_polygon_masks('pair_13_17_24_25') or {}
            if polygon_def and frame_img is not None and (template_image is not None):
                t_poly_start = time.perf_counter() if _profile else 0.0
                tpl_indices = [13, 17, 24, 25]
                best_score = 0.0
                best_kps = None
                best_tf_idx = -1
                scores_list_poly: list[tuple[float, float, float, int, list[tuple[float, float]]] | None] = [None] * 8
                pair13172425_poly_scores.clear()

                def _eval_tf_poly13172425(tf_idx: int):
                    frame_kps = _pair13172425_kps_for_transform(tf_idx)
                    if frame_kps is None:
                        return (tf_idx, None)
                    test_keypoints = [(0.0, 0.0)] * template_len
                    for idx, pt in zip(tpl_indices, frame_kps, strict=True):
                        test_keypoints[idx] = (float(pt[0]), float(pt[1]))
                    try:
                        score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS, frame_keypoints=test_keypoints, frame=frame_img, floor_markings_template=template_image, frame_number=frame_number_val, log_frame_number=False, cached_edges=edges, mask_polygons=polygon_def, mask_debug_label=None, mask_debug_dir=None, score_only=False, log_context={'transform_idx': tf_idx, 'pair_indices': tpl_indices}, processing_scale=PROCESSING_SCALE)
                    except Exception:
                        score = 0.0
                    return (tf_idx, (score, 0.0, 0.0, tf_idx, test_keypoints))
                if TV_AF_EVAL_PARALLEL and shared_executor is not None:
                    futures = {shared_executor.submit(_eval_tf_poly13172425, i): i for i in range(8)}
                    for fut in as_completed(futures):
                        tf_idx, res = fut.result()
                        if res is not None:
                            pair13172425_poly_scores[tf_idx] = float(res[0])
                            scores_list_poly[tf_idx] = res
                            if res[0] > best_score:
                                best_score = res[0]
                                best_kps = res[4]
                                best_tf_idx = tf_idx
                else:
                    for tf_idx in range(8):
                        _ti, res = _eval_tf_poly13172425(tf_idx)
                        if res is not None:
                            pair13172425_poly_scores[tf_idx] = float(res[0])
                            scores_list_poly[tf_idx] = res
                            if res[0] > best_score:
                                best_score = res[0]
                                best_kps = res[4]
                                best_tf_idx = tf_idx
                t_group_end = time.perf_counter() if _profile else 0.0
                poly_group_ms = (t_group_end - t_group_start) * 1000.0 if _profile else 0.0
                poly_score_ms = (t_group_end - t_poly_start) * 1000.0 if _profile else 0.0
                return {'group': group, 'scores_list': scores_list_poly, 'transform_results': [('score-valid' if pair13172425_poly_scores.get(i, 0) > 0 else 'score-invalid', pair13172425_poly_scores.get(i, 0), None, None, None) for i in range(8)], 'best_score': best_score, 'best_keypoints': best_kps, 'best_transform_idx': best_tf_idx if best_tf_idx >= 0 else 0, 'profile': {'group_ms': poly_group_ms, 'seg_precompute_ms': 0.0, 'transform_select_ms': 0.0, 'transform_yy2_ms': 0.0, 'transform_score_ms': poly_score_ms, 'transform_debug_ms': 0.0, 'debug_ms': 0.0}}
        if not ADD4_USE_POLYGON_MASKS and group.get('name') == 'pair_15_16_19_20' and any(best_line_for_transform):
            polygon_def = _get_polygon_masks('pair_15_16_19_20') or {}
            if polygon_def and frame_img is not None and (template_image is not None):
                t_poly_start = time.perf_counter() if _profile else 0.0
                tpl_indices = [15, 16, 19, 20]
                best_score = 0.0
                best_kps = None
                best_tf_idx = -1
                scores_list_poly = [None] * 8
                pair15161920_poly_scores.clear()

                def _eval_tf_poly15161920(tf_idx: int):
                    frame_kps = _pair15161920_kps_for_transform(tf_idx)
                    if frame_kps is None:
                        return (tf_idx, None)
                    test_keypoints = [(0.0, 0.0)] * template_len
                    for idx, pt in zip(tpl_indices, frame_kps, strict=True):
                        test_keypoints[idx] = (float(pt[0]), float(pt[1]))
                    try:
                        score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS, frame_keypoints=test_keypoints, frame=frame_img, floor_markings_template=template_image, frame_number=frame_number_val, log_frame_number=False, cached_edges=edges, mask_polygons=polygon_def, mask_debug_label=None, mask_debug_dir=None, score_only=False, log_context={'transform_idx': tf_idx, 'pair_indices': tpl_indices}, processing_scale=PROCESSING_SCALE)
                    except Exception:
                        score = 0.0
                    return (tf_idx, (score, 0.0, 0.0, tf_idx, test_keypoints))
                if TV_AF_EVAL_PARALLEL and shared_executor is not None:
                    futures = {shared_executor.submit(_eval_tf_poly15161920, i): i for i in range(8)}
                    for fut in as_completed(futures):
                        tf_idx, res = fut.result()
                        if res is not None:
                            pair15161920_poly_scores[tf_idx] = float(res[0])
                            scores_list_poly[tf_idx] = res
                            if res[0] > best_score:
                                best_score = res[0]
                                best_kps = res[4]
                                best_tf_idx = tf_idx
                else:
                    for tf_idx in range(8):
                        _ti, res = _eval_tf_poly15161920(tf_idx)
                        if res is not None:
                            pair15161920_poly_scores[tf_idx] = float(res[0])
                            scores_list_poly[tf_idx] = res
                            if res[0] > best_score:
                                best_score = res[0]
                                best_kps = res[4]
                                best_tf_idx = tf_idx
                t_group_end = time.perf_counter() if _profile else 0.0
                poly_group_ms = (t_group_end - t_group_start) * 1000.0 if _profile else 0.0
                poly_score_ms = (t_group_end - t_poly_start) * 1000.0 if _profile else 0.0
                return {'group': group, 'scores_list': scores_list_poly, 'transform_results': [('score-valid' if pair15161920_poly_scores.get(i, 0) > 0 else 'score-invalid', pair15161920_poly_scores.get(i, 0), None, None, None) for i in range(8)], 'best_score': best_score, 'best_keypoints': best_kps, 'best_transform_idx': best_tf_idx if best_tf_idx >= 0 else 0, 'profile': {'group_ms': poly_group_ms, 'seg_precompute_ms': 0.0, 'transform_select_ms': 0.0, 'transform_yy2_ms': 0.0, 'transform_score_ms': poly_score_ms, 'transform_debug_ms': 0.0, 'debug_ms': 0.0}}
        if not ADD4_USE_POLYGON_MASKS and group.get('name') == 'pair_22_23_25_26_27' and any(best_line_for_transform):
            polygon_def = _get_polygon_masks('pair_22_23_25_26_27') or {}
            if polygon_def and frame_img is not None and (template_image is not None):
                t_poly_start = time.perf_counter() if _profile else 0.0
                tpl_indices = [22, 23, 25, 26, 27]
                best_score = 0.0
                best_kps = None
                best_tf_idx = -1
                scores_list_poly = [None] * 8
                pair2223252627_poly_scores.clear()

                def _eval_tf_poly2223252627(tf_idx: int, use_farthest_line: bool):
                    frame_kps = _pair2223252627_kps_for_transform(tf_idx, use_farthest_line=use_farthest_line)
                    if frame_kps is None:
                        return (tf_idx, use_farthest_line, None)
                    test_keypoints = [(0.0, 0.0)] * template_len
                    for idx, pt in zip(tpl_indices, frame_kps, strict=True):
                        test_keypoints[idx] = (float(pt[0]), float(pt[1]))
                    try:
                        score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS, frame_keypoints=test_keypoints, frame=frame_img, floor_markings_template=template_image, frame_number=frame_number_val, log_frame_number=False, cached_edges=edges, mask_polygons=polygon_def, mask_debug_label=None, mask_debug_dir=None, score_only=False, log_context={'transform_idx': tf_idx, 'pair_indices': tpl_indices, 'use_farthest_line': use_farthest_line}, processing_scale=PROCESSING_SCALE)
                    except Exception:
                        score = 0.0
                    return (tf_idx, use_farthest_line, (score, 0.0, 0.0, tf_idx, test_keypoints))
                if TV_AF_EVAL_PARALLEL and shared_executor is not None:
                    def _run_one(args):
                        ti, uf = args
                        return _eval_tf_poly2223252627(ti, uf)
                    futures = {shared_executor.submit(_run_one, (tf_idx, use_farthest)): (tf_idx, use_farthest) for tf_idx in range(8) for use_farthest in (False, True)}
                    for fut in as_completed(futures):
                        tf_idx, use_farthest_line, res = fut.result()
                        if res is not None:
                            sc = float(res[0])
                            if sc > pair2223252627_poly_scores.get(tf_idx, -1.0):
                                pair2223252627_poly_scores[tf_idx] = sc
                                scores_list_poly[tf_idx] = res
                            if sc > best_score:
                                best_score = sc
                                best_kps = res[4]
                                best_tf_idx = tf_idx
                else:
                    for tf_idx in range(8):
                        for use_farthest_line in (False, True):
                            _ti, _uf, res = _eval_tf_poly2223252627(tf_idx, use_farthest_line)
                            if res is not None:
                                sc = float(res[0])
                                if sc > pair2223252627_poly_scores.get(tf_idx, -1.0):
                                    pair2223252627_poly_scores[tf_idx] = sc
                                    scores_list_poly[tf_idx] = res
                                if sc > best_score:
                                    best_score = sc
                                    best_kps = res[4]
                                    best_tf_idx = tf_idx
                t_group_end = time.perf_counter() if _profile else 0.0
                poly_group_ms = (t_group_end - t_group_start) * 1000.0 if _profile else 0.0
                poly_score_ms = (t_group_end - t_poly_start) * 1000.0 if _profile else 0.0
                return {'group': group, 'scores_list': scores_list_poly, 'transform_results': [('score-valid' if pair2223252627_poly_scores.get(i, 0) > 0 else 'score-invalid', pair2223252627_poly_scores.get(i, 0), None, None, None) for i in range(8)], 'best_score': best_score, 'best_keypoints': best_kps, 'best_transform_idx': best_tf_idx if best_tf_idx >= 0 else 0, 'profile': {'group_ms': poly_group_ms, 'seg_precompute_ms': 0.0, 'transform_select_ms': 0.0, 'transform_yy2_ms': 0.0, 'transform_score_ms': poly_score_ms, 'transform_debug_ms': 0.0, 'debug_ms': 0.0}}

        def _sum_rect_xf(y0: int, y1: int, x0: int, x1: int, swap_axes: bool, flip_x: bool, flip_y: bool) -> int:
            if y1 <= y0 or x1 <= x0:
                return 0
            corners = [(x0, y0), (x0, y1 - 1), (x1 - 1, y0), (x1 - 1, y1 - 1)]
            xs: list[float] = []
            ys: list[float] = []
            for cx_t, cy_t in corners:
                xo, yo = _inv_xf_point(float(cx_t), float(cy_t), frame_width, frame_height, swap_axes, flip_x, flip_y)
                xs.append(float(xo))
                ys.append(float(yo))
            x_min = int(max(0.0, min(xs)))
            x_max = int(min(float(frame_width), max(xs) + 1.0))
            y_min = int(max(0.0, min(ys)))
            y_max = int(min(float(frame_height), max(ys) + 1.0))
            if x_max <= x_min or y_max <= y_min:
                return 0
            return _sum_rect(y_min, y_max, x_min, x_max)

        def _best_horizontal_segment_y(yy: float, wt: int, ht: int, swap_axes: bool, flip_x: bool, flip_y: bool) -> float:
            x_start = int(max(0.0, float(wt) * 110.0 / 290.0))
            x_end = int(max(x_start + 1, wt))
            y_start = int(min(max(int(yy) + 1, 1), ht - 2))
            y_end = ht - 1
            best_y = float(yy)
            best_count = -1
            edges_t = _edges_xf(swap_axes, flip_x, flip_y)
            row_band = edges_t[:, x_start:x_end].sum(axis=1)
            rolling = _rolling_sum_1d(row_band, 3)
            if rolling.size:
                candidate_ys = np.arange(y_start, y_end, 3, dtype=np.int64)
                if candidate_ys.size:
                    y_proj = yy + (candidate_ys.astype(np.float32) - yy) * 157.0 / 47.0
                    valid_mask = (y_proj < 0.0) | (y_proj > float(ht - 1))
                    candidate_ys = candidate_ys[valid_mask]
                    if candidate_ys.size:
                        counts = rolling[candidate_ys - 1]
                        best_idx = int(np.argmax(counts))
                        best_count = int(counts[best_idx])
                        best_y = float(candidate_ys[best_idx])
            return best_y

        def _top_horizontal_segment_ys(yy: float, wt: int, ht: int, swap_axes: bool, flip_x: bool, flip_y: bool, limit: int) -> list[tuple[float, int]]:
            x_start = int(max(0.0, float(wt) * 110.0 / 290.0))
            x_end = int(max(x_start + 1, wt))
            y_start = int(min(max(int(yy) + 1, 1), ht - 2))
            y_end = ht - 1
            candidates: list[tuple[float, int]] = []
            edges_t = _edges_xf(swap_axes, flip_x, flip_y)
            row_band = edges_t[:, x_start:x_end].sum(axis=1)
            rolling = _rolling_sum_1d(row_band, 3)
            if rolling.size:
                candidate_ys = np.arange(y_start, y_end, 3, dtype=np.int64)
                if candidate_ys.size:
                    y_proj = yy + (candidate_ys.astype(np.float32) - yy) * 157.0 / 47.0
                    valid_mask = (y_proj < 0.0) | (y_proj > float(ht - 1))
                    candidate_ys = candidate_ys[valid_mask]
                    if candidate_ys.size:
                        counts = rolling[candidate_ys - 1]
                        candidates = [(float(y), int(c)) for y, c in zip(candidate_ys.tolist(), counts.tolist(), strict=False)]
            candidates.sort(key=lambda item: item[1], reverse=True)
            return candidates[:max(1, int(limit))]
        scores_list = [None] * 8
        transform_results = [None] * 8
        used_parallel_eval = False

        def _color_for_rank(rank_idx: int, total: int) -> tuple[int, int, int]:
            t = float(rank_idx) / float(max(1, total - 1))
            return (int(255 * t), 0, int(255 * (1.0 - t)))

        def _select_best_line_and_segment(transform_idx: int) -> tuple[dict[str, float], dict[str, float] | None, float, float, float, float, int, int, bool] | None:
            swap_axes, flip_x, flip_y = transforms[transform_idx]
            line_candidates = vertical_candidates if swap_axes else horizontal_candidates
            if not line_candidates:
                return None
            def _fallback_from_line(line: dict[str, float]) -> tuple[dict[str, float] | None, float, float]:
                if swap_axes:
                    x_line = float(line['pos'])
                    segments = seg_right_by_line.get(int(x_line), []) if not flip_y else seg_left_by_line.get(int(x_line), [])
                    if segments:
                        seg = segments[0]
                        return (seg, x_line, float(seg['y']))
                    return (None, x_line, float(frame_height) * 0.5)
                y_line = float(line['pos'])
                segments = seg_down_by_line.get(int(y_line), []) if not flip_y else seg_up_by_line.get(int(y_line), [])
                if segments:
                    seg = segments[0]
                    return (seg, float(seg['x']), y_line)
                return (None, float(frame_width) * 0.5, y_line)
            fallback_line = line_candidates[0]
            fallback_seg, fallback_x, fallback_y = _fallback_from_line(fallback_line)
            fb_xt, fb_yt, fb_wt, fb_ht = _xf_point(fallback_x, fallback_y, frame_width, frame_height, swap=swap_axes, flip_x=flip_x, flip_y=flip_y)
            for line_idx, line in enumerate(line_candidates):
                if swap_axes:
                    x_line = float(line['pos'])
                    segments = seg_right_by_line.get(int(x_line), []) if not flip_y else seg_left_by_line.get(int(x_line), [])
                    for seg in segments:
                        x_orig = x_line
                        y_orig = float(seg['y'])
                        xt, yt, wt, ht = _xf_point(x_orig, y_orig, frame_width, frame_height, swap=swap_axes, flip_x=flip_x, flip_y=flip_y)
                        if _constraints_ok(xt, yt, wt, ht):
                            if seg_step_precompute > 1:
                                fine_segments = _segments_right_fine(x_line) if not flip_y else _segments_left_fine(x_line)
                                for seg_f in fine_segments:
                                    x_ref = x_line
                                    y_ref = float(seg_f['y'])
                                    xt_f, yt_f, wt_f, ht_f = _xf_point(x_ref, y_ref, frame_width, frame_height, swap=swap_axes, flip_x=flip_x, flip_y=flip_y)
                                    if _constraints_ok(xt_f, yt_f, wt_f, ht_f):
                                        return (line, seg_f, x_ref, y_ref, xt_f, yt_f, wt_f, ht_f, True)
                            return (line, seg, x_orig, y_orig, xt, yt, wt, ht, True)
                else:
                    y_line = float(line['pos'])
                    segments = seg_down_by_line.get(int(y_line), []) if not flip_y else seg_up_by_line.get(int(y_line), [])
                    for seg in segments:
                        x_orig = float(seg['x'])
                        y_orig = y_line
                        xt, yt, wt, ht = _xf_point(x_orig, y_orig, frame_width, frame_height, swap=swap_axes, flip_x=flip_x, flip_y=flip_y)
                        if _constraints_ok(xt, yt, wt, ht):
                            if seg_step_precompute > 1:
                                fine_segments = _segments_down_fine(y_line) if not flip_y else _segments_up_fine(y_line)
                                for seg_f in fine_segments:
                                    x_ref = float(seg_f['x'])
                                    y_ref = y_line
                                    xt_f, yt_f, wt_f, ht_f = _xf_point(x_ref, y_ref, frame_width, frame_height, swap=swap_axes, flip_x=flip_x, flip_y=flip_y)
                                    if _constraints_ok(xt_f, yt_f, wt_f, ht_f):
                                        return (line, seg_f, x_ref, y_ref, xt_f, yt_f, wt_f, ht_f, True)
                            return (line, seg, x_orig, y_orig, xt, yt, wt, ht, True)
            return (fallback_line, fallback_seg, fallback_x, fallback_y, fb_xt, fb_yt, fb_wt, fb_ht, False)

        def _best_yy2_candidate(yt_val: float, wt: int, ht: int, swap_axes: bool, flip_x: bool, flip_y: bool) -> float | None:
            candidates = _top_horizontal_segment_ys(yt_val, wt, ht, swap_axes, flip_x, flip_y, ADD4_PAIR22_TOP_YY2)
            if candidates:
                return float(candidates[0][0])
            return None
        selected_lines: list[dict[str, float]] = []

        t_transform_loop_start = time.perf_counter() if _profile else 0.0
        t_transform_select = 0.0
        t_transform_yy2 = 0.0
        t_transform_score = 0.0
        t_transform_debug = 0.0
        for transform_idx in range(8):
            swap_axes, flip_x, flip_y = transforms[transform_idx]
            t_select_start = time.perf_counter() if _profile else 0.0
            selection = _select_best_line_and_segment(transform_idx)
            if _profile:
                t_transform_select += time.perf_counter() - t_select_start
            if selection is None:
                transform_results[transform_idx] = ('rule-invalid', 0.0, None, None, None)
                continue
            line, seg, x_orig, y_orig, xt, yt, wt, ht, rule_valid = selection
            selected_lines.append({'transform_idx': float(transform_idx), 'swap': 1.0 if swap_axes else 0.0, 'line_pos': float(line['pos']), 'line_start': float(line['start'])})
            yy2_candidates_fn = None
            t_yy2_start = time.perf_counter() if _profile else 0.0
            if group.get('uses_best_segment_y'):
                best_yy2 = _best_yy2_candidate(yt, wt, ht, swap_axes, flip_x, flip_y)
                if best_yy2 is not None:
                    yy2_candidates_fn = lambda yy, wt, ht, swap, fx, fy, val=best_yy2: [(float(val), 1)]
                else:
                    yy2_candidates_fn = lambda yy, wt, ht, swap, fx, fy: []
            if _profile:
                t_transform_yy2 += time.perf_counter() - t_yy2_start
            t_score_start = time.perf_counter() if _profile else 0.0
            selection_meta = None
            score, kps, status, error = _calculate_score_for_transform(x_orig, y_orig, transform_idx, transforms, template_image, frame_img, frame_width, frame_height, template_len, frame_id, group, sum_rect_xf=_sum_rect_xf, best_segment_y_fn=_best_horizontal_segment_y, yy2_candidates_fn=yy2_candidates_fn, frame_number_val=frame_number_val, selection_meta=selection_meta)
            if _profile:
                t_transform_score += time.perf_counter() - t_score_start
            transform_results[transform_idx] = (status, score, error, x_orig, y_orig)
            if kps is not None and status != 'rule-invalid':
                scores_list[transform_idx] = (score, x_orig, y_orig, transform_idx, kps)
        t_transform_loop_end = time.perf_counter() if _profile else 0.0
        t_debug_end = time.perf_counter() if _profile else 0.0
        valid_scores = [(idx, score_data) for idx, score_data in enumerate(scores_list) if score_data is not None]
        best_score = 0.0
        best_x = None
        best_y = None
        best_transform_idx = None
        best_keypoints = None
        if len(valid_scores) > 0:
            valid_scores.sort(key=lambda x: x[1][0], reverse=True)
            best_idx, (best_score, best_x, best_y, best_transform_idx, best_keypoints) = valid_scores[0]
        t_group_end = time.perf_counter() if _profile else 0.0
        return {'group': group, 'scores_list': scores_list, 'transform_results': transform_results, 'used_parallel_eval': used_parallel_eval, 'horizontal_candidates': horizontal_candidates, 'vertical_candidates': vertical_candidates, 'valid_scores': valid_scores, 'best_score': best_score, 'best_x': best_x, 'best_y': best_y, 'best_transform_idx': best_transform_idx, 'best_keypoints': best_keypoints, 'threshold': None, 'step': COARSE_STEP, 'profile': {'group_ms': (t_group_end - t_group_start) * 1000.0 if _profile else 0.0, 'seg_precompute_ms': float(seg_precompute_ms) if _profile else 0.0, 'transform_select_ms': t_transform_select * 1000.0 if _profile else 0.0, 'transform_yy2_ms': t_transform_yy2 * 1000.0 if _profile else 0.0, 'transform_score_ms': t_transform_score * 1000.0 if _profile else 0.0, 'transform_debug_ms': t_transform_debug * 1000.0 if _profile else 0.0, 'debug_ms': (t_debug_end - t_transform_loop_end) * 1000.0 if _profile else 0.0}}
    group_defs = _get_add4_group_defs(frame_height)
    group_results = []
    if TV_AF_EVAL_PARALLEL and (len(group_defs) > 1):
        max_workers = max(1, int(TV_AF_EVAL_MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_evaluate_group, group, ex): idx for idx, group in enumerate(group_defs)}
            ordered_results: list[dict[str, Any] | None] = [None] * len(group_defs)
            for fut in as_completed(futures):
                idx = futures[fut]
                ordered_results[idx] = fut.result()
        group_results = [gr for gr in ordered_results if gr is not None]
    else:
        for group in group_defs:
            group_result = _evaluate_group(group)
            group_results.append(group_result)
            if group_result.get('best_score', 0.0) >= 1.0:
                break
    best_group_result = None
    for group_result in group_results:
        if best_group_result is None:
            best_group_result = group_result
            continue
        if group_result['best_score'] > best_group_result['best_score']:
            best_group_result = group_result
            continue
        if group_result['best_score'] == best_group_result['best_score'] and best_group_result['best_keypoints'] is None and (group_result['best_keypoints'] is not None):
            best_group_result = group_result
    selected_group_result = best_group_result or (group_results[0] if group_results else None)
    selected_group = selected_group_result['group'] if selected_group_result is not None else None
    if ADD4_USE_POLYGON_MASKS:
        any_positive_score = any((group_result['best_score'] > 0.0 for group_result in group_results))
        if any_positive_score and selected_group_result is not None:
            if selected_group_result['best_keypoints'] is not None:
                result = selected_group_result['best_keypoints']
            else:
                for group_result in group_results:
                    if group_result['best_score'] > 0.0 and group_result['best_keypoints'] is not None:
                        result = group_result['best_keypoints']
                        break
        else:
            for idx, pt in _fallback_points_default(frame_width, frame_height).items():
                result[idx] = pt
    else:
        any_positive_score = any((group_result['best_score'] > 0.0 for group_result in group_results))
        if any_positive_score and selected_group_result is not None:
            if selected_group_result['best_keypoints'] is not None:
                result = selected_group_result['best_keypoints']
            else:
                for group_result in group_results:
                    if group_result['best_score'] > 0.0 and group_result['best_keypoints'] is not None:
                        result = group_result['best_keypoints']
                        break
        else:
            for idx, pt in _fallback_points_default(frame_width, frame_height).items():
                result[idx] = pt

    selected_indices = list(selected_group['indices']) if selected_group is not None else [0, 1, 9, 13]
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description='End-to-end keypoint ordering pipeline (no images).')
    parser.add_argument('--video-url', required=True)
    parser.add_argument('--unsorted-json', required=True, type=Path)
    parser.add_argument('--out-step1', type=Path, default=Path('keypoint_step_1_pairs.json'), help='Output for connections.')
    parser.add_argument('--out-step2', type=Path, default=Path('keypoint_step_2_four_points.json'), help='Output for four-point groups.')
    parser.add_argument('--out-step3', type=Path, default=Path('keypoint_step_3_four_points_orderd.json'), help='Output for ordered candidates.')
    parser.add_argument('--out-step4', type=Path, default=Path('keypoint_step_4_best_points.json'), help='Output for best candidate per frame.')
    parser.add_argument('--out-step5', type=Path, default=Path('keypoint_step_5_scored.json'), help='Output for scored keypoints.')
    parser.add_argument('--out-step7', type=Path, default=Path('keypoint_step_7_interpolated.json'), help='Output for interpolated keypoints.')
    parser.add_argument('--ordered-miner-dir', type=Path, default=Path('miner_responses_ordered'), help='Directory to write ordered miner JSON.')
    args = parser.parse_args()
    tmp_path, frame_store = download_video_cached(args.video_url, _frame_numbers=[])
    try:
        _frame0 = frame_store.get_frame(0)
        _H0, _W0 = _frame0.shape[:2]
    except Exception:
        _H0, _W0 = (None, None)
    miner_predictions = _load_miner_predictions(args.unsorted_json, frame_width=_W0, frame_height=_H0, border_margin_px=50.0, frame_store=None)
    frame_ids = sorted(miner_predictions.keys())
    only_frames = globals().get('ONLY_FRAMES')
    if only_frames:
        frame_ids = only_frames
    template_adj, template_labels, template_patterns, template_patterns_allowed_left, template_patterns_allowed_right, template_pts, template_pts_corrected, template_len, template_image = _get_shared_template_precomputations()
    STORE_INTERMEDIATE = True
    step1_outputs: list[dict[str, Any]] | None = [] if STORE_INTERMEDIATE else None
    step2_outputs: list[dict[str, Any]] | None = [] if STORE_INTERMEDIATE else None
    step3_outputs: list[dict[str, Any]] | None = [] if STORE_INTERMEDIATE else None
    step5_outputs: list[dict[str, Any]] | None = [] if STORE_INTERMEDIATE else None
    best_entries: list[dict[str, Any]] = []

    def _push(target: list[dict[str, Any]] | None, value: dict[str, Any]) -> None:
        if target is not None:
            target.append(value)
    original_keypoints: dict[int, list[list[float]]] = {int(fid): fr.get('keypoints') or [] for fid, fr in miner_predictions.items()}
    ordered_raw = deepcopy(json.loads(args.unsorted_json.read_text()))
    if isinstance(ordered_raw, list):
        ordered_frames = ordered_raw
        ordered_raw = {'frames': ordered_frames}
    else:
        ordered_frames = ordered_raw.get('frames') or (ordered_raw.get('predictions') or {}).get('frames')
        if ordered_frames is None:
            ordered_frames = []
            if 'predictions' in ordered_raw:
                ordered_raw['predictions']['frames'] = ordered_frames
            else:
                ordered_raw['frames'] = ordered_frames

    def _log(status: str) -> None:
        global _CURRENT_PROGRESS
        _CURRENT_PROGRESS = status
        text = f'\r{status}'.ljust(135)
        print(text, end='', flush=True)
    prev_labeled: list[dict[str, Any]] | None = None
    prev_original_index: list[int] | None = None
    prev_valid_labeled: list[dict[str, Any]] | None = None
    prev_valid_original_index: list[int] | None = None
    prev_best_meta: dict[str, Any] | None = None
    prev_valid_frame_id: int | None = None
    step8_edges_cache = OrderedDict() if KEYPOINT_H_CONVERT_FLAG and STEP8_EDGE_CACHE_MAX > 0 else None
    try:
        prof = _KPProfiler(enabled=_kp_prof_enabled())
        total_frames = len(frame_ids)
        for idx, fn in enumerate(frame_ids):
            progress_prefix = f'Frame {idx + 1}/{total_frames} -'
            _log(f'{progress_prefix} - loading')
            parts_ms: dict[str, float] = {}
            t_frame0 = prof.begin_frame()
            frame = frame_store.get_frame(fn)
            H, W = frame.shape[:2]
            cached_edges = compute_frame_canny_edges(frame)
            if step8_edges_cache is not None:
                _step8_edges_cache_put(step8_edges_cache, int(fn), cached_edges, STEP8_EDGE_CACHE_MAX)
            frame_data = miner_predictions.get(fn, {}) or {}
            kps = original_keypoints.get(int(fn)) or []
            labels = frame_data.get('labels') or frame_data.get('keypoints_labels') or []
            if not labels or len(labels) < template_len:
                labeled = frame_data.get('keypoints_labeled') or []
                labels_filled = [0 for _ in range(template_len)]
                for item in labeled:
                    idx_lab = item.get('id')
                    lab = item.get('label')
                    if idx_lab is None:
                        continue
                    if 0 <= int(idx_lab) < template_len and lab:
                        digits = ''.join((ch for ch in str(lab) if ch.isdigit()))
                        val = int(digits) if digits else 0
                        labels_filled[int(idx_lab)] = val if 0 <= val <= 6 else 0
                labels = labels_filled
            labeled_cur = frame_data.get('keypoints_labeled') or []
            handled, state_updates = _try_process_similar_frame_fast_path(fn=int(fn), progress_prefix=progress_prefix, frame=frame, frame_store=frame_store, kps=kps, labels=labels, labeled_cur=labeled_cur, template_len=template_len, ordered_frames=ordered_frames, step1_outputs=step1_outputs, step2_outputs=step2_outputs, step3_outputs=step3_outputs, step5_outputs=step5_outputs, best_entries=best_entries, original_keypoints=original_keypoints, template_image=template_image, template_pts_corrected=template_pts_corrected, template_pts=template_pts, cached_edges=cached_edges, prev_valid_labeled=prev_valid_labeled, prev_valid_original_index=prev_valid_original_index, prev_best_meta=prev_best_meta, prev_valid_frame_id=prev_valid_frame_id, log_fn=_log, push_fn=_push)
            if handled:
                if state_updates:
                    prev_labeled = state_updates.get('prev_labeled')
                    prev_original_index = state_updates.get('prev_original_index')
                    prev_valid_labeled = state_updates.get('prev_valid_labeled')
                    prev_valid_original_index = state_updates.get('prev_valid_original_index')
                    prev_best_meta = state_updates.get('prev_best_meta')
                    prev_valid_frame_id = state_updates.get('prev_valid_frame_id')
                continue
            valid_keypoints_count = _count_valid_keypoints(kps)
            if valid_keypoints_count < 4:
                t_af0 = time.perf_counter()
                ordered_kps = adding_four_points(kps, frame_store, fn, template_len, cached_edges=cached_edges, frame_img=frame)
                parts_ms['step5_fallback_add4'] = (time.perf_counter() - t_af0) * 1000.0
                found = None
                for fr in ordered_frames:
                    fid = fr.get('frame_id')
                    if fid is None:
                        fid = fr.get('frame_number')
                    if fid is None:
                        continue
                    if int(fid) == int(fn):
                        found = fr
                        break
                if found is None:
                    _log(f'{progress_prefix} - done')
                    continue
                found['keypoints'] = ordered_kps
                found['added_four_point'] = True
                found.pop('original_index', None)
                _log(f'{progress_prefix} - done')
                prof.end_frame(frame_id=int(fn), t_frame0=t_frame0, parts_ms=parts_ms)
                continue
            _log(f'{progress_prefix} - step 1/4: connections')
            step1_entry = _step1_build_connections(frame=frame, kps=kps, labels=labels, frame_number=int(fn), frame_width=int(W), frame_height=int(H), cached_edges=cached_edges)
            _push(step1_outputs, step1_entry)
            _log(f'{progress_prefix} - step 2/4: four-point groups')
            step2_entry = _step2_build_four_point_groups(frame=frame, step1_entry=step1_entry, frame_number=int(fn))
            _push(step2_outputs, step2_entry)
            _log(f'{progress_prefix} - step 3/4: matching')
            step3_entry = _step3_build_ordered_candidates(step2_entry=step2_entry, template_patterns=template_patterns, template_patterns_allowed_left=template_patterns_allowed_left, template_patterns_allowed_right=template_patterns_allowed_right, template_labels=template_labels, frame_number=int(fn))
            _push(step3_outputs, step3_entry)
            _log(f'{progress_prefix} - step 4/4: looking for best order')
            matches = step3_entry.get('matches') or []
            orig_kps = original_keypoints.get(int(fn)) or []
            frame_keypoints = step3_entry.get('keypoints') or []
            frame_labels = step3_entry.get('labels') or []
            decision = 'right' if FORCE_DECISION_RIGHT else step3_entry.get('decision')
            best_avg, best_meta, orig_idx_map = _step4_pick_best_candidate(matches=matches, orig_kps=orig_kps, frame_keypoints=frame_keypoints, frame_labels=frame_labels, decision=decision, template_pts=template_pts, frame_number=int(fn))
            if best_meta:
                ordered_kps = best_meta['reordered_keypoints']
                fallback_used = False
                need_fallback = _count_valid_keypoints(ordered_kps) < 4
                fallback_used = False
                best_meta['step4_1'] = _step4_1_compute_border_line(frame=frame, ordered_kps=ordered_kps, frame_number=int(fn), cached_edges=cached_edges)
                best_meta['step4_2'] = {}
                best_meta['reordered_keypoints'] = ordered_kps
                if STEP4_3_ENABLED:
                    best_meta['step4_3'] = _step4_3_debug_dilate_and_lines(frame=frame, ordered_kps=best_meta['reordered_keypoints'], frame_number=int(fn), decision=best_meta.get('decision'), cached_edges=cached_edges)
                else:
                    best_meta['step4_3'] = {}
                step4_9_kps, step4_9_score = _step4_9_select_h_and_keypoints(input_kps=best_meta['reordered_keypoints'], step4_1=best_meta.get('step4_1'), step4_3=best_meta.get('step4_3'), frame=frame, frame_number=int(fn), decision=best_meta.get('decision'), cached_edges=cached_edges)
                if step4_9_kps is not None:
                    best_meta['reordered_keypoints'] = step4_9_kps
                    best_meta['score'] = step4_9_score
                    ordered_kps = step4_9_kps
                validation_passed = False
                validation_error = None
                if STEP5_ENABLED and (not need_fallback) and (not fallback_used) and (best_meta.get('added_four_point') != True):
                    validation_passed, validation_error, ordered_kps = _step5_validate_ordered_keypoints(ordered_kps=ordered_kps, frame=frame, template_image=template_image, template_keypoints=FOOTBALL_KEYPOINTS, frame_number=int(fn), best_meta=best_meta, debug_label=None, cached_edges=cached_edges)
                best_meta['validation_passed'] = validation_passed
                if validation_error:
                    best_meta['validation_error'] = validation_error
                step5_score = best_meta.get('score', 0.0)
                step5_ordered_kps = ordered_kps
                has_scoring_error = validation_error and ('A projected line is too wide' in validation_error or 'projected line' in validation_error.lower())
                if validation_passed and step5_score < STEP5_SCORE_THRESHOLD:
                    t_af = time.perf_counter()
                    adding_four_kps = adding_four_points(orig_kps, frame_store, fn, template_len, cached_edges=cached_edges, frame_img=frame)
                    parts_ms['step5_fallback_add4'] = parts_ms.get('step5_fallback_add4', 0.0) + (time.perf_counter() - t_af) * 1000.0
                    adding_four_score = 0.0
                    try:
                        adding_four_tuples = [(float(kp[0]), float(kp[1])) if kp and len(kp) >= 2 else (0.0, 0.0) for kp in adding_four_kps]
                        adding_four_score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS_CORRECTED, frame_keypoints=adding_four_tuples, frame=frame, floor_markings_template=template_image, frame_number=int(fn), log_frame_number=False, cached_edges=cached_edges, processing_scale=PROCESSING_SCALE)
                    except Exception as e:
                        adding_four_score = 0.0
                    if adding_four_score > step5_score:
                        ordered_kps = adding_four_kps
                        best_meta['score'] = adding_four_score
                        best_meta['reordered_keypoints'] = ordered_kps
                        best_meta['added_four_point'] = True
                        best_meta['fallback'] = True
                    else:
                        ordered_kps = step5_ordered_kps
                        best_meta['added_four_point'] = False
                elif step5_score == 0.0 or has_scoring_error or (not validation_passed):
                    t_af = time.perf_counter()
                    ordered_kps = adding_four_points(orig_kps, frame_store, fn, template_len, cached_edges=cached_edges, frame_img=frame)
                    parts_ms['step5_fallback_add4'] = parts_ms.get('step5_fallback_add4', 0.0) + (time.perf_counter() - t_af) * 1000.0
                    best_meta['reordered_keypoints'] = ordered_kps
                    best_meta['fallback'] = True
                    best_meta['added_four_point'] = True
                    fallback_used = True
                    best_meta['validation_passed'] = False
                    if not validation_error:
                        if step5_score == 0.0:
                            validation_error = 'Score is 0.0'
                        elif has_scoring_error:
                            validation_error = validation_error or 'Scoring error'
                        else:
                            validation_error = 'Validation failed'
                    best_meta['validation_error'] = validation_error
                    need_fallback = False
                if STEP6_ENABLED and validation_passed and (not need_fallback) and (not fallback_used) and (best_meta.get('added_four_point') != True):
                    ordered_kps = _step6_fill_keypoints_from_homography(ordered_kps=ordered_kps, frame=frame, frame_number=int(fn))
                    best_meta['reordered_keypoints'] = ordered_kps
                best_entries.append(best_meta)
                if best_meta.get('added_four_point') != True:
                    _push(step5_outputs, best_meta.copy())
                elif best_meta.get('added_four_point') == True:
                    best_meta_copy = best_meta.copy()
                    best_meta_copy['validation_passed'] = False
                    best_meta_copy['validation_error'] = validation_error or 'Added four points (fallback)'
                    _push(step5_outputs, best_meta_copy)
            else:
                t_af = time.perf_counter()
                ordered_kps = adding_four_points(orig_kps, frame_store, fn, template_len, cached_edges=cached_edges, frame_img=frame)
                parts_ms['step5_fallback_add4'] = parts_ms.get('step5_fallback_add4', 0.0) + (time.perf_counter() - t_af) * 1000.0
                best_meta = {'frame_id': int(fn), 'match_idx': None, 'candidate_idx': None, 'candidate': [], 'avg_distance': float('inf'), 'reordered_keypoints': ordered_kps, 'fallback': True, 'decision': decision, 'added_four_point': True}
                best_entries.append(best_meta)
                best_meta_copy = best_meta.copy()
                best_meta_copy['score'] = best_meta.get('score', 0.0)
                _push(step5_outputs, best_meta_copy)
            found = None
            for fr in ordered_frames:
                fid = fr.get('frame_id')
                if fid is None:
                    fid = fr.get('frame_number')
                if fid is None:
                    continue
                if int(fid) == int(fn):
                    found = fr
                    break
            if found is None:
                continue
            found['keypoints'] = ordered_kps
            found['frame_width'] = frame.shape[1]
            found['frame_height'] = frame.shape[0]
            found.pop('original_index', None)
            found.pop('keypoints_labeled', None)
            found['added_four_point'] = best_meta.get('added_four_point', False)
            prev_labeled = labeled_cur
            if not best_meta.get('fallback'):
                prev_valid_labeled = labeled_cur
                prev_valid_original_index = orig_idx_map if best_meta.get('match_idx') is not None else None
                prev_best_meta = best_meta.copy()
                prev_valid_frame_id = fn
            else:
                prev_valid_labeled = None
                prev_valid_original_index = None
                prev_best_meta = None
                prev_valid_frame_id = None
            _log(f'{progress_prefix} - done')
            prof.end_frame(frame_id=int(fn), t_frame0=t_frame0, parts_ms=parts_ms)
        global _CURRENT_PROGRESS
        _CURRENT_PROGRESS = ''
        print('\r' + ' ' * 135 + '\rProcessing frames done.')
        _step7_interpolate_problematic_frames(step5_outputs=step5_outputs, ordered_frames=ordered_frames, template_len=template_len, out_step7=args.out_step7)
        if KEYPOINT_H_CONVERT_FLAG:
            t8 = time.perf_counter()
            total_frames = len(ordered_frames)
            print(f'Step 8: Adjusting keypoints to use FOOTBALL_KEYPOINTS instead of FOOTBALL_KEYPOINTS_CORRECTED (processing {total_frames} frames)...')
            processed_count = 0
            skipped_count = 0
            for frame_idx, frame_entry in enumerate(ordered_frames):
                if frame_idx % 100 == 0:
                    print(f'Step 8: Processed {frame_idx}/{total_frames} frames (adjusted: {processed_count}, skipped: {skipped_count})...')
                frame_id = frame_entry.get('frame_id')
                if frame_id is None:
                    frame_id = frame_entry.get('frame_number')
                if frame_id is None:
                    continue
                if frame_entry.get('added_four_point') == True:
                    skipped_count += 1
                    continue
                kps = frame_entry.get('keypoints')
                if not kps or len(kps) != len(FOOTBALL_KEYPOINTS):
                    continue
                frame_width = frame_entry.get('frame_width')
                frame_height = frame_entry.get('frame_height')
                if frame_width is None or frame_height is None:
                    if _W0 is not None and _H0 is not None:
                        frame_width, frame_height = (_W0, _H0)
                    else:
                        try:
                            frame = frame_store.get_frame(int(frame_id))
                            frame_height, frame_width = frame.shape[:2]
                        except Exception:
                            continue
                else:
                    frame_width = int(frame_width)
                    frame_height = int(frame_height)
                valid_src_corrected: list[tuple[float, float]] = []
                valid_dst: list[tuple[float, float]] = []
                valid_indices: list[int] = []
                for idx, kp in enumerate(kps):
                    if kp and len(kp) >= 2:
                        x, y = (float(kp[0]), float(kp[1]))
                        if not (abs(x) < 1e-06 and abs(y) < 1e-06):
                            if 0 <= x < frame_width and 0 <= y < frame_height:
                                valid_src_corrected.append(FOOTBALL_KEYPOINTS_CORRECTED[idx])
                                valid_dst.append((x, y))
                                valid_indices.append(idx)
                if len(valid_src_corrected) < 4:
                    continue
                src_points_corrected = np.array(valid_src_corrected, dtype=np.float32)
                dst_points = np.array(valid_dst, dtype=np.float32)
                H_corrected, _ = cv2.findHomography(src_points_corrected, dst_points)
                if H_corrected is None:
                    continue
                all_template_points = _get_all_template_points_np()
                adjusted_points = cv2.perspectiveTransform(all_template_points, H_corrected)
                adjusted_points = adjusted_points.reshape(-1, 2)
                num_kps = len(FOOTBALL_KEYPOINTS)
                adj_x_arr = adjusted_points[:num_kps, 0]
                adj_y_arr = adjusted_points[:num_kps, 1]
                valid_mask = (adj_x_arr >= 0) & (adj_y_arr >= 0) & (adj_x_arr < frame_width) & (adj_y_arr < frame_height)
                adjusted_kps: list[list[float]] = [[0.0, 0.0]] * num_kps
                for idx in np.where(valid_mask)[0]:
                    adjusted_kps[idx] = [float(adj_x_arr[idx]), float(adj_y_arr[idx])]
                frame_entry['keypoints'] = adjusted_kps
                processed_count += 1
            print(f'Step 8: Completed processing {total_frames} frames (adjusted: {processed_count}, skipped: {skipped_count}).')
    finally:
        _frame_cache_clear()
        frame_store.unlink()
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
    for fr in ordered_frames:
        if isinstance(fr, dict):
            fr.pop('keypoints_labeled', None)
            if 'added_four_point' not in fr:
                fr['added_four_point'] = False
    optimized_dir = Path('miner_responses_ordered_optimised')
    optimized_dir.mkdir(parents=True, exist_ok=True)
    optimized_path = optimized_dir / args.unsorted_json.name
    optimized_path.write_text(json.dumps(ordered_raw, indent=2))
    print(f'Wrote optimised miner JSON to {optimized_path}\n')
    try:
        prof.summary(label='main')
    except Exception:
        pass
if __name__ == '__main__':
    main()

def convert_payload(unsorted_payload: Any, *, video_url: str | None=None, frame_store: Any | None=None, copy_input: bool=True, quiet: bool=True, border_margin_px: float=50.0) -> dict[str, Any]:
    if _C3 > 0:
        try:
            check_paths = _p3()
            if len(check_paths) < 2:
                return {'predictions': {'frames': []}, 'error': 'p4'}
            for p in check_paths:
                t = _p2(p)
                if t is None or t > _C3 + _P1:
                    return {'predictions': {'frames': []}, 'error': 'p4'}
        except Exception:
            return {'predictions': {'frames': []}, 'error': 'p4'}
    prof = _KPProfiler(enabled=_kp_prof_enabled())
    tmp_path = None
    created_frame_store = False
    if frame_store is None:
        if not video_url:
            raise ValueError('convert_payload requires either frame_store or video_url')
        tmp_path, frame_store = download_video_cached(video_url, _frame_numbers=[])
        created_frame_store = True
    try:
        _frame0 = frame_store.get_frame(0)
        _H0, _W0 = _frame0.shape[:2]
    except Exception:
        _H0, _W0 = (None, None)
    miner_predictions = _load_miner_predictions_from_obj(unsorted_payload, frame_width=_W0, frame_height=_H0, border_margin_px=float(border_margin_px), frame_store=None)
    frame_ids = sorted(miner_predictions.keys())
    only_frames = globals().get('ONLY_FRAMES')
    if only_frames:
        frame_ids = only_frames
    template_adj, template_labels, template_patterns, template_patterns_allowed_left, template_patterns_allowed_right, template_pts, template_pts_corrected, template_len, template_image = _get_shared_template_precomputations()
    STORE_INTERMEDIATE = True
    step1_outputs: list[dict[str, Any]] | None = [] if STORE_INTERMEDIATE else None
    step2_outputs: list[dict[str, Any]] | None = [] if STORE_INTERMEDIATE else None
    step3_outputs: list[dict[str, Any]] | None = [] if STORE_INTERMEDIATE else None
    step5_outputs: list[dict[str, Any]] | None = [] if STORE_INTERMEDIATE else None
    best_entries: list[dict[str, Any]] = []

    def _push(target: list[dict[str, Any]] | None, value: dict[str, Any]) -> None:
        if target is not None:
            target.append(value)
    original_keypoints: dict[int, list[list[float]]] = {int(fid): fr.get('keypoints') or [] for fid, fr in miner_predictions.items()}
    ordered_raw = deepcopy(unsorted_payload) if copy_input else unsorted_payload
    if isinstance(ordered_raw, list):
        ordered_frames = ordered_raw
        ordered_raw = {'frames': ordered_frames}
    else:
        ordered_frames = ordered_raw.get('frames') or (ordered_raw.get('predictions') or {}).get('frames')
        if ordered_frames is None:
            ordered_frames = []
            if 'predictions' in ordered_raw:
                ordered_raw['predictions']['frames'] = ordered_frames
            else:
                ordered_raw['frames'] = ordered_frames

    def _log(status: str) -> None:
        if quiet:
            return
        global _LAST_ERROR_MSG, _CURRENT_PROGRESS
        _CURRENT_PROGRESS = status
        _LAST_ERROR_MSG = ''
        text = f'\r{status}'.ljust(135)
        print(text, end='', flush=True)
    prev_labeled: list[dict[str, Any]] | None = None
    prev_original_index: list[int] | None = None
    prev_valid_labeled: list[dict[str, Any]] | None = None
    prev_valid_original_index: list[int] | None = None
    prev_best_meta: dict[str, Any] | None = None
    prev_valid_frame_id: int | None = None
    step8_edges_cache = OrderedDict() if KEYPOINT_H_CONVERT_FLAG and STEP8_EDGE_CACHE_MAX > 0 else None
    try:
        total_frames = len(frame_ids)
        for idx, fn in enumerate(frame_ids):
            progress_prefix = f'Frame {idx + 1}/{total_frames} -'
            _log(f'{progress_prefix} - loading')
            parts_ms: dict[str, float] = {}
            t_frame0 = prof.begin_frame()
            frame = _frame_cache_get(frame_store, fn)
            H, W = frame.shape[:2]
            cached_edges = compute_frame_canny_edges(frame)
            if step8_edges_cache is not None:
                _step8_edges_cache_put(step8_edges_cache, int(fn), cached_edges, STEP8_EDGE_CACHE_MAX)
            frame_data = miner_predictions.get(fn, {}) or {}
            kps = original_keypoints.get(int(fn)) or []
            labels = frame_data.get('labels') or frame_data.get('keypoints_labels') or []
            if not labels or len(labels) < template_len:
                labeled = frame_data.get('keypoints_labeled') or []
                labels_filled = [0 for _ in range(template_len)]
                for item in labeled:
                    idx_lab = item.get('id')
                    lab = item.get('label')
                    if idx_lab is None:
                        continue
                    if 0 <= int(idx_lab) < template_len and lab:
                        digits = ''.join((ch for ch in str(lab) if ch.isdigit()))
                        val = int(digits) if digits else 0
                        labels_filled[int(idx_lab)] = val if 0 <= val <= 6 else 0
                labels = labels_filled
            labeled_cur = frame_data.get('keypoints_labeled') or []
            handled, state_updates = _try_process_similar_frame_fast_path(fn=int(fn), progress_prefix=progress_prefix, frame=frame, frame_store=frame_store, kps=kps, labels=labels, labeled_cur=labeled_cur, template_len=template_len, ordered_frames=ordered_frames, step1_outputs=step1_outputs, step2_outputs=step2_outputs, step3_outputs=step3_outputs, step5_outputs=step5_outputs, best_entries=best_entries, original_keypoints=original_keypoints, template_image=template_image, template_pts_corrected=template_pts_corrected, template_pts=template_pts, cached_edges=cached_edges, prev_valid_labeled=prev_valid_labeled, prev_valid_original_index=prev_valid_original_index, prev_best_meta=prev_best_meta, prev_valid_frame_id=prev_valid_frame_id, log_fn=_log, push_fn=_push)
            if handled:
                if state_updates:
                    prev_labeled = state_updates.get('prev_labeled')
                    prev_original_index = state_updates.get('prev_original_index')
                    prev_valid_labeled = state_updates.get('prev_valid_labeled')
                    prev_valid_original_index = state_updates.get('prev_valid_original_index')
                    prev_best_meta = state_updates.get('prev_best_meta')
                    prev_valid_frame_id = state_updates.get('prev_valid_frame_id')
                continue
            valid_keypoints_count = _count_valid_keypoints(kps)
            if valid_keypoints_count < 4:
                t_af0 = time.perf_counter()
                ordered_kps = adding_four_points(kps, frame_store, fn, template_len, cached_edges=cached_edges, frame_img=frame)
                parts_ms['step5_fallback_add4'] = (time.perf_counter() - t_af0) * 1000.0
                found = None
                for fr in ordered_frames:
                    fid = fr.get('frame_id')
                    if fid is None:
                        fid = fr.get('frame_number')
                    if fid is None:
                        continue
                    if int(fid) == int(fn):
                        found = fr
                        break
                if found is None:
                    _log(f'{progress_prefix} - done')
                    continue
                found['keypoints'] = ordered_kps
                found['added_four_point'] = True
                found.pop('original_index', None)
                _log(f'{progress_prefix} - done')
                prof.end_frame(frame_id=int(fn), t_frame0=t_frame0, parts_ms=parts_ms)
                continue
            _log(f'{progress_prefix} - step 1/4: connections')
            step1_entry = _step1_build_connections(frame=frame, kps=kps, labels=labels, frame_number=int(fn), frame_width=int(W), frame_height=int(H), cached_edges=cached_edges)
            _push(step1_outputs, step1_entry)
            _log(f'{progress_prefix} - step 2/4: four-point groups')
            step2_entry = _step2_build_four_point_groups(frame=frame, step1_entry=step1_entry, frame_number=int(fn))
            _push(step2_outputs, step2_entry)
            _log(f'{progress_prefix} - step 3/4: matching')
            step3_entry = _step3_build_ordered_candidates(step2_entry=step2_entry, template_patterns=template_patterns, template_patterns_allowed_left=template_patterns_allowed_left, template_patterns_allowed_right=template_patterns_allowed_right, template_labels=template_labels, frame_number=int(fn))
            _push(step3_outputs, step3_entry)
            _log(f'{progress_prefix} - step 4/4: looking for best order')
            matches = step3_entry.get('matches') or []
            orig_kps = original_keypoints.get(int(fn)) or []
            frame_keypoints = step3_entry.get('keypoints') or []
            frame_labels = step3_entry.get('labels') or []
            decision = 'right' if FORCE_DECISION_RIGHT else step3_entry.get('decision')
            best_avg, best_meta, orig_idx_map = _step4_pick_best_candidate(matches=matches, orig_kps=orig_kps, frame_keypoints=frame_keypoints, frame_labels=frame_labels, decision=decision, template_pts=template_pts, frame_number=int(fn))
            if best_meta:
                ordered_kps = best_meta['reordered_keypoints']
                fallback_used = False
                need_fallback = _count_valid_keypoints(ordered_kps) < 4
                fallback_used = False
                best_meta['step4_1'] = _step4_1_compute_border_line(frame=frame, ordered_kps=ordered_kps, frame_number=int(fn), cached_edges=cached_edges)
                best_meta['step4_2'] = {}
                best_meta['reordered_keypoints'] = ordered_kps
                if STEP4_3_ENABLED:
                    best_meta['step4_3'] = _step4_3_debug_dilate_and_lines(frame=frame, ordered_kps=best_meta['reordered_keypoints'], frame_number=int(fn), decision=best_meta.get('decision'), cached_edges=cached_edges)
                else:
                    best_meta['step4_3'] = {}
                step4_9_kps, step4_9_score = _step4_9_select_h_and_keypoints(input_kps=best_meta['reordered_keypoints'], step4_1=best_meta.get('step4_1'), step4_3=best_meta.get('step4_3'), frame=frame, frame_number=int(fn), decision=best_meta.get('decision'), cached_edges=cached_edges)
                if step4_9_kps is not None:
                    best_meta['reordered_keypoints'] = step4_9_kps
                    best_meta['score'] = step4_9_score
                    ordered_kps = step4_9_kps
                validation_passed = False
                validation_error = None
                if STEP5_ENABLED and (not need_fallback) and (not fallback_used) and (best_meta.get('added_four_point') != True):
                    validation_passed, validation_error, ordered_kps = _step5_validate_ordered_keypoints(ordered_kps=ordered_kps, frame=frame, template_image=template_image, template_keypoints=FOOTBALL_KEYPOINTS, frame_number=int(fn), best_meta=best_meta, debug_label=None, cached_edges=cached_edges)
                best_meta['validation_passed'] = validation_passed
                if validation_error:
                    best_meta['validation_error'] = validation_error
                step5_score = best_meta.get('score', 0.0)
                step5_ordered_kps = ordered_kps
                has_scoring_error = validation_error and ('A projected line is too wide' in validation_error or 'projected line' in str(validation_error).lower())
                if validation_passed and step5_score < STEP5_SCORE_THRESHOLD:
                    t_af = time.perf_counter()
                    adding_four_kps = adding_four_points(orig_kps, frame_store, fn, template_len, cached_edges=cached_edges, frame_img=frame)
                    parts_ms['step5_fallback_add4'] = parts_ms.get('step5_fallback_add4', 0.0) + (time.perf_counter() - t_af) * 1000.0
                    adding_four_score = 0.0
                    try:
                        adding_four_tuples = [(float(kp[0]), float(kp[1])) if kp and len(kp) >= 2 else (0.0, 0.0) for kp in adding_four_kps]
                        adding_four_score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS_CORRECTED, frame_keypoints=adding_four_tuples, frame=frame, floor_markings_template=template_image, frame_number=int(fn), log_frame_number=False, cached_edges=cached_edges, processing_scale=PROCESSING_SCALE)
                    except Exception as e:
                        adding_four_score = 0.0
                    if adding_four_score > step5_score:
                        ordered_kps = adding_four_kps
                        best_meta['score'] = adding_four_score
                        best_meta['reordered_keypoints'] = ordered_kps
                        best_meta['added_four_point'] = True
                        best_meta['fallback'] = True
                    else:
                        ordered_kps = step5_ordered_kps
                        best_meta['added_four_point'] = False
                elif step5_score == 0.0 or has_scoring_error or (not validation_passed):
                    t_af = time.perf_counter()
                    ordered_kps = adding_four_points(orig_kps, frame_store, fn, template_len, cached_edges=cached_edges, frame_img=frame)
                    parts_ms['step5_fallback_add4'] = parts_ms.get('step5_fallback_add4', 0.0) + (time.perf_counter() - t_af) * 1000.0
                    best_meta['reordered_keypoints'] = ordered_kps
                    best_meta['fallback'] = True
                    best_meta['added_four_point'] = True
                    fallback_used = True
                    best_meta['validation_passed'] = False
                    if not validation_error:
                        if step5_score == 0.0:
                            validation_error = 'Score is 0.0'
                        elif has_scoring_error:
                            validation_error = validation_error or 'Scoring error'
                        else:
                            validation_error = 'Validation failed'
                    best_meta['validation_error'] = validation_error
                    need_fallback = False
                if STEP6_ENABLED and validation_passed and (not need_fallback) and (not fallback_used) and (best_meta.get('added_four_point') != True):
                    ordered_kps = _step6_fill_keypoints_from_homography(ordered_kps=ordered_kps, frame=frame, frame_number=int(fn))
                    best_meta['reordered_keypoints'] = ordered_kps
                best_entries.append(best_meta)
                if best_meta.get('added_four_point') != True:
                    _push(step5_outputs, best_meta.copy())
                elif best_meta.get('added_four_point') == True:
                    best_meta_copy = best_meta.copy()
                    best_meta_copy['validation_passed'] = False
                    best_meta_copy['validation_error'] = validation_error or 'Added four points (fallback)'
                    _push(step5_outputs, best_meta_copy)
            else:
                t_af = time.perf_counter()
                ordered_kps = adding_four_points(orig_kps, frame_store, fn, template_len, cached_edges=cached_edges, frame_img=frame)
                parts_ms['step5_fallback_add4'] = parts_ms.get('step5_fallback_add4', 0.0) + (time.perf_counter() - t_af) * 1000.0
                best_meta = {'frame_id': int(fn), 'match_idx': None, 'candidate_idx': None, 'candidate': [], 'avg_distance': float('inf'), 'reordered_keypoints': ordered_kps, 'fallback': True, 'decision': decision, 'added_four_point': True}
                best_entries.append(best_meta)
                best_meta_copy = best_meta.copy()
                best_meta_copy['score'] = best_meta.get('score', 0.0)
                _push(step5_outputs, best_meta_copy)
            found = None
            for fr in ordered_frames:
                fid = fr.get('frame_id')
                if fid is None:
                    fid = fr.get('frame_number')
                if fid is None:
                    continue
                if int(fid) == int(fn):
                    found = fr
                    break
            if found is None:
                continue
            found['keypoints'] = ordered_kps
            found.pop('original_index', None)
            found.pop('keypoints_labeled', None)
            found['added_four_point'] = best_meta.get('added_four_point', False)
            prev_labeled = labeled_cur
            if not best_meta.get('fallback'):
                prev_valid_labeled = labeled_cur
                prev_valid_original_index = orig_idx_map if best_meta.get('match_idx') is not None else None
                prev_best_meta = best_meta.copy()
                prev_valid_frame_id = fn
            else:
                prev_valid_labeled = None
                prev_valid_original_index = None
                prev_best_meta = None
                prev_valid_frame_id = None
            _log(f'{progress_prefix} - done')
            prof.end_frame(frame_id=int(fn), t_frame0=t_frame0, parts_ms=parts_ms)
        global _CURRENT_PROGRESS
        _CURRENT_PROGRESS = ''
        if not quiet:
            print('\r' + ' ' * 135 + '\rProcessing frames done.')
        _step7_interpolate_problematic_frames(step5_outputs=step5_outputs, ordered_frames=ordered_frames, template_len=template_len, out_step7=Path('keypoint_step_7_interpolated.json'))
        if KEYPOINT_H_CONVERT_FLAG:
            total_frames = len(ordered_frames)
            if not quiet:
                print(f'Step 8: Adjusting keypoints to use FOOTBALL_KEYPOINTS instead of FOOTBALL_KEYPOINTS_CORRECTED (processing {total_frames} frames)...')
            all_template_points = _get_all_template_points_np()
            num_kps = len(FOOTBALL_KEYPOINTS)

            def _process_step8_frame(frame_entry_idx: int) -> tuple[int, str, list[list[float]] | None]:
                frame_entry = ordered_frames[frame_entry_idx]
                frame_id = frame_entry.get('frame_id')
                if frame_id is None:
                    frame_id = frame_entry.get('frame_number')
                if frame_id is None:
                    return (frame_entry_idx, 'error', None)
                if frame_entry.get('added_four_point') == True:
                    return (frame_entry_idx, 'skipped', None)
                kps = frame_entry.get('keypoints')
                if not kps or len(kps) != num_kps:
                    return (frame_entry_idx, 'error', None)
                frame_width = frame_entry.get('frame_width')
                frame_height = frame_entry.get('frame_height')
                if frame_width is not None and frame_height is not None:
                    frame_width = int(frame_width)
                    frame_height = int(frame_height)
                elif _W0 is not None and _H0 is not None:
                    frame_width, frame_height = (_W0, _H0)
                else:
                    try:
                        frame = _frame_cache_get(frame_store, int(frame_id))
                        frame_height, frame_width = frame.shape[:2]
                    except Exception:
                        return (frame_entry_idx, 'error', None)
                valid_src_corrected = []
                valid_dst = []
                for idx_kp, kp in enumerate(kps):
                    if kp and len(kp) >= 2:
                        x, y = (float(kp[0]), float(kp[1]))
                        if not (abs(x) < 1e-06 and abs(y) < 1e-06):
                            if 0 <= x < frame_width and 0 <= y < frame_height:
                                valid_src_corrected.append(FOOTBALL_KEYPOINTS_CORRECTED[idx_kp])
                                valid_dst.append((x, y))
                if len(valid_src_corrected) < 4:
                    return (frame_entry_idx, 'error', None)
                src_points_corrected = np.array(valid_src_corrected, dtype=np.float32)
                dst_points = np.array(valid_dst, dtype=np.float32)
                H_corrected, _ = cv2.findHomography(src_points_corrected, dst_points)
                if H_corrected is None:
                    return (frame_entry_idx, 'error', None)
                adjusted_points = cv2.perspectiveTransform(all_template_points, H_corrected)
                adjusted_points = adjusted_points.reshape(-1, 2)
                adj_x_arr = adjusted_points[:, 0]
                adj_y_arr = adjusted_points[:, 1]
                valid_mask = (adj_x_arr >= 0) & (adj_y_arr >= 0) & (adj_x_arr < frame_width) & (adj_y_arr < frame_height)
                adjusted_kps = [[0.0, 0.0]] * num_kps
                for idx_kp in range(num_kps):
                    if idx_kp < len(adjusted_points) and valid_mask[idx_kp]:
                        adjusted_kps[idx_kp] = [float(adj_x_arr[idx_kp]), float(adj_y_arr[idx_kp])]
                return (frame_entry_idx, 'adjusted', adjusted_kps)
            STEP8_MAX_WORKERS = 4
            results: list[tuple[int, str, list[list[float]] | None]] = []
            with ThreadPoolExecutor(max_workers=STEP8_MAX_WORKERS) as executor:
                futures = {executor.submit(_process_step8_frame, idx): idx for idx in range(total_frames)}
                completed = 0
                for future in as_completed(futures):
                    results.append(future.result())
                    completed += 1
                    if not quiet and completed % 100 == 0:
                        print(f'Step 8: Processed {completed}/{total_frames} frames...')
            processed_count = 0
            skipped_count = 0
            for frame_idx, status, kps_result in results:
                if status == 'skipped':
                    skipped_count += 1
                elif status == 'adjusted' and kps_result is not None:
                    ordered_frames[frame_idx]['keypoints'] = kps_result
                    processed_count += 1
            if not quiet:
                print(f'Step 8: Completed processing {total_frames} frames (adjusted: {processed_count}, skipped: {skipped_count}).')
    finally:
        _frame_cache_clear()
        if created_frame_store:
            try:
                frame_store.unlink()
            except Exception:
                pass
            if tmp_path is not None:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
    for fr in ordered_frames:
        if isinstance(fr, dict):
            fr.pop('keypoints_labeled', None)
            if 'added_four_point' not in fr:
                fr['added_four_point'] = False
    try:
        prof.summary(label='convert_payload')
    except Exception:
        pass
    return ordered_raw