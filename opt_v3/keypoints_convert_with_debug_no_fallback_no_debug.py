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
# ONLY_FRAMES = list(range(625, 626))
STEP4_1_ENABLED = True
STEP4_3_ENABLED = True
STEP6_ENABLED = True
STEP6_FILL_MISSING_ENABLED = True
_KP_SCORE_CACHE: OrderedDict[tuple, float] = OrderedDict()
_KP_WARP_CACHE: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
_KP_PRED_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_KP_CACHE_MAX = 512
STEP7_ENABLED = True
ADD4_POLYGON_SCORE_EDGE_RATE = False
ADD4_POLYGON_USE_JSON = True
PREV_RELATIVE_FLAG = True
BORDER_50PX_REMOVE_FLAG = True
KEYPOINT_H_CONVERT_FLAG = True
FORCE_DECISION_RIGHT = False
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
_eval_table_header_printed: bool = False
TV_KP_PROFILE_EVERY: int = 1
TV_AF_EVAL_MAX_WORKERS: int = 8

def _kp_prof_enabled() -> bool:
    return bool(TV_KP_PROFILE)

def _kp_prof_every() -> int:
    try:
        v = int(TV_KP_PROFILE_EVERY)
    except Exception:
        v = 1
    return max(1, int(v))

class _KPProfiler:

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._every = _kp_prof_every() if self.enabled else 0
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
        if self._every > 0 and self._frame_count % self._every == 0:
            keys = ['step1', 'step2', 'step3', 'step4', 'step4_1', 'step4_3', 'step4_9', 'step5_validate', 'step5_score_compare', 'step5_fallback_add4', 'step6_fill', 'step7_interpolate', 'step8_adjust']
            sum_steps = sum((parts_ms.get(k, 0.0) for k in keys))
            other_ms = total_ms - sum_steps
            seg = ' '.join([f'{k}_ms={parts_ms.get(k, 0.0):.2f}' for k in keys])
            seg = f'{seg} other_ms={other_ms:.2f}'
            extra_keys = sorted([k for k in parts_ms.keys() if k.startswith('step3_') and k not in ('step3',) or (k.startswith('step4_1_') and k not in ('step4_1',)) or (k.startswith('step4_3_') and k not in ('step4_3',)) or (k.startswith('step4_9_') and k not in ('step4_9',)) or (k.startswith('other_') and k != 'other')])
            extra = ''
            if extra_keys:
                extras_fmt: list[str] = []
                for k in extra_keys:
                    v = parts_ms.get(k, 0.0)
                    if k.endswith('_n'):
                        extras_fmt.append(f'{k}={int(v)}')
                    else:
                        extras_fmt.append(f'{k}_ms={float(v):.2f}')
                extra = ' ' + ' '.join(extras_fmt)
            print(f'[tv][kp_profile] frame={frame_id} total_ms={total_ms:.2f} {seg}{extra}')

    def summary(self, *, label: str) -> None:
        if not self.enabled:
            return
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        frames = max(1, self._frame_count)
        print(f'[tv][kp_profile] {label} frames={self._frame_count} elapsed_ms={elapsed_ms:.2f} avg_frame_ms={self.totals_ms.get('frame_total', 0.0) / frames:.2f}')
        keys = ['step1', 'step2', 'step3', 'step4', 'step4_1', 'step4_3', 'step4_9', 'step5_validate', 'step5_score_compare', 'step5_fallback_add4', 'step6_fill', 'step7_interpolate', 'step8_adjust']
        for k in keys:
            tot = self.totals_ms.get(k, 0.0)
            avg = tot / frames
            print(f'[tv][kp_profile] {label} total_{k}_ms={tot:.2f} avg_{k}_ms={avg:.2f}')
        sum_steps = sum((self.totals_ms.get(k, 0.0) for k in keys))
        frame_total = self.totals_ms.get('frame_total', 0.0)
        other_tot = frame_total - sum_steps
        other_avg = other_tot / frames
        print(f'[tv][kp_profile] {label} total_other_ms={other_tot:.2f} avg_other_ms={other_avg:.2f}')
        for ok in ('other_load', 'other_setup', 'other_similar', 'other_result'):
            ot = self.totals_ms.get(ok, 0.0)
            if ot != 0.0:
                print(f'[tv][kp_profile] {label} total_{ok}_ms={ot:.2f} avg_{ok}_ms={ot / frames:.2f}')

def _infer_step_index_from_name(name: str) -> int:
    n = (name or '').lower()
    if 'step4_mask_lines_predicted' in n or 'step5_overlap_expected_red_predicted_green' in n:
        return 4
    if 'step8' in n:
        return 8
    if 'step7' in n:
        return 7
    if 'step6' in n:
        return 6
    if 'step5' in n:
        return 5
    if 'step4_9' in n or 'step4_3' in n or 'step4_1' in n or ('step4' in n):
        return 4
    if 'step3' in n:
        return 3
    if 'step2' in n or 'four_points' in n:
        return 2
    if 'step1' in n or 'original_keypoints' in n:
        return 1
    return 9

def _sanitize_debug_label(name: str) -> str:
    safe = ''.join((ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in name or 'unnamed'))
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_') or 'unnamed'

def _save_ordered_step_image(step_idx: int, name: str, img: np.ndarray, frame_number: int | None) -> None:
    if img is None:
        return
    if frame_number is None:
        frame_tag = 'frame_unknown'
    else:
        frame_tag = f'frame_{int(frame_number):05d}'
    out_dir = Path('debug_frames') / 'step_by_step' / frame_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f'step_{int(step_idx):02d}_{_sanitize_debug_label(name)}.png'
    cv2.imwrite(str(out_dir / out_name), img)

def _render_keypoints_canvas(kps: list[list[float]], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((max(1, int(height)), max(1, int(width)), 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, kp in enumerate(kps or []):
        if not kp or len(kp) < 2:
            continue
        x, y = (float(kp[0]), float(kp[1]))
        if abs(x) < 1e-06 and abs(y) < 1e-06:
            continue
        ix, iy = (int(round(x)), int(round(y)))
        if 0 <= ix < canvas.shape[1] and 0 <= iy < canvas.shape[0]:
            cv2.circle(canvas, (ix, iy), 4, (0, 0, 255), -1)
            cv2.putText(canvas, str(idx), (ix + 5, iy - 4), font, 0.55, (0, 0, 255), 2)
    return canvas

def _render_keypoints_overlay_on_frame(frame: np.ndarray, kps: list[list[float]]) -> np.ndarray:
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, kp in enumerate(kps or []):
        if not kp or len(kp) < 2:
            continue
        x, y = (float(kp[0]), float(kp[1]))
        if abs(x) < 1e-06 and abs(y) < 1e-06:
            continue
        ix, iy = (int(round(x)), int(round(y)))
        if 0 <= ix < out.shape[1] and 0 <= iy < out.shape[0]:
            cv2.circle(out, (ix, iy), 4, (0, 0, 255), -1)
            cv2.putText(out, str(idx), (ix + 5, iy - 4), font, 0.55, (0, 0, 255), 2)
    return out

def _save_step(name: str, img: np.ndarray, frame_number: int | None=None) -> None:
    try:
        if img is None:
            return
        out_dir = Path('debug_frames') / 'eval_steps'
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = name
        if frame_number is not None:
            fname += f'_frame_{int(frame_number):03d}'
        out_path = out_dir / f'{fname}.png'
        cv2.imwrite(str(out_path), img)
        _save_ordered_step_image(_infer_step_index_from_name(name), name, img, frame_number)
    except Exception as exc:
        logger.error('Failed to save step %s: %s', name, exc)

def _step4_1_compute_border_line(*, frame: np.ndarray, ordered_kps: list[list[float]], frame_number: int, cached_edges: np.ndarray | None=None, prev_best_aay: int | None=None, prev_best_bby: int | None=None) -> dict[str, Any]:
    result: dict[str, Any] = {'frame_number': int(frame_number), 'has_h': False, 'p5': None, 'p29': None, 'Ey': None, 'Fy': None, 'best_aay': None, 'best_bby': None, 'best_white_rate': None, 'kkp5': None, 'kkp29': None, 'kkp13': None, 'kkp14': None, 'kkp15': None, 'kkp16': None, 'ef_left_a': None, 'ef_left_b': None, 'ef_right_a': None, 'ef_right_b': None, 'seg_13_16_a': None, 'seg_13_16_b': None, 'profile': {}}
    t_start = time.perf_counter()
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
            result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
            return result
        t_ = time.perf_counter()
        H_mat, _ = cv2.findHomography(np.array(filtered_src, dtype=np.float32), np.array(filtered_dst, dtype=np.float32))
        if H_mat is None:
            result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
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
            result['profile']['homography_red_ms'] = (time.perf_counter() - t_) * 1000.0
            result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
            return result
        Ey = y1 + (0.0 - x1) * (dy_line / dx_line)
        Fy = y1 + (float(W_img - 1) - x1) * (dy_line / dx_line)
        result['Ey'] = float(Ey)
        result['Fy'] = float(Fy)
        result['profile']['homography_red_ms'] = (time.perf_counter() - t_) * 1000.0
        all_proj = cv2.perspectiveTransform(np.array([[float(p[0]), float(p[1])] for p in FOOTBALL_KEYPOINTS_CORRECTED], dtype=np.float32).reshape(-1, 1, 2), H_mat)
        all_proj_list = [[float(all_proj[i][0][0]), float(all_proj[i][0][1])] for i in range(len(all_proj))]
        red_line_y_min = min(Ey, Fy)
        red_line_y_max = max(Ey, Fy)
        if red_line_y_max < 0 or red_line_y_min > H_img - 1:
            result['skipped'] = 'red line outside image range'
            result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
            return result
        t_ = time.perf_counter()
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
        result['profile']['edges_dilate_ms'] = (time.perf_counter() - t_) * 1000.0
        t_green = time.perf_counter()
        LINE_WIDTH_PX_41 = 15
        half_width_41 = LINE_WIDTH_PX_41 // 2
        LINE_SAMPLE_MAX_41 = 256
        edges_dilated_flat = edges_dilated.ravel() if _sloping_line_white_count_cy is not None else None
        _y_left_at_x: np.ndarray | None = None
        _y_right_at_x: np.ndarray | None = None

        def _border_line_white_rate(y_left: float, y_right: float) -> float:
            ax, ay = (0.0, y_left)
            bx, by = (float(W_img - 1), y_right)
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
            if ef_left is not None or ef_right is not None:
                if _y_left_at_x is not None and _y_right_at_x is not None:
                    qx_clip = np.clip(qx_flat, 0, W_img - 1)
                    under_mask = (qy_flat.astype(np.float64) >= _y_left_at_x[qx_clip]) | (qy_flat.astype(np.float64) >= _y_right_at_x[qx_clip])
                else:
                    under_mask = np.zeros(qx_flat.shape, dtype=bool)
                    if ef_left is not None:
                        (lax, lay), (lbx, lby) = ef_left
                        if abs(float(lbx - lax)) > 1e-09:
                            y_left_line = float(lay) + (qx_flat.astype(np.float64) - float(lax)) * (float(lby - lay) / float(lbx - lax))
                            under_mask |= qy_flat.astype(np.float64) >= y_left_line
                    if ef_right is not None:
                        (rax, ray), (rbx, rby) = ef_right
                        if abs(float(rbx - rax)) > 1e-09:
                            y_right_line = float(ray) + (qx_flat.astype(np.float64) - float(rax)) * (float(rby - ray) / float(rbx - rax))
                            under_mask |= qy_flat.astype(np.float64) >= y_right_line
                qx_flat = qx_flat[under_mask]
                qy_flat = qy_flat[under_mask]
                if qx_flat.size == 0:
                    return 0.0
            linear = np.unique(qy_flat.astype(np.int64) * W_img + qx_flat.astype(np.int64))
            qy_u = linear // W_img
            qx_u = linear % W_img
            white = int(np.count_nonzero(edges_dilated[qy_u, qx_u]))
            return float(white)

        def _line_segment_white_rate(ax: int, ay: int, bx: int, by: int, half_width: int=5) -> float:
            if _sloping_line_white_count_cy is not None and edges_dilated_flat is not None:
                try:
                    white, total = _sloping_line_white_count_cy(edges_dilated_flat, W_img, H_img, float(ax), float(ay), float(bx), float(by), half_width, LINE_SAMPLE_MAX_41)
                    return white / total if total > 0 else 0.0
                except Exception:
                    pass
            L = float(np.hypot(float(bx - ax), float(by - ay)))
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

        def _best_ef_line(idx_e: int, idx_f: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
            if idx_e >= len(all_proj_list) or idx_f >= len(all_proj_list):
                return None
            x_e, y_e = (float(all_proj_list[idx_e][0]), float(all_proj_list[idx_e][1]))
            x_f, y_f = (float(all_proj_list[idx_f][0]), float(all_proj_list[idx_f][1]))
            x_e_int, y_e_int = (int(round(x_e)), int(round(y_e)))
            x_f_int, y_f_int = (int(round(x_f)), int(round(y_f)))
            if abs(x_f_int - x_e_int) < 1 and abs(y_f_int - y_e_int) < 1:
                return None
            ef_a_y_min = y_e_int - 30
            ef_a_y_max = y_e_int + 30
            ef_b_y_min = y_f_int - 30
            ef_b_y_max = y_f_int + 30
            best_ef_rate = -1.0
            best_ef_a_y = y_e_int
            best_ef_b_y = y_f_int
            ef_a_y_coarse = list(range(ef_a_y_min, ef_a_y_max + 1, 8))
            if ef_a_y_coarse and ef_a_y_coarse[-1] != ef_a_y_max:
                ef_a_y_coarse.append(ef_a_y_max)
            ef_b_y_coarse = list(range(ef_b_y_min, ef_b_y_max + 1, 8))
            if ef_b_y_coarse and ef_b_y_coarse[-1] != ef_b_y_max:
                ef_b_y_coarse.append(ef_b_y_max)
            for ef_a_y in ef_a_y_coarse:
                for ef_b_y in ef_b_y_coarse:
                    rate = _line_segment_white_rate(x_e_int, ef_a_y, x_f_int, ef_b_y, half_width=5)
                    if rate > best_ef_rate:
                        best_ef_rate = rate
                        best_ef_a_y = ef_a_y
                        best_ef_b_y = ef_b_y
            ref_a_min = max(ef_a_y_min, best_ef_a_y - 2)
            ref_a_max = min(ef_a_y_max, best_ef_a_y + 2)
            ref_b_min = max(ef_b_y_min, best_ef_b_y - 2)
            ref_b_max = min(ef_b_y_max, best_ef_b_y + 2)
            for ef_a_y in range(ref_a_min, ref_a_max + 1):
                for ef_b_y in range(ref_b_min, ref_b_max + 1):
                    rate = _line_segment_white_rate(x_e_int, ef_a_y, x_f_int, ef_b_y, half_width=5)
                    if rate > best_ef_rate:
                        best_ef_rate = rate
                        best_ef_a_y = ef_a_y
                        best_ef_b_y = ef_b_y
            return ((x_e_int, best_ef_a_y), (x_f_int, best_ef_b_y))
        ef_left = _best_ef_line(0, 5)
        ef_right = _best_ef_line(24, 29)
        _y_left_at_x = np.full(W_img, -np.inf, dtype=np.float64)
        _y_right_at_x = np.full(W_img, -np.inf, dtype=np.float64)
        if ef_left is not None:
            (lax, lay), (lbx, lby) = ef_left
            if abs(float(lbx - lax)) > 1e-09:
                inv = float(lby - lay) / float(lbx - lax)
                for x in range(W_img):
                    _y_left_at_x[x] = float(lay) + (x - float(lax)) * inv
        if ef_right is not None:
            (rax, ray), (rbx, rby) = ef_right
            if abs(float(rbx - rax)) > 1e-09:
                inv = float(rby - ray) / float(rbx - rax)
                for x in range(W_img):
                    _y_right_at_x[x] = float(ray) + (x - float(rax)) * inv

        def _best_green_segment(idx_a: int, idx_b: int) -> tuple[tuple[int, int], tuple[int, int], float] | None:
            if idx_a >= len(all_proj_list) or idx_b >= len(all_proj_list):
                return None
            x_a, y_a = (float(all_proj_list[idx_a][0]), float(all_proj_list[idx_a][1]))
            x_b, y_b = (float(all_proj_list[idx_b][0]), float(all_proj_list[idx_b][1]))
            x_a_int, y_a_int = (int(round(x_a)), int(round(y_a)))
            x_b_int, y_b_int = (int(round(x_b)), int(round(y_b)))
            if abs(x_b_int - x_a_int) < 1 and abs(y_b_int - y_a_int) < 1:
                return None
            a_y_min = y_a_int - 30
            a_y_max = y_a_int + 30
            b_y_min = y_b_int - 30
            b_y_max = y_b_int + 30
            best_rate = -1.0
            best_a_y = y_a_int
            best_b_y = y_b_int
            a_y_coarse = list(range(a_y_min, a_y_max + 1, 8))
            if a_y_coarse and a_y_coarse[-1] != a_y_max:
                a_y_coarse.append(a_y_max)
            b_y_coarse = list(range(b_y_min, b_y_max + 1, 8))
            if b_y_coarse and b_y_coarse[-1] != b_y_max:
                b_y_coarse.append(b_y_max)
            for ay in a_y_coarse:
                for by in b_y_coarse:
                    rate = _line_segment_white_rate(x_a_int, ay, x_b_int, by, half_width=5)
                    if rate > best_rate:
                        best_rate = rate
                        best_a_y = ay
                        best_b_y = by
            ref_a_min = max(a_y_min, best_a_y - 2)
            ref_a_max = min(a_y_max, best_a_y + 2)
            ref_b_min = max(b_y_min, best_b_y - 2)
            ref_b_max = min(b_y_max, best_b_y + 2)
            for ay in range(ref_a_min, ref_a_max + 1):
                for by in range(ref_b_min, ref_b_max + 1):
                    rate = _line_segment_white_rate(x_a_int, ay, x_b_int, by, half_width=5)
                    if rate > best_rate:
                        best_rate = rate
                        best_a_y = ay
                        best_b_y = by
            return ((x_a_int, best_a_y), (x_b_int, best_b_y), float(best_rate))
        seg_13_16 = _best_green_segment(13, 16)
        STEP_COARSE_41 = 20
        REFINE_RADIUS_41 = 2
        STEP_FINE_41 = 2
        SEED_WINDOW_41 = 20
        SMALL_RANGE_41 = 80
        WIDE_RANGE_41 = 200
        ADAPTIVE_MIN_WHITE_41 = 80
        best_rate = -1.0
        best_aay: int | None = None
        best_bby: int | None = None

        def _run_coarse_refine(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> None:
            nonlocal best_rate, best_aay, best_bby
            for aay in range(a_lo, a_hi + 1, STEP_COARSE_41):
                for bby in range(b_lo, b_hi + 1, STEP_COARSE_41):
                    rate = _border_line_white_rate(float(aay), float(bby))
                    if rate > best_rate:
                        best_rate = rate
                        best_aay = aay
                        best_bby = bby
            if best_aay is not None and best_bby is not None:
                a_ref_min = max(a_lo, best_aay - REFINE_RADIUS_41)
                a_ref_max = min(a_hi, best_aay + REFINE_RADIUS_41)
                b_ref_min = max(b_lo, best_bby - REFINE_RADIUS_41)
                b_ref_max = min(b_hi, best_bby + REFINE_RADIUS_41)
                for aay in range(a_ref_min, a_ref_max + 1, STEP_FINE_41):
                    for bby in range(b_ref_min, b_ref_max + 1, STEP_FINE_41):
                        rate = _border_line_white_rate(float(aay), float(bby))
                        if rate > best_rate:
                            best_rate = rate
                            best_aay = aay
                            best_bby = bby
        used_seed = False
        if prev_best_aay is not None and prev_best_bby is not None:
            a_seed_lo = max(0, prev_best_aay - SEED_WINDOW_41)
            a_seed_hi = min(H_img - 1, prev_best_aay + SEED_WINDOW_41)
            b_seed_lo = max(0, prev_best_bby - SEED_WINDOW_41)
            b_seed_hi = min(H_img - 1, prev_best_bby + SEED_WINDOW_41)
            _run_coarse_refine(a_seed_lo, a_seed_hi, b_seed_lo, b_seed_hi)
            if best_rate >= ADAPTIVE_MIN_WHITE_41:
                used_seed = True
        if not used_seed:
            a_min_s = int(np.clip(Ey - SMALL_RANGE_41, 0, H_img - 1))
            a_max_s = int(np.clip(Ey + SMALL_RANGE_41, 0, H_img - 1))
            b_min_s = int(np.clip(Fy - SMALL_RANGE_41, 0, H_img - 1))
            b_max_s = int(np.clip(Fy + SMALL_RANGE_41, 0, H_img - 1))
            _run_coarse_refine(a_min_s, a_max_s, b_min_s, b_max_s)
            if best_rate < ADAPTIVE_MIN_WHITE_41:
                a_min = int(np.floor(Ey - WIDE_RANGE_41))
                a_max = int(np.ceil(Ey + WIDE_RANGE_41))
                b_min = int(np.floor(Fy - WIDE_RANGE_41))
                b_max = int(np.ceil(Fy + WIDE_RANGE_41))
                a_min = max(0, min(a_min, H_img - 1))
                a_max = max(0, min(a_max, H_img - 1))
                b_min = max(0, min(b_min, H_img - 1))
                b_max = max(0, min(b_max, H_img - 1))
                _run_coarse_refine(a_min, a_max, b_min, b_max)
        if best_aay is None or best_bby is None:
            best_aay = int(round(Ey))
            best_bby = int(round(Fy))
            if best_rate < 0.0:
                best_rate = _border_line_white_rate(float(best_aay), float(best_bby))
        result['profile']['green_search_ms'] = (time.perf_counter() - t_green) * 1000.0
        if best_aay is not None and best_bby is not None:
            pA = (0, best_aay)
            pB = (W_img - 1, best_bby)
            green_thickness = 3
            result['best_aay'] = best_aay
            result['best_bby'] = best_bby
            result['best_white_rate'] = best_rate
            if seg_13_16 is not None:
                (sax, say), (sbx, sby), seg_rate = seg_13_16
                result['seg_13_16_a'] = [float(sax), float(say)]
                result['seg_13_16_b'] = [float(sbx), float(sby)]
                result.setdefault('profile', {})['seg_13_16_best_white_rate'] = float(seg_rate)

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
            t_kkp = time.perf_counter()

            def _project_point_to_line(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float] | None:
                vx, vy = (bx - ax, by - ay)
                vv = vx * vx + vy * vy
                if vv < 1e-12:
                    return None
                t = ((px - ax) * vx + (py - ay) * vy) / vv
                return (ax + t * vx, ay + t * vy)
            if seg_13_16 is not None:
                (sax, say), (sbx, sby), _ = seg_13_16
                saxf, sayf, sbxf, sbyf = (float(sax), float(say), float(sbx), float(sby))
                pt16 = _line_line_intersection_41(green_ax, green_ay, green_bx, green_by, saxf, sayf, sbxf, sbyf)
                if pt16 is not None:
                    result['kkp16'] = [float(pt16[0]), float(pt16[1])]
                for idx in (13, 14, 15):
                    src_pt: tuple[float, float] | None = None
                    if idx < len(ordered_kps):
                        kp = ordered_kps[idx]
                        if kp and len(kp) >= 2:
                            xk, yk = (float(kp[0]), float(kp[1]))
                            if not (abs(xk) < 1e-06 and abs(yk) < 1e-06):
                                src_pt = (xk, yk)
                    if src_pt is None and idx < len(all_proj_list):
                        src_pt = (float(all_proj_list[idx][0]), float(all_proj_list[idx][1]))
                    if src_pt is None:
                        continue
                    proj_pt = _project_point_to_line(float(src_pt[0]), float(src_pt[1]), saxf, sayf, sbxf, sbyf)
                    if proj_pt is None:
                        continue
                    result[f'kkp{idx}'] = [float(proj_pt[0]), float(proj_pt[1])]
            if ef_left is not None:
                (lax, lay), (lbx, lby) = ef_left
                result['ef_left_a'] = [float(lax), float(lay)]
                result['ef_left_b'] = [float(lbx), float(lby)]
                pt = _line_line_intersection_41(green_ax, green_ay, green_bx, green_by, float(lax), float(lay), float(lbx), float(lby))
                if pt is not None:
                    result['kkp5'] = [float(pt[0]), float(pt[1])]
            if ef_right is not None:
                (rax, ray), (rbx, rby) = ef_right
                result['ef_right_a'] = [float(rax), float(ray)]
                result['ef_right_b'] = [float(rbx), float(rby)]
                pt = _line_line_intersection_41(green_ax, green_ay, green_bx, green_by, float(rax), float(ray), float(rbx), float(rby))
                if pt is not None:
                    result['kkp29'] = [float(pt[0]), float(pt[1])]
            result['profile']['kkp_intersect_ms'] = (time.perf_counter() - t_kkp) * 1000.0
        result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
    except Exception as exc:
        logger.error('Failed Step 4.1 border line compute for frame %s: %s', frame_number, exc)
        result['error'] = str(exc)
    if 'total_ms' not in result.get('profile', {}):
        result.setdefault('profile', {})['total_ms'] = (time.perf_counter() - t_start) * 1000.0
    return result

def _step4_9_select_h_and_keypoints(*, input_kps: list[list[float]], step4_1: dict[str, Any] | None, step4_3: dict[str, Any] | None, frame: np.ndarray, frame_number: int, decision: str | None, cached_edges: np.ndarray | None=None) -> tuple[list[list[float]] | None, float]:
    H_img, W_img = frame.shape[:2]
    template = FOOTBALL_KEYPOINTS_CORRECTED
    n_tpl = len(template)
    if not input_kps or len(input_kps) < n_tpl:
        return (None, 0.0)

    def _build_weighted_correspondences(kps: list[list[float]], weight_43: int, weight_41: int, use_right: bool, weight_idx16: int=1) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
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
            elif i in idx_41 and i != 16:
                w = weight_41
            if i == 16:
                w = weight_idx16
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
            return evaluate_keypoints_for_frame(template_keypoints=template, frame_keypoints=frame_kps_tuples, frame=frame, floor_markings_template=template_image, frame_number=frame_number, cached_edges=cached_edges)
        except Exception:
            return 0.0

    def _score_h_with_reason(H: np.ndarray | None) -> tuple[float, str | None]:
        if H is None:
            return (0.0, None)
        try:
            tpl_pts = np.array([[float(p[0]), float(p[1])] for p in template], dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(tpl_pts, H).reshape(-1, 2)
            frame_kps_tuples = [(float(proj[i][0]), float(proj[i][1])) for i in range(len(proj))]
            template_image = challenge_template()
            score = float(evaluate_keypoints_for_frame(template_keypoints=template, frame_keypoints=frame_kps_tuples, frame=frame, floor_markings_template=template_image, frame_number=frame_number, cached_edges=cached_edges))
            if score > 0.0:
                return (score, None)
            try:
                warped = project_image_using_keypoints(image=template_image, source_keypoints=template, destination_keypoints=frame_kps_tuples, destination_width=W_img, destination_height=H_img)
                _ = extract_masks_for_ground_and_lines(warped, debug_frame_id=frame_number)
            except InvalidMask as e:
                return (score, str(e))
            except Exception:
                pass
            return (score, None)
        except Exception:
            return (0.0, None)

    def _homography_from_variant(label: str, kps_variant: list[list[float]]) -> np.ndarray | None:
        if label == 'H1':
            return _homography_from_kps(kps_variant)
        if label == 'H2':
            src2, dst2 = _build_weighted_correspondences(kps_variant, weight_43=h2_weight_43, weight_41=1, use_right=use_right)
            if len(src2) >= 4:
                H2_try, _ = cv2.findHomography(np.array(src2, dtype=np.float32), np.array(dst2, dtype=np.float32))
                return H2_try
            return None
        if label == 'H3':
            src3, dst3 = _build_weighted_correspondences(kps_variant, weight_43=h3_weight_43, weight_41=h3_weight_41, use_right=use_right, weight_idx16=h3h4_weight_16)
            if len(src3) >= 4:
                H3_try, _ = cv2.findHomography(np.array(src3, dtype=np.float32), np.array(dst3, dtype=np.float32))
                return H3_try
            return None
        if label == 'H4':
            src4, dst4 = _build_weighted_correspondences(kps_variant, weight_43=1, weight_41=h4_weight_41, use_right=use_right, weight_idx16=h3h4_weight_16)
            if len(src4) >= 4:
                H4_try, _ = cv2.findHomography(np.array(src4, dtype=np.float32), np.array(dst4, dtype=np.float32))
                return H4_try
            return None
        return None

    def _retry_wide_line_for_variant(label: str, H_init: np.ndarray | None, kps_variant: list[list[float]]) -> tuple[float, np.ndarray | None]:

        def _save_retry_debug_image(tag: str, H_val: np.ndarray | None, kps_draw: list[list[float]]) -> None:
            if H_val is None:
                return
            try:
                if cached_edges is not None:
                    edges = cached_edges
                else:
                    edges = compute_frame_canny_edges(frame)
                try:
                    dil_k = _kernel_rect_3()
                except Exception:
                    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                edges_dilated = cv2.dilate(edges, dil_k, iterations=3)
                edge_dilated_bgr = np.zeros((H_img, W_img, 3), dtype=np.uint8)
                edge_dilated_bgr[edges_dilated > 0] = (255, 255, 255)
                template_image = challenge_template()
                img = edge_dilated_bgr.copy()
                if template_image is not None and template_image.size > 0:
                    warped = cv2.warpPerspective(template_image, H_val, (W_img, H_img))
                    img = cv2.addWeighted(img, 0.55, warped, 0.45, 0)
                for kp in kps_draw:
                    if not kp or len(kp) < 2:
                        continue
                    x, y = (float(kp[0]), float(kp[1]))
                    if abs(x) < 1e-06 and abs(y) < 1e-06:
                        continue
                    ix, iy = (int(round(x)), int(round(y)))
                    if 0 <= ix < W_img and 0 <= iy < H_img:
                        cv2.circle(img, (ix, iy), 5, (0, 0, 255), -1)
                out_dir = Path('debug_frames') / 'x'
                out_dir.mkdir(parents=True, exist_ok=True)
                safe_tag = str(tag).replace('+', 'p').replace('-', 'm')
                out_path = out_dir / f'frame_{int(frame_number):05d}_step4_9_{label}_retry_{safe_tag}.png'
                cv2.imwrite(str(out_path), img)
                _save_ordered_step_image(4, f'step4_9_{label}_retry_{safe_tag}', img, frame_number)
            except Exception:
                pass
        score_init, reason_init = _score_h_with_reason(H_init)
        if not (score_init <= 0.0 and reason_init and ('A projected line is too wide' in reason_init)):
            return (score_init, H_init)
        valid_indices = [i for i in range(min(len(kps_variant), n_tpl)) if kps_variant[i] and len(kps_variant[i]) >= 2 and (not (abs(float(kps_variant[i][0])) < 1e-06 and abs(float(kps_variant[i][1])) < 1e-06)) and (0 <= float(kps_variant[i][0]) < W_img) and (0 <= float(kps_variant[i][1]) < H_img)]
        if not valid_indices:
            return (score_init, H_init)
        smallest_y_idx = min(valid_indices, key=lambda i: (float(kps_variant[i][1]), float(kps_variant[i][0]), int(i)))
        base_x = float(kps_variant[smallest_y_idx][0])
        base_y = float(kps_variant[smallest_y_idx][1])
        for dx, dy, tag in [(0.0, -1.0, 'y-1'), (0.0, 1.0, 'y+1'), (1.0, 0.0, 'x+1'), (-1.0, 0.0, 'x-1')]:
            nx, ny = (base_x + dx, base_y + dy)
            if not (0.0 <= nx < W_img and 0.0 <= ny < H_img):
                continue
            kps_try = [list(kp) if kp else [0.0, 0.0] for kp in kps_variant]
            kps_try[smallest_y_idx] = [nx, ny]
            H_try = _homography_from_variant(label, kps_try)
            if H_try is None:
                continue
            _save_retry_debug_image(tag, H_try, kps_try)
            score_try, reason_try = _score_h_with_reason(H_try)
            if score_try > 0.0:
                return (score_try, H_try)
        return (score_init, H_init)

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
    h2_weight_43 = 4 if STEP4_3_ENABLED else 1
    h3_weight_43 = 2 if STEP4_3_ENABLED else 1
    h3_weight_41 = 4 if STEP4_1_ENABLED else 1
    h4_weight_41 = 4 if STEP4_1_ENABLED else 1
    h3h4_weight_16 = 8
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
        for idx in (13, 14, 15, 16):
            if step4_1.get(f'kkp{idx}') is not None:
                kps3[idx] = list(step4_1[f'kkp{idx}'])
    kps4 = [list(kp) if kp else [0.0, 0.0] for kp in kps1]
    if step4_1:
        if step4_1.get('kkp5') is not None:
            kps4[5] = list(step4_1['kkp5'])
        if step4_1.get('kkp29') is not None:
            kps4[29] = list(step4_1['kkp29'])
        for idx in (13, 14, 15, 16):
            if step4_1.get(f'kkp{idx}') is not None:
                kps4[idx] = list(step4_1[f'kkp{idx}'])
    H1 = None
    if step4_1 and step4_1.get('H') is not None:
        try:
            H1 = np.array(step4_1['H'], dtype=np.float32)
        except (TypeError, ValueError):
            H1 = _homography_from_kps(kps1)
    if H1 is None:
        H1 = _homography_from_kps(kps1)
    H2 = None
    if STEP4_3_ENABLED:
        src2, dst2 = _build_weighted_correspondences(kps2, weight_43=h2_weight_43, weight_41=1, use_right=use_right)
        if len(src2) >= 4:
            H2, _ = cv2.findHomography(np.array(src2, dtype=np.float32), np.array(dst2, dtype=np.float32))
    H3 = None
    if STEP4_1_ENABLED or STEP4_3_ENABLED:
        src3, dst3 = _build_weighted_correspondences(kps3, weight_43=h3_weight_43, weight_41=h3_weight_41, use_right=use_right, weight_idx16=h3h4_weight_16)
        if len(src3) >= 4:
            H3, _ = cv2.findHomography(np.array(src3, dtype=np.float32), np.array(dst3, dtype=np.float32))
    H4 = None
    if STEP4_1_ENABLED:
        src4, dst4 = _build_weighted_correspondences(kps4, weight_43=1, weight_41=h4_weight_41, use_right=use_right, weight_idx16=h3h4_weight_16)
        if len(src4) >= 4:
            H4, _ = cv2.findHomography(np.array(src4, dtype=np.float32), np.array(dst4, dtype=np.float32))
    s1, H1 = _retry_wide_line_for_variant('H1', H1, kps1)
    s2, H2 = _retry_wide_line_for_variant('H2', H2, kps2)
    s3, H3 = _retry_wide_line_for_variant('H3', H3, kps3)
    s4, H4 = _retry_wide_line_for_variant('H4', H4, kps4)
    best_score = max(s1, s2, s3, s4)
    if best_score <= 0.0:
        valid_indices = [i for i in range(min(len(kps1), n_tpl)) if kps1[i] and len(kps1[i]) >= 2 and (not (abs(float(kps1[i][0])) < 1e-06 and abs(float(kps1[i][1])) < 1e-06))]
        if valid_indices:
            smallest_y_idx = min(valid_indices, key=lambda i: (float(kps1[i][1]), float(kps1[i][0]), int(i)))
            base_x = float(kps1[smallest_y_idx][0])
            base_y = float(kps1[smallest_y_idx][1])
            for dx, dy, tag in [(0.0, -1.0, 'y-1'), (0.0, 1.0, 'y+1'), (-1.0, 0.0, 'x-1'), (1.0, 0.0, 'x+1')]:
                kps_try = [list(kp) if kp else [0.0, 0.0] for kp in kps1]
                kps_try[smallest_y_idx] = [base_x + dx, base_y + dy]
                H_try = _homography_from_kps(kps_try)
                if H_try is None:
                    continue
                score_try = _score_h(H_try)
                if score_try > 0.0:
                    out_kps_try = _project_and_valid_only(H_try)
                    if not STEP6_FILL_MISSING_ENABLED:
                        for i in range(min(len(out_kps_try), len(kps_try))):
                            w = kps_try[i]
                            if not w or len(w) < 2 or (abs(float(w[0])) < 1e-06 and abs(float(w[1])) < 1e-06):
                                out_kps_try[i] = [0.0, 0.0]
                    return (out_kps_try, float(score_try))
        return (None, 0.0)
    if H1 is None and H2 is None and (H3 is None) and (H4 is None):
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
    elif best_score == s4 and H4 is not None:
        best_H = H4
        best_label = 'H4'
    if best_H is None:
        return (None, 0.0)
    out_kps = _project_and_valid_only(best_H)
    if not STEP6_FILL_MISSING_ENABLED:
        winning_kps = kps1 if best_label == 'H1' else kps2 if best_label == 'H2' else kps3 if best_label == 'H3' else kps4
        for i in range(min(len(out_kps), len(winning_kps))):
            w = winning_kps[i]
            if not w or len(w) < 2 or (abs(float(w[0])) < 1e-06 and abs(float(w[1])) < 1e-06):
                out_kps[i] = [0.0, 0.0]
    return (out_kps, best_score)

def _merge_step4_1_profile_into_parts(step4_1_result: dict[str, Any] | None, parts_ms: dict[str, float] | None) -> None:
    if not parts_ms or not step4_1_result:
        return
    prof = step4_1_result.get('profile') or {}
    total = prof.get('total_ms')
    if total is not None:
        parts_ms['step4_1'] = float(total)
    for k, v in prof.items():
        if k == 'total_ms':
            continue
        if isinstance(v, (int, float)):
            if k.endswith('_ms'):
                parts_ms[f'step4_1_{k[:-3]}'] = float(v)
            else:
                parts_ms[f'step4_1_{k}'] = float(v)
_STEP4_3_COUNT_KEYS = frozenset({'AB_coarse_candidates', 'AB_refine_candidates', 'AB_best_white_rate', 'CD_best_white_rate', 'CD_coarse_candidates', 'CD_refine_candidates'})

def _merge_step4_3_profile_into_parts(step4_3_result: dict[str, Any] | None, parts_ms: dict[str, float] | None) -> None:
    if not parts_ms or not step4_3_result:
        return
    prof = step4_3_result.get('profile') or {}
    total = prof.get('total_ms')
    if total is not None:
        parts_ms['step4_3'] = float(total)
    for k, v in prof.items():
        if k == 'total_ms':
            continue
        if isinstance(v, (int, float)):
            if k in _STEP4_3_COUNT_KEYS:
                parts_ms[f'step4_3_{k}_n'] = float(v)
            elif k.endswith('_ms'):
                parts_ms[f'step4_3_{k[:-3]}'] = float(v)
            else:
                parts_ms[f'step4_3_{k}'] = float(v)

def _step4_3_debug_dilate_and_lines(*, frame: np.ndarray, ordered_kps: list[list[float]], frame_number: int, decision: str | None=None, step4_1: dict[str, Any] | None=None, cached_edges: np.ndarray | None=None) -> dict[str, Any]:
    result: dict[str, Any] = {'frame_number': int(frame_number), 'did_line_draw': False, 'kkp4': None, 'kkp12': None, 'kkp28': None, 'kkp20': None, 'profile': {}}
    t_start = time.perf_counter()
    try:
        H_img, W_img = frame.shape[:2]
        template_len = len(FOOTBALL_KEYPOINTS_CORRECTED)
        if not ordered_kps or len(ordered_kps) < template_len:
            result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
            return result
        use_right = decision == 'right'
        bottom_idx = 20 if use_right else 12
        idx_A = 28 if use_right else 4
        idx_C = 17 if use_right else 9
        idx_top = 25 if use_right else 1
        side_label = 'right' if use_right else 'left'
        max_idx_required = 28 if use_right else 12
        if len(ordered_kps) <= max_idx_required:
            result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
            return result
        t_ = time.perf_counter()
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
        result['profile']['edges_dilate_ms'] = (time.perf_counter() - t_) * 1000.0
        t_ = time.perf_counter()
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
            result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
            return result
        H_mat, _ = cv2.findHomography(np.array(filtered_src, dtype=np.float32), np.array(filtered_dst, dtype=np.float32))
        if H_mat is None:
            result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
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
            result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
            return result
        x_A, y_A = (pt_A[0], pt_A[1])
        x_bottom, y_bottom = (pt_bottom[0], pt_bottom[1])
        x_C, y_C = (pt_C[0], pt_C[1])
        pt_top = _red_line_pos(idx_top)
        result['profile']['homography_red_ms'] = (time.perf_counter() - t_) * 1000.0
        t_ = time.perf_counter()
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
        total_candidates = coarse_candidates + (a_ref_max - a_ref_min + 1) * (b_ref_max - b_ref_min + 1)
        result['profile']['AB_search_ms'] = (time.perf_counter() - t_) * 1000.0
        result['profile']['AB_coarse_candidates'] = coarse_candidates
        result['profile']['AB_refine_candidates'] = (a_ref_max - a_ref_min + 1) * (b_ref_max - b_ref_min + 1)
        result['profile']['AB_best_white_rate'] = best_rate
        t_ = time.perf_counter()
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
        result['profile']['CD_search_ms'] = (time.perf_counter() - t_) * 1000.0
        result['profile']['CD_coarse_candidates'] = cd_coarse_candidates
        result['profile']['CD_refine_candidates'] = cd_refine_candidates
        result['profile']['CD_best_white_rate'] = best_cd_rate

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
            t_ = time.perf_counter()
            x_E_int, y_E_int = (int(round(x_E)), int(round(y_E)))
            x_F_int, y_F_int = (int(round(x_F)), int(round(y_F)))
            best_ef_rate = -1.0
            ef_reused = False
            best_ef_a = (x_E_int, y_E_int)
            best_ef_b = (x_F_int, y_F_int)
            ef_key_a = 'ef_right_a' if use_right else 'ef_left_a'
            ef_key_b = 'ef_right_b' if use_right else 'ef_left_b'
            if step4_1 and step4_1.get(ef_key_a) is not None and (step4_1.get(ef_key_b) is not None):
                try:
                    ea = step4_1[ef_key_a]
                    eb = step4_1[ef_key_b]
                    if ea and eb and (len(ea) >= 2) and (len(eb) >= 2):
                        best_ef_a = (int(round(float(ea[0]))), int(round(float(ea[1]))))
                        best_ef_b = (int(round(float(eb[0]))), int(round(float(eb[1]))))
                        ef_reused = True
                except Exception:
                    ef_reused = False
            if not ef_reused:
                ef_a_y_min = y_E_int - 30
                ef_a_y_max = y_E_int + 30
                ef_b_y_min = y_F_int - 30
                ef_b_y_max = y_F_int + 30
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
            result['profile']['EF_search_ms'] = (time.perf_counter() - t_) * 1000.0
            result['profile']['EF_best_white_rate'] = best_ef_rate
            result['profile']['EF_reused_from_step4_1'] = 1.0 if ef_reused else 0.0
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
        result['profile']['total_ms'] = (time.perf_counter() - t_start) * 1000.0
    except Exception as e:
        logger.error('Step 4.3 failed for frame %s: %s', frame_number, e)
        result['error'] = str(e)
    return result

def _save_four_points_visualization(frame: np.ndarray, four_points: list[int], keypoints: list[list[float]], connections: list[list[int]], frame_number: int, labels: list[int] | None=None) -> None:
    return
    try:
        vis_frame = frame.copy()
        selected_indices = set(four_points[:4]) if len(four_points) >= 4 else set()
        if connections:
            for conn in connections:
                if len(conn) >= 2:
                    idx1, idx2 = (int(conn[0]), int(conn[1]))
                    if 0 <= idx1 < len(keypoints) and 0 <= idx2 < len(keypoints):
                        kp1 = keypoints[idx1]
                        kp2 = keypoints[idx2]
                        if kp1 and len(kp1) >= 2 and (not (abs(kp1[0]) < 1e-06 and abs(kp1[1]) < 1e-06)) and kp2 and (len(kp2) >= 2) and (not (abs(kp2[0]) < 1e-06 and abs(kp2[1]) < 1e-06)):
                            x1, y1 = (int(float(kp1[0])), int(float(kp1[1])))
                            x2, y2 = (int(float(kp2[0])), int(float(kp2[1])))
                            cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_offset_x = 8
        text_offset_y = -10
        for idx, kp in enumerate(keypoints):
            if kp and len(kp) >= 2:
                x, y = (float(kp[0]), float(kp[1]))
                if not (abs(x) < 1e-06 and abs(y) < 1e-06):
                    x_int, y_int = (int(x), int(y))
                    if idx in selected_indices:
                        cv2.circle(vis_frame, (x_int, y_int), 4, (0, 0, 255), -1)
                    else:
                        cv2.circle(vis_frame, (x_int, y_int), 3, (0, 0, 255), -1)
                    text_num = str(idx)
                    (text_width, text_height), baseline = cv2.getTextSize(text_num, font, font_scale, font_thickness)
                    text_x = x_int + text_offset_x
                    text_y = y_int + text_offset_y
                    cv2.putText(vis_frame, text_num, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)
                    if labels is not None and 0 <= idx < len(labels):
                        label_val = labels[idx]
                        if label_val is not None:
                            text_label = f'L{int(label_val)}'
                            label_y = text_y + text_height + 2
                            cv2.putText(vis_frame, text_label, (text_x, label_y), font, font_scale, (0, 0, 255), font_thickness)
        out_dir = Path('debug_frames')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'frame_{frame_number:05d}_four_points.png'
        cv2.imwrite(str(out_path), vis_frame)
        _save_ordered_step_image(2, 'four_points_selection', vis_frame, frame_number)
    except Exception as exc:
        logger.error('Failed to save four points visualization for frame %s: %s', frame_number, exc)

def _save_original_keypoints_debug(frame: np.ndarray, keypoints: list[list[float]] | list[Any], labels: list[Any] | None, frame_number: int) -> None:
    return
    try:
        vis_frame = frame.copy()
        H_img, W_img = vis_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        for idx, kp in enumerate(keypoints or []):
            if not kp or len(kp) < 2:
                continue
            x, y = (float(kp[0]), float(kp[1]))
            if abs(x) < 1e-06 and abs(y) < 1e-06:
                continue
            x_int, y_int = (int(round(x)), int(round(y)))
            if not (0 <= x_int < W_img and 0 <= y_int < H_img):
                continue
            cv2.circle(vis_frame, (x_int, y_int), 4, (0, 0, 255), -1)
            label_val = None
            if labels is not None and 0 <= idx < len(labels):
                label_val = labels[idx]
            label_txt = str(label_val) if label_val is not None else 'NA'
            text = f'id:{idx} L:{label_txt}'
            cv2.putText(vis_frame, text, (x_int + 6, y_int - 4), font, font_scale, (0, 0, 255), font_thickness)
        out_dir = Path('debug_frames') / 'original_keypoints'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'frame_{int(frame_number):05d}_original_keypoints.png'
        cv2.imwrite(str(out_path), vis_frame)
        _save_ordered_step_image(1, 'original_keypoints', vis_frame, frame_number)
    except Exception as exc:
        logger.error('Failed to save original keypoints debug image for frame %s: %s', frame_number, exc)

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

def evaluate_keypoints_for_frame(template_keypoints: list[tuple[int, int]], frame_keypoints: list[tuple[int, int]], frame: np.ndarray, floor_markings_template: np.ndarray, frame_number: int | None=None, *, log_frame_number: bool=False, debug_dir: Path | None=None, cached_edges: np.ndarray | None=None, mask_polygons: dict[str, list[list[tuple[float, float]]]] | None=None, mask_debug_label: str | None=None, mask_debug_dir: Path | None=None, score_only: bool=False, log_context: dict | None=None) -> float:
    try:
        warped_template = None
        mask_ground_bin = None
        mask_lines_expected = None
        homography_matrix = None
        _score_profile = bool(TV_KP_PROFILE)
        score_only = bool(score_only)
        source = 'template-projected' if mask_polygons is None else 'file-polygon' if ADD4_POLYGON_USE_JSON else 'incode-polygon'
        eval_debug_dir: Path | None = mask_debug_dir if mask_debug_dir is not None else None

        t_total_start = time.perf_counter() if _score_profile else 0.0
        t_validation_end: float | None = None
        t_score_start: float | None = None
        t_score_bbox = 0.0
        t_score_kp = 0.0
        t_score_outside = 0.0
        t_score_overlap = 0.0
        t_score_vis = 0.0
        t_val_clamp = 0.0
        t_val_blacklist = 0.0
        t_val_precheck = 0.0
        t_val_project = 0.0
        t_val_masks = 0.0
        t_val_masks_extract = 0.0
        t_val_pred = 0.0
        use_cache = True

        def _log_score_profile(status: str) -> None:
            if not _score_profile:
                return
            t_end = time.perf_counter()
            total_ms = (t_end - t_total_start) * 1000.0
            validation_ms = (t_validation_end - t_total_start) * 1000.0 if t_validation_end is not None else 0.0
            score_only_ms = (t_end - t_score_start) * 1000.0 if t_score_start is not None else 0.0
            print('[tv][kp_score_profile] frame=%s status=%s total_ms=%.2f validation_ms=%.2f score_ms=%.2f diff_ms=%.2f' % (str(frame_number), str(status), float(total_ms), float(validation_ms), float(score_only_ms), float(score_only_ms)))
            print('[tv][kp_score_profile] validation clamp_ms=%.2f blacklist_ms=%.2f precheck_ms=%.2f project_ms=%.2f masks_ms=%.2f pred_ms=%.2f' % (float(t_val_clamp * 1000.0), float(t_val_blacklist * 1000.0), float(t_val_precheck * 1000.0), float(t_val_project * 1000.0), float(t_val_masks * 1000.0), float(t_val_pred * 1000.0)))
            print('[tv][kp_score_profile] masks breakdown extract_ms=%.2f' % (float(t_val_masks_extract * 1000.0),))
            print('[tv][kp_score_profile] score overlap_ms=%.2f bbox_ms=%.2f kp_ms=%.2f outside_ms=%.2f vis_ms=%.2f' % (float(t_score_overlap * 1000.0), float(t_score_bbox * 1000.0), float(t_score_kp * 1000.0), float(t_score_outside * 1000.0), float(t_score_vis * 1000.0)))

        def _write_mask_debug_image(value: float, status: str) -> None:
            if mask_debug_dir is None:
                return
            if mask_lines_expected is None:
                return
            try:
                edges_vis = cached_edges
                if edges_vis is None:
                    edges_vis = compute_frame_canny_edges(frame)
                base = cv2.cvtColor(edges_vis, cv2.COLOR_GRAY2BGR)
                line_mask = (mask_lines_expected > 0).astype(np.uint8)
                if line_mask.any():
                    overlay = base.copy()
                    overlay[line_mask > 0] = (0, 0, 255)
                    alpha = 0.5
                    base = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)
                text = f'score: {value:.4f}' if value > 0.0 else str(status)
                cv2.putText(base, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
                label = mask_debug_label or 'mask'
                frame_tag = 'na' if frame_number is None else f'{int(frame_number):03d}'
                mask_debug_dir.mkdir(parents=True, exist_ok=True)
                out_path = mask_debug_dir / f'frame_{frame_tag}_mask_{label}.png'
                cv2.imwrite(str(out_path), base)
            except Exception:
                pass

        def _write_eval_debug_images(status: str, **kwargs: Any) -> None:
            if eval_debug_dir is None:
                return
            try:
                frame_tag = 'na' if frame_number is None else f'{int(frame_number):03d}'
                tf_idx = log_context.get('transform_idx', -1) if log_context else -1
                pair_str = ','.join((str(i) for i in log_context.get('pair_indices', []) or []))
                suffix = f'_t{tf_idx}_pair_{pair_str}' if log_context and pair_str else ''
                base_name = f'frame_{frame_tag}{suffix}_{status}'
                eval_debug_dir.mkdir(parents=True, exist_ok=True)
                vis = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                for idx, (x, y) in enumerate(frame_keypoints):
                    if x == 0 and y == 0:
                        continue
                    ix, iy = (int(round(x)), int(round(y)))
                    if 0 <= ix < vis.shape[1] and 0 <= iy < vis.shape[0]:
                        color = (0, 0, 255)
                        r = 4
                        blacklist_tuple = kwargs.get('blacklist_tuple')
                        if blacklist_tuple and idx + 1 in blacklist_tuple:
                            r = 8
                        cv2.circle(vis, (ix, iy), r, color, 2)
                        cv2.putText(vis, str(idx + 1), (ix + 6, iy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                msg = status
                error_msg = kwargs.get('error_msg')
                if error_msg:
                    msg = f'{status}: {error_msg[:80]}'
                cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                out_path = eval_debug_dir / f'{base_name}.png'
                cv2.imwrite(str(out_path), vis)
                warped_template = kwargs.get('warped_template')
                if warped_template is not None and getattr(warped_template, 'shape', None):
                    warp_path = eval_debug_dir / f'frame_{frame_tag}{suffix}_warped.png'
                    cv2.imwrite(str(warp_path), warped_template)
            except Exception:
                pass

        def _score_profile_return(value: float, status: str, **kwargs: Any) -> float:
            nonlocal t_validation_end, t_score_start
            if _score_profile and t_validation_end is None:
                t_validation_end = time.perf_counter()
            if _score_profile and t_score_start is None:
                t_score_start = t_validation_end
            _log_score_profile(status)
            _write_mask_debug_image(value, status)
            if value == 0.0 and eval_debug_dir is not None:
                _write_eval_debug_images(status, **kwargs)
            return value
        frame_id_str = f'Frame {frame_number}' if frame_number is not None else 'Frame'
        frame_height, frame_width = frame.shape[:2]
        t_step = time.perf_counter() if _score_profile else 0.0
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
        if _score_profile:
            t_val_clamp += time.perf_counter() - t_step
        if not score_only:
            t_step = time.perf_counter() if _score_profile else 0.0
            non_idx_set = {idx + 1 for idx, kpts in enumerate(frame_keypoints) if not (kpts[0] == 0 and kpts[1] == 0)}
            for blacklist in BLACKLISTS:
                if non_idx_set.issubset(blacklist):
                    if _both_points_same_direction(frame_keypoints[blacklist[0] - 1], frame_keypoints[blacklist[1] - 1], frame_width, frame_height):
                        if log_context is not None:
                            tf_idx = log_context.get('transform_idx', -1)
                            pair_str = ','.join((str(i) for i in log_context.get('pair_indices', [])))
                            kp_a = frame_keypoints[blacklist[0] - 1]
                            kp_b = frame_keypoints[blacklist[1] - 1]
                            print(f'  [blacklist] transform[{tf_idx}] pair [{pair_str}]: matched blacklist {tuple(blacklist)}, same-direction check on keypoints (1-based) {blacklist[0]}={tuple(kp_a)}, {blacklist[1]}={tuple(kp_b)}')
                        return _score_profile_return(0.0, 'blacklist', blacklist_tuple=tuple(blacklist))
            if _score_profile:
                t_val_blacklist += time.perf_counter() - t_step
        cache_key: tuple = (int(frame_number) if frame_number is not None else -1, int(frame_width), int(frame_height), tuple(((float(x), float(y)) for x, y in frame_keypoints)))
        if use_cache and cache_key in _KP_SCORE_CACHE:
            cached_score = float(_KP_SCORE_CACHE[cache_key])
            _KP_SCORE_CACHE.move_to_end(cache_key)
            if _score_profile:
                t_validation_end = time.perf_counter()
                _log_score_profile('cache')
            return cached_score
        t_step = time.perf_counter() if _score_profile else 0.0
        valid_keypoints = [(x, y) for x, y in frame_keypoints if not (x == 0 and y == 0)]
        if len(valid_keypoints) < 4:
            return _score_profile_return(0.0, 'insufficient_kp')
        if not score_only:
            xs, ys = zip(*valid_keypoints)
            min_x_kp, max_x_kp = (min(xs), max(xs))
            min_y_kp, max_y_kp = (min(ys), max(ys))
            if max_x_kp < 0 or max_y_kp < 0 or min_x_kp >= frame_width or (min_y_kp >= frame_height):
                return _score_profile_return(0.0, 'kp_outside')
            if max_x_kp - min_x_kp > 2 * frame_width or max_y_kp - min_y_kp > 2 * frame_height:
                return _score_profile_return(0.0, 'kp_spread')
            if _score_profile:
                t_val_precheck += time.perf_counter() - t_step
        cached_warp = _KP_WARP_CACHE.get(cache_key) if use_cache else None
        if cached_warp is not None:
            _KP_WARP_CACHE.move_to_end(cache_key)
            warped_template, mask_ground_bin, mask_lines_expected = cached_warp
        if warped_template is None and homography_matrix is None:
            t_step = time.perf_counter() if _score_profile else 0.0
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
                    else:
                        warped_template, homography_matrix = project_image_using_keypoints(image=floor_markings_template, source_keypoints=template_keypoints, destination_keypoints=frame_keypoints, destination_width=frame_width, destination_height=frame_height, return_h=True)
            except ValueError as e:
                pass
                return _score_profile_return(0.0, 'proj_fail')
            except InvalidMask as e:
                pass
                return _score_profile_return(0.0, 'proj_invalid')
            if _score_profile:
                t_val_project += time.perf_counter() - t_step
        if mask_ground_bin is None or mask_lines_expected is None:
            t_step = time.perf_counter() if _score_profile else 0.0
            try:
                if mask_polygons is not None:
                    if homography_matrix is None:
                        raise ValueError('Homography not available for polygon masks')
                    mask_ground_bin, mask_lines_expected = _polygon_masks_from_homography(homography_matrix, (frame_height, frame_width), mask_polygons, debug_frame_id=frame_number)
                else:
                    if warped_template is None:
                        raise ValueError('Non-polygon path requires warped_template from project_image_using_keypoints')
                    _t = time.perf_counter() if _score_profile else 0.0
                    mask_ground_bin, mask_lines_expected = extract_masks_for_ground_and_lines(warped_template, debug_frame_id=frame_number)
                    if _score_profile:
                        t_val_masks_extract += time.perf_counter() - _t
            except InvalidMask as e:
                pass
                return _score_profile_return(0.0, 'mask_fail', error_msg=str(e), warped_template=warped_template)
            if use_cache:
                _KP_WARP_CACHE[cache_key] = (warped_template, mask_ground_bin, mask_lines_expected)
                _KP_WARP_CACHE.move_to_end(cache_key)
                if len(_KP_WARP_CACHE) > _KP_CACHE_MAX:
                    _KP_WARP_CACHE.popitem(last=False)
            if _score_profile:
                t_val_masks += time.perf_counter() - t_step
        t_step = time.perf_counter() if _score_profile else 0.0
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
        if _score_profile:
            t_val_pred += time.perf_counter() - t_step
        t_step = time.perf_counter() if _score_profile else 0.0
        pixels_overlapping_result = cv2.bitwise_and(mask_lines_expected, mask_lines_predicted)
        if _score_profile:
            t_score_overlap += time.perf_counter() - t_step
        if not score_only:
            t_step = time.perf_counter() if _score_profile else 0.0
            pts = cv2.findNonZero(mask_lines_expected)
            if pts is None or len(pts) == 0:
                return _score_profile_return(0.0, 'no_expected')
            else:
                min_x, min_y, w_box, h_box = cv2.boundingRect(pts)
                max_x = int(min_x + w_box - 1)
                max_y = int(min_y + h_box - 1)
                bbox = (int(min_x), int(min_y), int(max_x), int(max_y))
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = frame_height * frame_width
            if bbox_area / frame_area < 0.2:
                return _score_profile_return(0.0, 'bbox_small')
            if _score_profile:
                t_score_bbox += time.perf_counter() - t_step
            t_step = time.perf_counter() if _score_profile else 0.0
            valid_keypoints = [(x, y) for x, y in frame_keypoints if not (x == 0 and y == 0)]
            if not valid_keypoints:
                return _score_profile_return(0.0, 'no_valid_kp')
            xs, ys = zip(*valid_keypoints)
            min_x_kp, max_x_kp = (min(xs), max(xs))
            min_y_kp, max_y_kp = (min(ys), max(ys))
            if max_x_kp < 0 or max_y_kp < 0 or min_x_kp >= frame_width or (min_y_kp >= frame_height):
                return _score_profile_return(0.0, 'kp_outside2')
            if max_x_kp - min_x_kp > 2 * frame_width or max_y_kp - min_y_kp > 2 * frame_height:
                return _score_profile_return(0.0, 'kp_spread2')
            if _score_profile:
                t_score_kp += time.perf_counter() - t_step
            t_step = time.perf_counter() if _score_profile else 0.0
            inv_expected = cv2.bitwise_not(mask_lines_expected)
            pixels_rest = cv2.bitwise_and(inv_expected, mask_lines_predicted).sum()
            total_pixels = cv2.bitwise_or(mask_lines_expected, mask_lines_predicted).sum()
            if total_pixels == 0:
                return _score_profile_return(0.0, 'total_zero')
            if pixels_rest / total_pixels > 0.9:
                return _score_profile_return(0.0, 'outside_ratio')
            if _score_profile:
                t_score_outside += time.perf_counter() - t_step
        if _score_profile and t_validation_end is None:
            t_validation_end = time.perf_counter()
            t_score_start = t_validation_end
        t_step = time.perf_counter() if _score_profile else 0.0
        pixels_overlapping = pixels_overlapping_result.sum()
        pixels_on_lines = mask_lines_expected.sum()
        overlap_ratio = float(pixels_overlapping) / float(pixels_on_lines + 1e-08)
        if _score_profile:
            t_score_overlap += time.perf_counter() - t_step
        t_step = time.perf_counter() if _score_profile else 0.0
        if _score_profile:
            t_score_vis += time.perf_counter() - t_step
        logger.info('[evaluate_keypoints_for_frame] frame=%s overlap=%d expected_lines=%d ratio=%.6f', frame_number, int(pixels_overlapping), int(pixels_on_lines), overlap_ratio)
        if use_cache:
            _KP_SCORE_CACHE[cache_key] = float(overlap_ratio)
            _KP_SCORE_CACHE.move_to_end(cache_key)
            if len(_KP_SCORE_CACHE) > _KP_CACHE_MAX:
                _KP_SCORE_CACHE.popitem(last=False)
        _log_score_profile('ok')
        _write_mask_debug_image(float(overlap_ratio), 'ok')
        return overlap_ratio
    except Exception as e:
        pass
        pass
    return _score_profile_return(0.0, 'exception')

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
    filtered_edges_after_triangle = {e for e in pair_to_len.keys() if e not in edges_to_remove}
    adj_after_triangle: dict[int, set[int]] = {}
    for a, b in filtered_edges_after_triangle:
        adj_after_triangle.setdefault(int(a), set()).add(int(b))
        adj_after_triangle.setdefault(int(b), set()).add(int(a))

    def _label_at(idx: int) -> int:
        try:
            if 0 <= int(idx) < len(labels):
                return int(labels[int(idx)] or 0)
        except Exception:
            pass
        return 0

    def _has_other_label_neighbor(node: int, other: int, target_label: int) -> bool:
        for nb in adj_after_triangle.get(int(node), set()):
            if int(nb) == int(other):
                continue
            if _label_at(int(nb)) == int(target_label):
                return True
        return False
    bridge_edges_to_remove: set[tuple[int, int]] = set()
    for a, b in filtered_edges_after_triangle:
        la = _label_at(int(a))
        lb = _label_at(int(b))
        if {la, lb} != {2, 3}:
            continue
        a_has_l2 = _has_other_label_neighbor(int(a), int(b), 2)
        a_has_l3 = _has_other_label_neighbor(int(a), int(b), 3)
        b_has_l2 = _has_other_label_neighbor(int(b), int(a), 2)
        b_has_l3 = _has_other_label_neighbor(int(b), int(a), 3)
        if a_has_l2 and a_has_l3 and b_has_l2 and b_has_l3:
            bridge_edges_to_remove.add(tuple(sorted((int(a), int(b)))))
    all_edges_to_remove = edges_to_remove | bridge_edges_to_remove
    filtered_details = [(i, j, hr, lg) for i, j, hr, lg in good_details if tuple(sorted((i, j))) not in all_edges_to_remove]
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
    do_prof = _kp_prof_enabled()
    t_pat = t_dedupe = t_allowed = t_label = t_connlabel = t_conn = t_y = 0.0
    t_pat_expand = t_pat_collect = 0.0
    t_label_cy = t_connlabel_cy = t_conn_cy = 0.0
    n_quads = 0
    n_cand_all = n_cand_unique = 0
    n_after_allowed = n_after_label = n_after_connlabel = n_after_conn = n_after_y = 0
    max_cand_unique = max_cand_final = 0
    n_label_cy = n_connlabel_cy = n_conn_cy = 0
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
                candidates_before_cy_connlabel = list(candidates)
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
                if len(candidates) == 0 and len(candidates_before_cy_connlabel) > 0:
                    py_filtered = [cand for cand in candidates_before_cy_connlabel if _check_connection_label_constraints(list(cand) if isinstance(cand, tuple) else cand, quad)]
                    if py_filtered:
                        candidates = py_filtered
                        cand_arr = None
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
        if do_prof:
            n_quads = len(frame_four_points)
    else:
        for quad in frame_four_points:
            n_quads += 1
            if do_prof:
                t0 = time.perf_counter()
            entry = _process_one_quad(quad)
            if do_prof:
                t_pat += (time.perf_counter() - t0) * 1000.0
            mapped.append(entry)
            if do_prof:
                n_cand_all += entry.get('candidates_count', 0)
                max_cand_final = max(max_cand_final, entry.get('candidates_count', 0))
    out = {'frame_id': step2_entry['frame_id'], 'connections': list(frame_connections), 'keypoints': frame_keypoints, 'frame_size': frame_size, 'labels': frame_labels, 'decision': decision, 'matches': mapped}
    if do_prof:
        out['step3_profile'] = {'step3_pat': float(t_pat), 'step3_pat_expand': float(t_pat_expand), 'step3_pat_collect': float(t_pat_collect), 'step3_dedupe': float(t_dedupe), 'step3_allowed': float(t_allowed), 'step3_label': float(t_label), 'step3_label_cy': float(t_label_cy), 'step3_connlabel': float(t_connlabel), 'step3_connlabel_cy': float(t_connlabel_cy), 'step3_conn': float(t_conn), 'step3_conn_cy': float(t_conn_cy), 'step3_y': float(t_y), 'step3_quads_n': float(n_quads), 'step3_cand_all_n': float(n_cand_all), 'step3_cand_unique_n': float(n_cand_unique), 'step3_after_allowed_n': float(n_after_allowed), 'step3_after_label_n': float(n_after_label), 'step3_after_connlabel_n': float(n_after_connlabel), 'step3_after_conn_n': float(n_after_conn), 'step3_after_y_n': float(n_after_y), 'step3_max_unique_n': float(max_cand_unique), 'step3_max_final_n': float(max_cand_final), 'step3_label_cy_n': float(n_label_cy), 'step3_connlabel_cy_n': float(n_connlabel_cy), 'step3_conn_cy_n': float(n_conn_cy)}
    return out

def _step4_pick_best_candidate(*, matches: list[dict[str, Any]], orig_kps: list[list[float]], frame_keypoints: list, frame_labels: list, decision: str | None, template_pts: np.ndarray, frame_number: int, frame: np.ndarray | None=None) -> tuple[float, dict[str, Any] | None, list[int] | None]:
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
    if best_meta and best_avg > 80:
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

def _dedupe_close_unordered_keypoints(*, kps: list[list[float]] | list[Any], labels: list[int] | None, frame_number: int, proximity_px: float=20.0) -> tuple[list[list[float]], list[int], list[dict[str, Any]]]:
    template_len = len(FOOTBALL_KEYPOINTS)
    kps_in = list(kps or [])
    kps_out: list[list[float]] = []
    for i in range(template_len):
        if i < len(kps_in) and kps_in[i] and (len(kps_in[i]) >= 2):
            kps_out.append([float(kps_in[i][0]), float(kps_in[i][1])])
        else:
            kps_out.append([0.0, 0.0])
    labels_in = list(labels or [])
    labels_out: list[int] = [int(labels_in[i]) if i < len(labels_in) and labels_in[i] is not None else 0 for i in range(template_len)]
    valid_indices = [i for i in range(template_len) if labels_out[i] > 0 and (not (abs(kps_out[i][0]) < 1e-06 and abs(kps_out[i][1]) < 1e-06))]
    if len(valid_indices) < 2:
        labeled_cur = [{'id': i, 'x': float(kps_out[i][0]), 'y': float(kps_out[i][1]), 'label': f'kpv{int(labels_out[i]):02d}'} for i in range(template_len) if labels_out[i] > 0 and (not (abs(kps_out[i][0]) < 1e-06 and abs(kps_out[i][1]) < 1e-06))]
        return (kps_out, labels_out, labeled_cur)
    neighbors: dict[int, set[int]] = {i: set() for i in valid_indices}
    for ii in range(len(valid_indices)):
        a = valid_indices[ii]
        ax, ay = (float(kps_out[a][0]), float(kps_out[a][1]))
        for jj in range(ii + 1, len(valid_indices)):
            b = valid_indices[jj]
            bx, by = (float(kps_out[b][0]), float(kps_out[b][1]))
            if float(np.hypot(ax - bx, ay - by)) <= float(proximity_px):
                neighbors[a].add(b)
                neighbors[b].add(a)
    components: list[list[int]] = []
    seen: set[int] = set()
    for idx in valid_indices:
        if idx in seen:
            continue
        stack = [idx]
        comp: list[int] = []
        seen.add(idx)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in neighbors.get(cur, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(comp) > 1:
            components.append(sorted(comp))
    if not components:
        labeled_cur = [{'id': i, 'x': float(kps_out[i][0]), 'y': float(kps_out[i][1]), 'label': f'kpv{int(labels_out[i]):02d}'} for i in range(template_len) if labels_out[i] > 0 and (not (abs(kps_out[i][0]) < 1e-06 and abs(kps_out[i][1]) < 1e-06))]
        return (kps_out, labels_out, labeled_cur)
    removed_count = 0
    for comp in components:
        for idx in comp:
            if not (abs(kps_out[idx][0]) < 1e-06 and abs(kps_out[idx][1]) < 1e-06):
                removed_count += 1
            kps_out[idx] = [0.0, 0.0]
            labels_out[idx] = 0
    labeled_cur = [{'id': i, 'x': float(kps_out[i][0]), 'y': float(kps_out[i][1]), 'label': f'kpv{int(labels_out[i]):02d}'} for i in range(template_len) if labels_out[i] > 0 and (not (abs(kps_out[i][0]) < 1e-06 and abs(kps_out[i][1]) < 1e-06))]
    return (kps_out, labels_out, labeled_cur)

def _try_process_similar_frame_fast_path(*, fn: int, progress_prefix: str, frame: np.ndarray, frame_store: Any, kps: list[list[float]] | list[Any], labels: list[int], labeled_cur: list[dict[str, Any]], template_len: int, ordered_frames: list[dict[str, Any]], step1_outputs: list[dict[str, Any]] | None, step2_outputs: list[dict[str, Any]] | None, step3_outputs: list[dict[str, Any]] | None, step5_outputs: list[dict[str, Any]] | None, best_entries: list[dict[str, Any]], original_keypoints: dict[int, list[list[float]]], template_image: np.ndarray, template_pts_corrected: np.ndarray, template_pts: np.ndarray, cached_edges: np.ndarray | None, prev_valid_labeled: list[dict[str, Any]] | None, prev_valid_original_index: list[int] | None, prev_best_meta: dict[str, Any] | None, prev_valid_frame_id: int | None, similarity_snapshot_history: dict[int, dict[str, Any]] | None, log_fn: Any, push_fn: Any) -> tuple[bool, dict[str, Any]]:
    H, W = frame.shape[:2]
    similar_frame = False
    similar_ref_frame_id: int | None = None
    matches: list[tuple[int, int, float]] = []
    common_matches: list[tuple[int, int, float]] = []
    labeled_ref: list[dict[str, Any]] | None = None
    best_meta_prev: dict[str, Any] | None = None
    if not PREV_RELATIVE_FLAG:
        pass
    else:
        ref_candidates: list[tuple[int, dict[str, Any]]] = []
        seen_ref_ids: set[int] = set()
        for delta in (1, 2, 3):
            ref_id = int(fn - delta)
            if ref_id < 0 or ref_id in seen_ref_ids:
                continue
            ref_meta: dict[str, Any] | None = None
            if prev_valid_frame_id == ref_id and prev_best_meta is not None:
                ref_meta = prev_best_meta
            elif similarity_snapshot_history is not None:
                ref_meta = similarity_snapshot_history.get(ref_id)
            if ref_meta is not None:
                ref_candidates.append((ref_id, ref_meta))
                seen_ref_ids.add(ref_id)
        for ref_id, ref_meta in ref_candidates:
            prev_ordered_kps = ref_meta.get('reordered_keypoints')
            if not (prev_ordered_kps and isinstance(prev_ordered_kps, list)):
                continue
            labeled_ref_try = _ordered_keypoints_to_labeled(prev_ordered_kps, template_len)
            if not labeled_ref_try:
                continue
            _avg_dist, _match_count, matches_try = _avg_distance_same_label(labeled_cur, labeled_ref_try)
            valid_cur = [idx_lab for idx_lab, item in enumerate(labeled_cur) if item.get('x') is not None and item.get('y') is not None and (item.get('label') is not None)]
            common_try = [(cur_idx, prev_idx, dist) for cur_idx, prev_idx, dist in matches_try if dist != float('inf')]
            if len(common_try) >= 3 and all((dist < 40.0 for _, _, dist in common_try)):
                similar_frame = True
                similar_ref_frame_id = int(ref_id)
                best_meta_prev = ref_meta
                labeled_ref = labeled_ref_try
                matches = matches_try
                common_matches = common_try
                break
    if not similar_frame:
        return (False, {})
    log_fn(f'{progress_prefix} - similar to frame {similar_ref_frame_id}, reusing order')
    ordered_kps = [[0.0, 0.0] for _ in range(template_len)]
    orig_idx_map_current = [-1 for _ in range(template_len)]
    fallback_used = False
    prev_ordered_keypoints = best_meta_prev.get('reordered_keypoints') if best_meta_prev else None
    mapped_count = 0
    skipped_count = 0
    estimated_missing_count = 0
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
        matched_slot_motion: list[tuple[int, float, float, float, float]] = []
        estimated_candidate_slots: list[int] = []
        estimated_candidate_coords: dict[int, list[float]] = {}
        unmatched_candidate_slots: list[int] = []
        unmatched_kept_count = 0
        slot_debug_source: dict[int, str] = {}
        pre_score_compare_kps: list[list[float]] | None = None
        pre_score_compare_source: dict[int, str] | None = None

        def _is_valid_kp(pt: Any) -> bool:
            return pt is not None and len(pt) >= 2 and (not (abs(float(pt[0])) < 1e-06 and abs(float(pt[1])) < 1e-06))

        def _score_for_ordered(kps_for_score: list[list[float]]) -> float:
            try:
                kp_tuples = [(float(pt[0]), float(pt[1])) if pt and len(pt) >= 2 else (0.0, 0.0) for pt in kps_for_score]
                return float(evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS_CORRECTED, frame_keypoints=kp_tuples, frame=frame, floor_markings_template=template_image, frame_number=int(fn), log_frame_number=False, cached_edges=cached_edges))
            except Exception:
                return 0.0
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
            slot_debug_source[int(target_slot)] = 'M'
            mapped_count += 1
            prev_slot_kp = prev_ordered_keypoints[target_slot]
            if prev_slot_kp and len(prev_slot_kp) >= 2:
                matched_slot_motion.append((int(target_slot), float(prev_slot_kp[0]), float(prev_slot_kp[1]), float(cur_coord[0]), float(cur_coord[1])))
        if matched_slot_motion and prev_ordered_keypoints is not None:
            for missing_slot in range(template_len):
                if not _is_valid_kp(prev_ordered_keypoints[missing_slot]):
                    continue
                if _is_valid_kp(ordered_kps[missing_slot]):
                    continue
                prev_missing = prev_ordered_keypoints[missing_slot]
                px_m = float(prev_missing[0])
                py_m = float(prev_missing[1])
                mov_x_num = 0.0
                mov_y_num = 0.0
                mov_den = 0.0
                for _, px_c_prev, py_c_prev, px_c_cur, py_c_cur in matched_slot_motion:
                    dist_xy = float(np.hypot(px_c_prev - px_m, py_c_prev - py_m))
                    if dist_xy <= 1e-06:
                        continue
                    inv_dist = 1.0 / dist_xy
                    mov_x_num += (px_c_cur - px_c_prev) / dist_xy
                    mov_y_num += (py_c_cur - py_c_prev) / dist_xy
                    mov_den += inv_dist
                if mov_den <= 1e-12:
                    continue
                movement_x = float(mov_x_num / mov_den)
                movement_y = float(mov_y_num / mov_den)
                est_x = px_m + movement_x
                est_y = py_m + movement_y
                if 0.0 <= est_x < float(W) and 0.0 <= est_y < float(H):
                    ordered_kps[missing_slot] = [est_x, est_y]
                    orig_idx_map_current[missing_slot] = -1
                    estimated_candidate_slots.append(int(missing_slot))
                    estimated_candidate_coords[int(missing_slot)] = [float(est_x), float(est_y)]
                    slot_debug_source[int(missing_slot)] = 'E'
        matched_src: list[tuple[float, float]] = []
        matched_dst: list[tuple[float, float]] = []
        h_support_slots: list[int] = []
        for slot in range(template_len):
            if not _is_valid_kp(ordered_kps[slot]):
                continue
            src_tag = slot_debug_source.get(int(slot))
            if src_tag not in {'M', 'E'}:
                continue
            matched_src.append((float(FOOTBALL_KEYPOINTS_CORRECTED[slot][0]), float(FOOTBALL_KEYPOINTS_CORRECTED[slot][1])))
            matched_dst.append((float(ordered_kps[slot][0]), float(ordered_kps[slot][1])))
            h_support_slots.append(int(slot))
        H_match = None
        projected_all: np.ndarray | None = None
        if len(matched_src) >= 4:
            H_match, _ = cv2.findHomography(np.array(matched_src, dtype=np.float32), np.array(matched_dst, dtype=np.float32))
            if H_match is not None:
                tpl_all = np.array([[float(p[0]), float(p[1])] for p in FOOTBALL_KEYPOINTS_CORRECTED], dtype=np.float32).reshape(-1, 1, 2)
                projected_all = cv2.perspectiveTransform(tpl_all, H_match).reshape(-1, 2)
        unmatched_cur_indices = [cur_idx for cur_idx, prev_idx, dist in matches or [] if prev_idx == -1 or dist == float('inf')]
        reserved_slots = {s for s in range(template_len) if _is_valid_kp(ordered_kps[s])}
        if projected_all is not None and unmatched_cur_indices:
            for cur_idx in unmatched_cur_indices:
                if cur_idx < 0 or cur_idx >= len(labeled_cur):
                    continue
                cur_item = labeled_cur[cur_idx]
                cur_x = cur_item.get('x')
                cur_y = cur_item.get('y')
                cur_label = _label_val(cur_item.get('label'))
                cur_orig_id = cur_item.get('id')
                if cur_x is None or cur_y is None or cur_label is None:
                    continue
                cur_coord = _coord_for_id(int(cur_orig_id)) if cur_orig_id is not None else None
                if cur_coord is None:
                    cur_coord = [float(cur_x), float(cur_y)]
                same_label_slots = [s for s in range(template_len) if KEYPOINT_LABELS[s] == cur_label and s not in reserved_slots]
                if not same_label_slots:
                    continue
                best_slot = None
                best_dist = float('inf')
                for s in same_label_slots:
                    px, py = (float(projected_all[s][0]), float(projected_all[s][1]))
                    d = float(np.hypot(cur_coord[0] - px, cur_coord[1] - py))
                    if d < best_dist:
                        best_dist = d
                        best_slot = int(s)
                if best_slot is None:
                    continue
                ordered_kps[best_slot] = [float(cur_coord[0]), float(cur_coord[1])]
                orig_idx_map_current[best_slot] = int(cur_orig_id) if cur_orig_id is not None else -1
                unmatched_candidate_slots.append(best_slot)
                reserved_slots.add(best_slot)
                slot_debug_source[int(best_slot)] = 'U'
        pre_score_compare_kps = [[float(pt[0]), float(pt[1])] if pt and len(pt) >= 2 else [0.0, 0.0] for pt in ordered_kps]
        pre_score_compare_source = dict(slot_debug_source)
        for slot in unmatched_candidate_slots:
            if slot < 0 or slot >= len(ordered_kps):
                continue
            kp_slot = ordered_kps[slot]
            if not _is_valid_kp(kp_slot):
                continue
            with_kp = [list(pt) if pt and len(pt) >= 2 else [0.0, 0.0] for pt in ordered_kps]
            without_kp = [list(pt) if pt and len(pt) >= 2 else [0.0, 0.0] for pt in ordered_kps]
            without_kp[slot] = [0.0, 0.0]
            score_with = _score_for_ordered(with_kp)
            score_without = _score_for_ordered(without_kp)
            if score_with >= score_without:
                unmatched_kept_count += 1
                slot_debug_source[int(slot)] = 'U'
            else:
                ordered_kps[slot] = [0.0, 0.0]
                orig_idx_map_current[slot] = -1
                slot_debug_source.pop(int(slot), None)
        if estimated_candidate_slots:
            for slot in estimated_candidate_slots:
                if 0 <= slot < len(ordered_kps):
                    ordered_kps[slot] = [0.0, 0.0]
                    orig_idx_map_current[slot] = -1
                    slot_debug_source.pop(int(slot), None)
            always_keep_e_slots = {13, 14, 15, 16}
            estimated_missing_count = 0
            remaining_candidate_slots: list[int] = []
            for slot in estimated_candidate_slots:
                if slot < 0 or slot >= len(ordered_kps):
                    continue
                est_coord = estimated_candidate_coords.get(int(slot))
                if not est_coord or len(est_coord) < 2:
                    continue
                if int(slot) in always_keep_e_slots:
                    ordered_kps[slot] = [float(est_coord[0]), float(est_coord[1])]
                    orig_idx_map_current[slot] = -1
                    slot_debug_source[int(slot)] = 'E'
                    estimated_missing_count += 1
                else:
                    remaining_candidate_slots.append(int(slot))
            baseline_kps = [list(pt) if pt and len(pt) >= 2 else [0.0, 0.0] for pt in ordered_kps]
            baseline_score = _score_for_ordered(baseline_kps)
            best_slot: int | None = None
            best_score = baseline_score
            for slot in remaining_candidate_slots:
                if slot < 0 or slot >= len(ordered_kps):
                    continue
                est_coord = estimated_candidate_coords.get(int(slot))
                if not est_coord or len(est_coord) < 2:
                    continue
                trial_kps = [list(pt) for pt in baseline_kps]
                trial_kps[slot] = [float(est_coord[0]), float(est_coord[1])]
                trial_score = _score_for_ordered(trial_kps)
                if trial_score > best_score:
                    best_score = trial_score
                    best_slot = int(slot)
            if best_slot is not None:
                best_coord = estimated_candidate_coords.get(best_slot)
                if best_coord and len(best_coord) >= 2:
                    ordered_kps[best_slot] = [float(best_coord[0]), float(best_coord[1])]
                    orig_idx_map_current[best_slot] = -1
                    slot_debug_source[int(best_slot)] = 'E'
                    estimated_missing_count += 1
    step4_0_ordered_kps = [[float(pt[0]), float(pt[1])] if pt and len(pt) >= 2 else [0.0, 0.0] for pt in ordered_kps]
    step1_entry = {'frame_id': int(fn), 'frame_size': [int(W), int(H)], 'keypoints': [None if pt is None else [float(x) for x in pt] for pt in kps or []], 'labels': labels, 'connections': []}
    step2_entry = {'frame_id': int(fn), 'frame_size': [int(W), int(H)], 'connections': [], 'keypoints': step1_entry['keypoints'], 'labels': labels, 'four_points': []}
    step3_entry = {'frame_id': int(fn), 'connections': [], 'keypoints': step1_entry['keypoints'], 'frame_size': [int(W), int(H)], 'labels': labels, 'decision': None, 'matches': []}
    prev_decision = best_meta_prev.get('decision') if best_meta_prev else None
    similarity_snapshot = {'frame_id': int(fn), 'reordered_keypoints': step4_0_ordered_kps, 'decision': prev_decision}
    best_meta: dict[str, Any] = {'frame_id': int(fn), 'match_idx': None, 'candidate_idx': None, 'candidate': [], 'avg_distance': 0.0, 'reordered_keypoints': ordered_kps, 'step4_0_reordered_keypoints': step4_0_ordered_kps, 'decision': prev_decision, 'similar_prev_frame': True, 'added_four_point': False}
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
    fallback_used = False
    if STEP4_1_ENABLED:
        best_meta['step4_1'] = _step4_1_compute_border_line(frame=frame, ordered_kps=ordered_kps, frame_number=int(fn))
    else:
        best_meta['step4_1'] = {}
    best_meta['step4_2'] = {}
    best_meta['reordered_keypoints'] = ordered_kps
    if STEP4_3_ENABLED:
        best_meta['step4_3'] = _step4_3_debug_dilate_and_lines(frame=frame, ordered_kps=best_meta['reordered_keypoints'], frame_number=int(fn), decision=best_meta.get('decision'), step4_1=best_meta.get('step4_1'), cached_edges=cached_edges)
    else:
        best_meta['step4_3'] = {}
    step4_9_kps, step4_9_score = _step4_9_select_h_and_keypoints(input_kps=best_meta['reordered_keypoints'], step4_1=best_meta.get('step4_1'), step4_3=best_meta.get('step4_3'), frame=frame, frame_number=int(fn), decision=best_meta.get('decision'), cached_edges=cached_edges)
    if step4_9_kps is not None:
        best_meta['reordered_keypoints'] = step4_9_kps
        best_meta['score'] = step4_9_score
        ordered_kps = step4_9_kps
    else:
        fallback_score = 0.0
        try:
            kp_tuples = [(float(kp[0]), float(kp[1])) if kp and len(kp) >= 2 else (0.0, 0.0) for kp in ordered_kps]
            scorer_style_template = _template_from_project_root()
            fallback_score = evaluate_keypoints_for_frame(template_keypoints=FOOTBALL_KEYPOINTS_CORRECTED, frame_keypoints=kp_tuples, frame=frame, floor_markings_template=scorer_style_template.copy(), frame_number=int(fn), log_frame_number=False, cached_edges=None)
            best_meta['score'] = fallback_score
        except Exception as e:
            pass
            best_meta['score'] = 0.0
    step4_9_passed = step4_9_kps is not None
    best_meta['validation_passed'] = step4_9_passed
    if not step4_9_passed:
        best_meta['validation_error'] = 'Step 4.9 did not produce keypoints'
    else:
        best_meta.pop('validation_error', None)
    if STEP6_ENABLED and STEP6_FILL_MISSING_ENABLED and step4_9_passed and (not fallback_used) and (best_meta.get('added_four_point') != True):
        ordered_kps = _step6_fill_keypoints_from_homography(ordered_kps=ordered_kps, frame=frame, frame_number=int(fn))
        best_meta['reordered_keypoints'] = ordered_kps
    similarity_score = float(best_meta.get('score') or 0.0)
    if similarity_score <= 0.0:
        log_fn(f'{progress_prefix} - similar score 0.0, retrying with self logic')
        return (False, {})
    found['keypoints'] = ordered_kps
    found.pop('original_index', None)
    found['added_four_point'] = best_meta.get('added_four_point', False)
    push_fn(step1_outputs, step1_entry)
    push_fn(step2_outputs, step2_entry)
    push_fn(step3_outputs, step3_entry)
    best_entries.append(best_meta)
    push_fn(step5_outputs, best_meta.copy())
    state_updates: dict[str, Any] = {}
    state_updates['prev_labeled'] = labeled_cur
    state_updates['prev_original_index'] = orig_idx_map_current
    state_updates['prev_valid_labeled'] = labeled_cur
    state_updates['prev_valid_original_index'] = orig_idx_map_current
    state_updates['prev_best_meta'] = similarity_snapshot
    state_updates['prev_valid_frame_id'] = fn
    log_fn(f'{progress_prefix} - done')
    return (True, state_updates)

def _step7_interpolate_problematic_frames(*, step4_9_outputs: list[dict[str, Any]] | None, ordered_frames: list[dict[str, Any]], template_len: int, out_step7: Path) -> None:
    if not STEP7_ENABLED:
        return
    if step4_9_outputs is None:
        return
    if len(step4_9_outputs) == 0:
        return
    step4_9_outputs_serializable_for_step7: list[dict[str, Any]] = []
    for entry in step4_9_outputs:
        entry_copy = entry.copy()
        avg_dist = entry_copy.get('avg_distance')
        if avg_dist is not None and (avg_dist == float('inf') or np.isinf(avg_dist)):
            entry_copy['avg_distance'] = 99999
        elif avg_dist is not None and isinstance(avg_dist, (int, float)):
            entry_copy['avg_distance'] = round(float(avg_dist), 2)
        score = entry_copy.get('score')
        if score is not None and isinstance(score, (int, float)):
            entry_copy['score'] = round(float(score), 2)
        step4_9_outputs_serializable_for_step7.append(entry_copy)
    if not step4_9_outputs_serializable_for_step7:
        return
    step7_entries: list[dict[str, Any]] = []
    frame_map: dict[int, dict[str, Any]] = {}
    for entry in step4_9_outputs_serializable_for_step7:
        frame_id = entry.get('frame_id')
        if frame_id is not None:
            frame_map[int(frame_id)] = entry
    sorted_frame_ids = sorted(frame_map.keys())
    problematic_frames: list[int] = []
    for frame_id in sorted_frame_ids:
        entry = frame_map[frame_id]
        score = entry.get('score')
        if score is not None and score < 0.3:
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
                if score is not None and score >= 0.3:
                    forward_good_frame = frame_id
                    break
        backward_good_frame = None
        for frame_id in reversed(sorted_frame_ids):
            if frame_id < problem_frame_id:
                entry = frame_map[frame_id]
                score = entry.get('score')
                if score is not None and score >= 0.3:
                    backward_good_frame = frame_id
                    break
        if forward_good_frame is None or backward_good_frame is None:
            continue
        frame_distance = forward_good_frame - backward_good_frame
        if frame_distance > 30:
            continue
        forward_entry = frame_map[forward_good_frame]
        backward_entry = frame_map[backward_good_frame]
        forward_kps = forward_entry.get('reordered_keypoints', [])
        backward_kps = backward_entry.get('reordered_keypoints', [])
        forward_valid = count_valid_keypoints(forward_kps)
        backward_valid = count_valid_keypoints(backward_kps)
        if not (forward_valid >= 4 and backward_valid >= 4):
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
            interpolated_kps = step7_keypoints_map[fid_int]
            fr['keypoints'] = interpolated_kps
            updated_count += 1

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
def _template_from_project_root() -> np.ndarray:
    template_path = Path(__file__).resolve().parent.parent / 'football_pitch_template.png'
    if not template_path.exists():
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    img = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return np.zeros((720, 1280, 3), dtype=np.uint8)
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
        border_points_for_debug: list[tuple[int, int, float, float]] = []
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
    step7_outputs: list[dict[str, Any]] | None = [] if STORE_INTERMEDIATE else None
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
    similarity_snapshot_history: dict[int, dict[str, Any]] = {}
    step8_edges_cache = OrderedDict() if KEYPOINT_H_CONVERT_FLAG and STEP8_EDGE_CACHE_MAX > 0 else None
    prev_step4_1_seed: dict[str, int] | None = None
    try:
        prof = _KPProfiler(enabled=_kp_prof_enabled())
        total_frames = len(frame_ids)
        for idx, fn in enumerate(frame_ids):
            progress_prefix = f'Frame {idx + 1}/{total_frames} -'
            _log(f'{progress_prefix} - loading')
            parts_ms: dict[str, float] = {}
            t_frame0 = prof.begin_frame()
            t_other_load = time.perf_counter()
            frame = frame_store.get_frame(fn)
            H, W = frame.shape[:2]
            cached_edges = compute_frame_canny_edges(frame)
            if step8_edges_cache is not None:
                _step8_edges_cache_put(step8_edges_cache, int(fn), cached_edges, STEP8_EDGE_CACHE_MAX)
            parts_ms['other_load'] = (time.perf_counter() - t_other_load) * 1000.0
            t_other_setup = time.perf_counter()
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
            kps, labels, labeled_cur = _dedupe_close_unordered_keypoints(kps=kps, labels=labels, frame_number=int(fn), proximity_px=20.0)
            parts_ms['other_setup'] = (time.perf_counter() - t_other_setup) * 1000.0
            t_other_similar = time.perf_counter()
            handled, state_updates = _try_process_similar_frame_fast_path(fn=int(fn), progress_prefix=progress_prefix, frame=frame, frame_store=frame_store, kps=kps, labels=labels, labeled_cur=labeled_cur, template_len=template_len, ordered_frames=ordered_frames, step1_outputs=step1_outputs, step2_outputs=step2_outputs, step3_outputs=step3_outputs, step5_outputs=step5_outputs, best_entries=best_entries, original_keypoints=original_keypoints, template_image=template_image, template_pts_corrected=template_pts_corrected, template_pts=template_pts, cached_edges=cached_edges, prev_valid_labeled=prev_valid_labeled, prev_valid_original_index=prev_valid_original_index, prev_best_meta=prev_best_meta, prev_valid_frame_id=prev_valid_frame_id, similarity_snapshot_history=similarity_snapshot_history, log_fn=_log, push_fn=_push)
            parts_ms['other_similar'] = (time.perf_counter() - t_other_similar) * 1000.0
            if handled:
                if state_updates:
                    prev_labeled = state_updates.get('prev_labeled')
                    prev_original_index = state_updates.get('prev_original_index')
                    prev_valid_labeled = state_updates.get('prev_valid_labeled')
                    prev_valid_original_index = state_updates.get('prev_valid_original_index')
                    prev_best_meta = state_updates.get('prev_best_meta')
                    prev_valid_frame_id = state_updates.get('prev_valid_frame_id')
                    if prev_best_meta is not None and prev_valid_frame_id is not None:
                        similarity_snapshot_history[int(prev_valid_frame_id)] = prev_best_meta
                        for old_fid in list(similarity_snapshot_history.keys()):
                            if old_fid < int(fn) - 3:
                                similarity_snapshot_history.pop(old_fid, None)
                continue
            valid_keypoints_count = _count_valid_keypoints(kps)
            if valid_keypoints_count < 4:
                kps_list = kps or []
                ordered_kps = []
                for i in range(template_len):
                    if i < len(kps_list) and kps_list[i] and (len(kps_list[i]) >= 2):
                        ordered_kps.append([float(kps_list[i][0]), float(kps_list[i][1])])
                    else:
                        ordered_kps.append([0.0, 0.0])
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
                found['added_four_point'] = False
                found.pop('original_index', None)
                best_entries.append({'frame_id': int(fn), 'match_idx': None, 'candidate_idx': None, 'candidate': [], 'avg_distance': float('inf'), 'reordered_keypoints': ordered_kps, 'score': 0.0, 'decision': 'right' if FORCE_DECISION_RIGHT else None, 'added_four_point': False})
                parts_ms['other_result'] = 0.0
                _log(f'{progress_prefix} - done')
                prof.end_frame(frame_id=int(fn), t_frame0=t_frame0, parts_ms=parts_ms)
                continue
            _log(f'{progress_prefix} - step 1/4: connections')
            t1 = time.perf_counter()
            step1_entry = _step1_build_connections(frame=frame, kps=kps, labels=labels, frame_number=int(fn), frame_width=int(W), frame_height=int(H), cached_edges=cached_edges)
            parts_ms['step1'] = (time.perf_counter() - t1) * 1000.0
            _push(step1_outputs, step1_entry)
            _log(f'{progress_prefix} - step 2/4: four-point groups')
            t2 = time.perf_counter()
            step2_entry = _step2_build_four_point_groups(frame=frame, step1_entry=step1_entry, frame_number=int(fn))
            parts_ms['step2'] = (time.perf_counter() - t2) * 1000.0
            _push(step2_outputs, step2_entry)
            _log(f'{progress_prefix} - step 3/4: matching')
            t3 = time.perf_counter()
            step3_entry = _step3_build_ordered_candidates(step2_entry=step2_entry, template_patterns=template_patterns, template_patterns_allowed_left=template_patterns_allowed_left, template_patterns_allowed_right=template_patterns_allowed_right, template_labels=template_labels, frame_number=int(fn))
            parts_ms['step3'] = (time.perf_counter() - t3) * 1000.0
            if prof.enabled:
                sub = step3_entry.get('step3_profile') or {}
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        try:
                            parts_ms[str(k)] = float(v)
                        except Exception:
                            pass
            _push(step3_outputs, step3_entry)
            _log(f'{progress_prefix} - step 4/4: looking for best order')
            matches = step3_entry.get('matches') or []
            orig_kps = original_keypoints.get(int(fn)) or []
            frame_keypoints = step3_entry.get('keypoints') or []
            frame_labels = step3_entry.get('labels') or []
            decision = 'right' if FORCE_DECISION_RIGHT else step3_entry.get('decision')
            t4 = time.perf_counter()
            best_avg, best_meta, orig_idx_map = _step4_pick_best_candidate(matches=matches, orig_kps=orig_kps, frame_keypoints=frame_keypoints, frame_labels=frame_labels, decision=decision, template_pts=template_pts, frame_number=int(fn), frame=frame)
            parts_ms['step4'] = (time.perf_counter() - t4) * 1000.0
            similarity_snapshot_for_next: dict[str, Any] | None = None
            if best_meta:
                ordered_kps = best_meta['reordered_keypoints']
                step4_0_ordered_kps = [[float(pt[0]), float(pt[1])] if pt and len(pt) >= 2 else [0.0, 0.0] for pt in ordered_kps]
                similarity_snapshot_for_next = {'frame_id': int(fn), 'reordered_keypoints': step4_0_ordered_kps, 'decision': best_meta.get('decision')}
                best_meta['step4_0_reordered_keypoints'] = step4_0_ordered_kps
                fallback_used = False
                need_fallback = _count_valid_keypoints(ordered_kps) < 4
                fallback_used = False
                if STEP4_1_ENABLED:
                    t_step41 = time.perf_counter()
                    best_meta['step4_1'] = _step4_1_compute_border_line(frame=frame, ordered_kps=ordered_kps, frame_number=int(fn), cached_edges=cached_edges, prev_best_aay=prev_step4_1_seed.get('best_aay') if prev_step4_1_seed else None, prev_best_bby=prev_step4_1_seed.get('best_bby') if prev_step4_1_seed else None)
                    parts_ms['step4_1'] = (time.perf_counter() - t_step41) * 1000.0
                    _merge_step4_1_profile_into_parts(best_meta.get('step4_1'), parts_ms)
                    s41 = best_meta.get('step4_1') or {}
                    if s41.get('best_aay') is not None and s41.get('best_bby') is not None:
                        prev_step4_1_seed = {'best_aay': int(s41['best_aay']), 'best_bby': int(s41['best_bby'])}
                else:
                    best_meta['step4_1'] = {}
                best_meta['step4_2'] = {}
                best_meta['reordered_keypoints'] = ordered_kps
                if STEP4_3_ENABLED:
                    best_meta['step4_3'] = _step4_3_debug_dilate_and_lines(frame=frame, ordered_kps=best_meta['reordered_keypoints'], frame_number=int(fn), decision=best_meta.get('decision'), step4_1=best_meta.get('step4_1'), cached_edges=cached_edges)
                    _merge_step4_3_profile_into_parts(best_meta.get('step4_3'), parts_ms)
                else:
                    best_meta['step4_3'] = {}
                t_step49 = time.perf_counter()
                step4_9_kps, step4_9_score = _step4_9_select_h_and_keypoints(input_kps=best_meta['reordered_keypoints'], step4_1=best_meta.get('step4_1'), step4_3=best_meta.get('step4_3'), frame=frame, frame_number=int(fn), decision=best_meta.get('decision'), cached_edges=cached_edges)
                parts_ms['step4_9'] = (time.perf_counter() - t_step49) * 1000.0
                if step4_9_kps is not None:
                    best_meta['reordered_keypoints'] = step4_9_kps
                    best_meta['score'] = step4_9_score
                    ordered_kps = step4_9_kps
                step4_9_passed = step4_9_kps is not None
                best_meta['validation_passed'] = step4_9_passed
                if not step4_9_passed:
                    best_meta['validation_error'] = 'Step 4.9 did not produce keypoints'
                    need_fallback = False
                else:
                    best_meta.pop('validation_error', None)
                if STEP6_ENABLED and STEP6_FILL_MISSING_ENABLED and step4_9_passed and (not need_fallback) and (not fallback_used) and (best_meta.get('added_four_point') != True):
                    t6 = time.perf_counter()
                    ordered_kps = _step6_fill_keypoints_from_homography(ordered_kps=ordered_kps, frame=frame, frame_number=int(fn))
                    parts_ms['step6_fill'] = parts_ms.get('step6_fill', 0.0) + (time.perf_counter() - t6) * 1000.0
                    best_meta['reordered_keypoints'] = ordered_kps
                best_entries.append(best_meta)
                _push(step5_outputs, best_meta.copy())
            else:
                orig_list = orig_kps or []
                ordered_kps = []
                for i in range(template_len):
                    if i < len(orig_list) and orig_list[i] and (len(orig_list[i]) >= 2):
                        ordered_kps.append([float(orig_list[i][0]), float(orig_list[i][1])])
                    else:
                        ordered_kps.append([0.0, 0.0])
                best_meta = {'frame_id': int(fn), 'match_idx': None, 'candidate_idx': None, 'candidate': [], 'avg_distance': float('inf'), 'reordered_keypoints': ordered_kps, 'fallback': False, 'decision': decision, 'added_four_point': False}
                step4_0_ordered_kps = [[float(pt[0]), float(pt[1])] if pt and len(pt) >= 2 else [0.0, 0.0] for pt in ordered_kps]
                similarity_snapshot_for_next = {'frame_id': int(fn), 'reordered_keypoints': step4_0_ordered_kps, 'decision': decision}
                best_meta['step4_0_reordered_keypoints'] = step4_0_ordered_kps
                best_meta['score'] = float(best_meta.get('score', 0.0) or 0.0)
                best_entries.append(best_meta)
                best_meta_copy = best_meta.copy()
                best_meta_copy['score'] = best_meta.get('score', 0.0)
                _push(step5_outputs, best_meta_copy)
            t_other_result = time.perf_counter()
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
            prev_valid_labeled = labeled_cur
            prev_valid_original_index = orig_idx_map if best_meta.get('match_idx') is not None else None
            prev_best_meta = similarity_snapshot_for_next if similarity_snapshot_for_next is not None else best_meta.copy()
            prev_valid_frame_id = fn
            if prev_best_meta is not None:
                similarity_snapshot_history[int(fn)] = prev_best_meta
                for old_fid in list(similarity_snapshot_history.keys()):
                    if old_fid < int(fn) - 3:
                        similarity_snapshot_history.pop(old_fid, None)
            _log(f'{progress_prefix} - done')
            parts_ms['other_result'] = (time.perf_counter() - t_other_result) * 1000.0
            prof.end_frame(frame_id=int(fn), t_frame0=t_frame0, parts_ms=parts_ms)
        global _LAST_ERROR_MSG, _CURRENT_PROGRESS
        _LAST_ERROR_MSG = ''
        _CURRENT_PROGRESS = ''
        print('\r' + ' ' * 135 + '\rProcessing frames done.')
        t7 = time.perf_counter()
        _step7_interpolate_problematic_frames(step4_9_outputs=best_entries, ordered_frames=ordered_frames, template_len=template_len, out_step7=args.out_step7)
        if prof.enabled:
            prof._add('step7_interpolate', (time.perf_counter() - t7) * 1000.0)
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
                            pass
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
                valid_indices_set = set(valid_indices)
                adjusted_kps: list[list[float]] = [[0.0, 0.0]] * num_kps
                for idx in np.where(valid_mask)[0]:
                    if STEP6_FILL_MISSING_ENABLED or idx in valid_indices_set:
                        adjusted_kps[idx] = [float(adj_x_arr[idx]), float(adj_y_arr[idx])]
                frame_entry['keypoints'] = adjusted_kps
                processed_count += 1
            print(f'Step 8: Completed processing {total_frames} frames (adjusted: {processed_count}, skipped: {skipped_count}).')
            if prof.enabled:
                prof._add('step8_adjust', (time.perf_counter() - t8) * 1000.0)
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