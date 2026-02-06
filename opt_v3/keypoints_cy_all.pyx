# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
cimport numpy as cnp
import cython
from libc.math cimport sqrt, floor

np.import_array()


# ---- from _add4_segment_candidates_cy.pyx ----
def segments_from_col_band(
    np.ndarray[np.int64_t, ndim=1] col_band,
    int seg_width,
    int step,
    int total_pixels,
):
    """
    Build segment candidates for vertical slices.
    Returns list of {"x": float, "rate": float} sorted by rate desc.
    """
    cdef Py_ssize_t n = col_band.shape[0]
    if n <= 0 or seg_width <= 0 or step <= 0:
        return []
    cdef int half = seg_width // 2
    cdef np.ndarray[np.int64_t, ndim=1] csum = np.empty(n, dtype=np.int64)
    cdef long long running = 0
    cdef Py_ssize_t i
    for i in range(n):
        running += <long long>col_band[i]
        csum[i] = running
    cdef double denom = float(total_pixels)
    cdef long long white_count
    cdef double rate
    cdef Py_ssize_t x_start
    cdef Py_ssize_t x_end
    res = []
    for x_pos in range(half, n - half, step):
        x_start = x_pos - half
        x_end = x_pos + half
        if x_start <= 0:
            white_count = <long long>csum[x_end]
        else:
            white_count = <long long>(csum[x_end] - csum[x_start - 1])
        if denom > 0.0:
            rate = white_count / denom
        else:
            rate = 0.0
        res.append({"x": float(x_pos), "rate": float(rate)})
    res.sort(key=lambda item: item["rate"], reverse=True)
    return res


def segments_from_row_band(
    np.ndarray[np.int64_t, ndim=1] row_band,
    int seg_width,
    int step,
    int total_pixels,
):
    """
    Build segment candidates for horizontal slices.
    Returns list of {"y": float, "rate": float} sorted by rate desc.
    """
    cdef Py_ssize_t n = row_band.shape[0]
    if n <= 0 or seg_width <= 0 or step <= 0:
        return []
    cdef int half = seg_width // 2
    cdef np.ndarray[np.int64_t, ndim=1] csum = np.empty(n, dtype=np.int64)
    cdef long long running = 0
    cdef Py_ssize_t i
    for i in range(n):
        running += <long long>row_band[i]
        csum[i] = running
    cdef double denom = float(total_pixels)
    cdef long long white_count
    cdef double rate
    cdef Py_ssize_t y_start
    cdef Py_ssize_t y_end
    res = []
    for y_pos in range(half, n - half, step):
        y_start = y_pos - half
        y_end = y_pos + half
        if y_start <= 0:
            white_count = <long long>csum[y_end]
        else:
            white_count = <long long>(csum[y_end] - csum[y_start - 1])
        if denom > 0.0:
            rate = white_count / denom
        else:
            rate = 0.0
        res.append({"y": float(y_pos), "rate": float(rate)})
    res.sort(key=lambda item: item["rate"], reverse=True)
    return res


cdef list _segments_from_col_prefix(
    np.int32_t[:, :] col_prefix,
    int y_start,
    int y_end,
    int frame_width,
    int seg_width,
    int seg_step,
):
    cdef int n = frame_width
    cdef int height = y_end - y_start
    if n <= 0 or seg_width <= 0 or seg_step <= 0 or height <= 0:
        return []
    cdef int half = seg_width // 2
    if n < seg_width:
        return []
    cdef long long total_pixels = <long long>seg_width * <long long>height
    if total_pixels <= 0:
        return []

    cdef np.ndarray[np.int64_t, ndim=1] col_band = np.empty(n, dtype=np.int64)
    cdef long long[:] col_band_mv = col_band
    cdef int x
    cdef int y_end_idx = y_end - 1
    cdef int y_start_idx = y_start - 1
    if y_start <= 0:
        for x in range(n):
            col_band_mv[x] = <long long>col_prefix[y_end_idx, x]
    else:
        for x in range(n):
            col_band_mv[x] = (
                <long long>col_prefix[y_end_idx, x]
                - <long long>col_prefix[y_start_idx, x]
            )

    cdef int roll_len = n - seg_width + 1
    cdef np.ndarray[np.int64_t, ndim=1] rolling = np.empty(roll_len, dtype=np.int64)
    cdef long long[:] rolling_mv = rolling
    cdef long long running = 0
    for x in range(seg_width):
        running += col_band_mv[x]
    rolling_mv[0] = running
    for x in range(1, roll_len):
        running += col_band_mv[x + seg_width - 1] - col_band_mv[x - 1]
        rolling_mv[x] = running

    res = []
    cdef double denom = float(total_pixels)
    cdef long long white_count
    cdef double rate
    cdef int x_pos
    cdef int x_start
    for x_pos in range(half, n - half, seg_step):
        x_start = x_pos - half
        if x_start < 0 or x_start >= roll_len:
            continue
        white_count = rolling_mv[x_start]
        rate = white_count / denom if denom > 0.0 else 0.0
        res.append({"x": float(x_pos), "rate": float(rate)})
    res.sort(key=lambda item: item["rate"], reverse=True)
    return res


cdef list _segments_from_row_prefix(
    np.int32_t[:, :] row_prefix,
    int x_start,
    int x_end,
    int frame_height,
    int seg_width,
    int seg_step,
):
    cdef int n = frame_height
    cdef int width = x_end - x_start
    if n <= 0 or seg_width <= 0 or seg_step <= 0 or width <= 0:
        return []
    cdef int half = seg_width // 2
    if n < seg_width:
        return []
    cdef long long total_pixels = <long long>seg_width * <long long>width
    if total_pixels <= 0:
        return []

    cdef np.ndarray[np.int64_t, ndim=1] row_band = np.empty(n, dtype=np.int64)
    cdef long long[:] row_band_mv = row_band
    cdef int y
    cdef int x_end_idx = x_end - 1
    cdef int x_start_idx = x_start - 1
    if x_start <= 0:
        for y in range(n):
            row_band_mv[y] = <long long>row_prefix[y, x_end_idx]
    else:
        for y in range(n):
            row_band_mv[y] = (
                <long long>row_prefix[y, x_end_idx]
                - <long long>row_prefix[y, x_start_idx]
            )

    cdef int roll_len = n - seg_width + 1
    cdef np.ndarray[np.int64_t, ndim=1] rolling = np.empty(roll_len, dtype=np.int64)
    cdef long long[:] rolling_mv = rolling
    cdef long long running = 0
    for y in range(seg_width):
        running += row_band_mv[y]
    rolling_mv[0] = running
    for y in range(1, roll_len):
        running += row_band_mv[y + seg_width - 1] - row_band_mv[y - 1]
        rolling_mv[y] = running

    res = []
    cdef double denom = float(total_pixels)
    cdef long long white_count
    cdef double rate
    cdef int y_pos
    cdef int y_start
    for y_pos in range(half, n - half, seg_step):
        y_start = y_pos - half
        if y_start < 0 or y_start >= roll_len:
            continue
        white_count = rolling_mv[y_start]
        rate = white_count / denom if denom > 0.0 else 0.0
        res.append({"y": float(y_pos), "rate": float(rate)})
    res.sort(key=lambda item: item["rate"], reverse=True)
    return res


def precompute_segments_from_prefix(
    np.ndarray[np.int32_t, ndim=2] col_prefix,
    np.ndarray[np.int32_t, ndim=2] row_prefix,
    np.ndarray[np.int32_t, ndim=1] horiz_positions,
    np.ndarray[np.int32_t, ndim=1] vert_positions,
    int frame_height,
    int frame_width,
    int seg_width,
    int seg_step,
):
    """
    Precompute segment candidates for all horizontal/vertical lines using prefix sums.
    Returns (seg_down_by_line, seg_up_by_line, seg_right_by_line, seg_left_by_line).
    """
    cdef dict seg_down_by_line = {}
    cdef dict seg_up_by_line = {}
    cdef dict seg_right_by_line = {}
    cdef dict seg_left_by_line = {}

    cdef Py_ssize_t i
    cdef int pos
    for i in range(horiz_positions.shape[0]):
        pos = <int>horiz_positions[i]
        seg_down_by_line[pos] = _segments_from_col_prefix(
            col_prefix, pos, frame_height, frame_width, seg_width, seg_step
        )
        seg_up_by_line[pos] = _segments_from_col_prefix(
            col_prefix, 0, pos, frame_width, seg_width, seg_step
        )

    for i in range(vert_positions.shape[0]):
        pos = <int>vert_positions[i]
        seg_right_by_line[pos] = _segments_from_row_prefix(
            row_prefix, pos, frame_width, frame_height, seg_width, seg_step
        )
        seg_left_by_line[pos] = _segments_from_row_prefix(
            row_prefix, 0, pos, frame_height, seg_width, seg_step
        )

    return seg_down_by_line, seg_up_by_line, seg_right_by_line, seg_left_by_line


# ---- from _connected_by_segment_cy.pyx ----
cdef object _py_round = round


@cython.boundscheck(False)
@cython.wraparound(False)
def connected_by_segment(
    cnp.ndarray[cnp.uint8_t, ndim=2] mask,
    tuple p1,
    tuple p2,
    int sample_radius=5,
    int close_ksize=0,
    double min_hit_ratio=0.35,
    int max_gap_px=35,
):
    """
    Cythonized variant of _connected_by_segment.
    Assumes mask is uint8 and already closed (close_ksize must be 0).
    """
    cdef int x1 = <int>p1[0]
    cdef int y1 = <int>p1[1]
    cdef int x2 = <int>p2[0]
    cdef int y2 = <int>p2[1]
    cdef int H = mask.shape[0]
    cdef int W = mask.shape[1]
    cdef int dx = x2 - x1
    cdef int dy = y2 - y1
    cdef int L = <int>sqrt(dx * dx + dy * dy)
    if L < 2:
        return False, 0.0, L

    cdef double step_x = dx / <double>L
    cdef double step_y = dy / <double>L
    cdef int total = L + 1
    cdef int hits = 0
    cdef int longest = 0
    cdef int cur = 0
    cdef int i, xi, yi, x0, x1b, y0, y1b
    cdef int yy, xx
    cdef bint hit
    cdef double x, y
    cdef int r = sample_radius

    for i in range(total):
        x = x1 + step_x * i
        y = y1 + step_y * i
        xi = <int>_py_round(x)
        yi = <int>_py_round(y)
        x0 = xi - r
        if x0 < 0:
            x0 = 0
        x1b = xi + r + 1
        if x1b > W:
            x1b = W
        y0 = yi - r
        if y0 < 0:
            y0 = 0
        y1b = yi + r + 1
        if y1b > H:
            y1b = H

        hit = False
        for yy in range(y0, y1b):
            for xx in range(x0, x1b):
                if mask[yy, xx] > 0:
                    hit = True
                    break
            if hit:
                break

        if hit:
            hits += 1
            cur = 0
        else:
            cur += 1
            if cur > longest:
                longest = cur

    cdef double hit_ratio = hits / <double>total
    cdef bint ok = (hit_ratio >= min_hit_ratio) and (longest <= max_gap_px)
    return ok, hit_ratio, longest


# ---- from _eval_kp_helpers_cy.pyx ----
def normalize_keypoints(original_keypoints, int frame_width, int frame_height):
    """
    Normalize keypoints to float tuples and clamp out-of-bounds to (0, 0).
    Mirrors evaluate_keypoints_for_frame Step 0 behavior.
    """
    cdef Py_ssize_t n = len(original_keypoints)
    cdef Py_ssize_t i
    cdef object pt
    cdef double xf, yf
    out = [None] * n
    for i in range(n):
        pt = original_keypoints[i]
        xf = float(pt[0])
        yf = float(pt[1])
        if (xf != 0.0 or yf != 0.0) and (
            xf < 0.0 or yf < 0.0 or xf >= frame_width or yf >= frame_height
        ):
            out[i] = (0.0, 0.0)
        else:
            out[i] = (xf, yf)
    return out


# ---- from _step3_conn_constraints_cy.pyx ----
cdef inline int _idx_in_quad(int v, int q0, int q1, int q2, int q3) nogil:
    if v == q0:
        return 0
    if v == q1:
        return 1
    if v == q2:
        return 2
    if v == q3:
        return 3
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
def filter_connection_constraints(
    cnp.ndarray[cnp.int32_t, ndim=2] candidates,
    cnp.ndarray[cnp.int32_t, ndim=1] quad,
    cnp.ndarray[cnp.int32_t, ndim=2] frame_edges,
    cnp.ndarray[cnp.uint64_t, ndim=1] frame_reach3,
    cnp.ndarray[cnp.uint64_t, ndim=1] template_adj_mask,
    cnp.ndarray[cnp.uint64_t, ndim=1] template_reach2_mask,
):
    """
    Fast connection-constraint filtering for Step 3.
    Returns a list of candidate indices to keep.
    """
    cdef Py_ssize_t n = candidates.shape[0]
    cdef Py_ssize_t i, j, k
    cdef int q0 = quad[0]
    cdef int q1 = quad[1]
    cdef int q2 = quad[2]
    cdef int q3 = quad[3]
    cdef int qi, qj, ti, tj
    cdef int idx_i, idx_j
    cdef bint ok
    cdef unsigned long long reach_mask
    out = []

    for i in range(n):
        ok = True
        # reachability constraint (<=3 hops in frame => <=2 hops in template)
        for idx_i in range(4):
            qi = quad[idx_i]
            ti = candidates[i, idx_i]
            reach_mask = frame_reach3[qi]
            for idx_j in range(4):
                if idx_j == idx_i:
                    continue
                qj = quad[idx_j]
                if (reach_mask >> qj) & 1:
                    tj = candidates[i, idx_j]
                    if ((template_reach2_mask[ti] >> tj) & 1) == 0:
                        ok = False
                        break
            if not ok:
                break
        if not ok:
            continue

        # direct edge constraint: if frame edge exists between mapped points, template must have edge
        for k in range(frame_edges.shape[0]):
            idx_i = _idx_in_quad(frame_edges[k, 0], q0, q1, q2, q3)
            if idx_i < 0:
                continue
            idx_j = _idx_in_quad(frame_edges[k, 1], q0, q1, q2, q3)
            if idx_j < 0:
                continue
            ti = candidates[i, idx_i]
            tj = candidates[i, idx_j]
            if ((template_adj_mask[ti] >> tj) & 1) == 0:
                ok = False
                break
        if ok:
            out.append(i)

    return out


# ---- from _step3_conn_label_constraints_cy.pyx ----
@cython.boundscheck(False)
@cython.wraparound(False)
def filter_connection_label_constraints(
    cnp.ndarray[cnp.int32_t, ndim=2] candidates,
    cnp.ndarray[cnp.int32_t, ndim=1] quad,
    cnp.ndarray[cnp.int32_t, ndim=1] frame_labels,
    cnp.ndarray[cnp.uint64_t, ndim=1] frame_adj_mask,
    cnp.ndarray[cnp.uint64_t, ndim=1] template_adj_mask,
    cnp.ndarray[cnp.uint64_t, ndim=1] template_neighbor_label_mask,
    int label1_min_idx,
    int label1_max_idx,
    int decision_flag,  # -1 none, 0 left, 1 right
):
    """
    Fast connection-label-constraint filtering for Step 3.
    Returns a list of candidate indices to keep.
    """
    cdef Py_ssize_t n = candidates.shape[0]
    cdef Py_ssize_t i
    cdef int idx, qi, ti, nb, lab
    cdef unsigned long long mask, lsb, labels_mask
    cdef bint ok
    cdef unsigned long long adj_mask, label_mask
    out = []

    for i in range(n):
        ok = True
        for idx in range(4):
            qi = quad[idx]
            ti = candidates[i, idx]
            if qi < 0 or qi >= frame_adj_mask.shape[0]:
                ok = False
                break
            mask = frame_adj_mask[qi]

            # label1 min/max neighbor adjacency constraints
            if decision_flag >= 0:
                if label1_min_idx >= 0 and ((mask >> label1_min_idx) & 1):
                    if decision_flag == 1:
                        # right: min -> 24
                        if ((template_adj_mask[ti] >> 24) & 1) == 0:
                            ok = False
                            break
                    else:
                        # left: min -> 0
                        if ((template_adj_mask[ti] >> 0) & 1) == 0:
                            ok = False
                            break
                if label1_max_idx >= 0 and ((mask >> label1_max_idx) & 1):
                    if decision_flag == 1:
                        # right: max -> 29
                        if ((template_adj_mask[ti] >> 29) & 1) == 0:
                            ok = False
                            break
                    else:
                        # left: max -> 5
                        if ((template_adj_mask[ti] >> 5) & 1) == 0:
                            ok = False
                            break

            # connected labels subset check
            labels_mask = 0
            while mask:
                lsb = mask & (~mask + 1)
                nb = <int>(lsb.bit_length() - 1)
                lab = frame_labels[nb]
                if lab > 0:
                    labels_mask |= (1 << lab)
                mask &= mask - 1

            label_mask = template_neighbor_label_mask[ti]
            if labels_mask & (~label_mask):
                ok = False
                break

        if ok:
            out.append(i)

    return out


# ---- from _step3_filter_labels_cy.pyx ----
@cython.boundscheck(False)
@cython.wraparound(False)
def filter_labels(
    cnp.ndarray[cnp.int32_t, ndim=2] candidates,
    cnp.ndarray[cnp.int32_t, ndim=1] quad,
    cnp.ndarray[cnp.int32_t, ndim=1] frame_labels,
    cnp.ndarray[cnp.int32_t, ndim=1] template_labels,
    cnp.ndarray[cnp.int64_t, ndim=1] constraints_mask,
):
    """
    Fast label/constraint filtering for Step 3.
    Returns a list of candidate indices to keep.
    """
    cdef Py_ssize_t n = candidates.shape[0]
    cdef Py_ssize_t i, j
    cdef int ci, qi
    cdef long long mask
    cdef bint ok
    out = []

    for i in range(n):
        ok = True
        for j in range(4):
            ci = candidates[i, j]
            qi = quad[j]
            if qi < 0 or qi >= frame_labels.shape[0]:
                ok = False
                break
            if ci < 0 or ci >= template_labels.shape[0]:
                ok = False
                break
            mask = constraints_mask[qi]
            if mask != -1 and ((mask >> ci) & 1) == 0:
                ok = False
                break
            if template_labels[ci] != frame_labels[qi]:
                ok = False
                break
        if ok:
            out.append(i)

    return out


# ---- Sloping line white count and area search (for h_candidates / v_candidates speedup) ----
@cython.boundscheck(False)
@cython.wraparound(False)
def sloping_line_white_count_cy(
    np.ndarray[np.uint8_t, ndim=1] edges_flat,
    int w,
    int h,
    double ax,
    double ay,
    double bx,
    double by,
    int half_width,
    int sample_max,
):
    """
    Count white pixels in a (2*half_width+1)-px band along (ax,ay)->(bx,by).
    Returns (white_count, total_unique_pixel_count). Matches Python _sloping_line_white_count logic.
    """
    cdef double L = sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay))
    if L < 1.0:
        return 0, 0
    cdef int n_steps = min(max(1, <int>L + 1), sample_max)
    cdef double perp_x = -(by - ay) / L
    cdef double perp_y = (bx - ax) / L
    cdef int buf_size = (2 * half_width + 1) * n_steps
    if buf_size > 2048:
        buf_size = 2048
    cdef np.ndarray[np.int64_t, ndim=1] buf = np.empty(buf_size, dtype=np.int64)
    cdef long long[:] buf_mv = buf
    cdef int n_used = 0
    cdef double t, px, py
    cdef int k, qx, qy
    cdef long long linear
    cdef int i
    for i in range(n_steps):
        t = (<double>i) / (<double>max(1, n_steps - 1))
        px = ax + t * (bx - ax)
        py = ay + t * (by - ay)
        for k in range(-half_width, half_width + 1):
            qx = <int>floor(px + <double>k * perp_x + 0.5)
            qy = <int>floor(py + <double>k * perp_y + 0.5)
            if 0 <= qx < w and 0 <= qy < h and n_used < buf_size:
                linear = <long long>qy * w + qx
                buf_mv[n_used] = linear
                n_used += 1
    if n_used == 0:
        return 0, 0
    buf[:n_used].sort()
    cdef long long prev = -1
    cdef int total = 0
    cdef int white = 0
    cdef long long idx
    for i in range(n_used):
        idx = buf_mv[i]
        if idx != prev:
            total += 1
            if edges_flat[idx]:
                white += 1
            prev = idx
    return white, total


@cython.boundscheck(False)
@cython.wraparound(False)
def search_horizontal_in_area_cy(
    np.ndarray[np.uint8_t, ndim=1] edges_flat,
    int w,
    int h,
    int y_lo,
    int y_hi,
    double max_slope,
    int coarse_step,
    int refine_radius,
    int refine_step,
    int min_slope_px,
    int half_width,
    int sample_max,
):
    """
    Search horizontal sloping line in [y_lo, y_hi]. Returns (best_y0, best_y1, best_white, best_total)
    or (-1, -1, -1, -1) if no valid candidate.
    """
    if y_hi <= y_lo + min_slope_px:
        return -1, -1, -1, -1
    cdef int max_slope_int = max(min_slope_px, <int>max_slope)
    cdef int best_count = -1
    cdef int best_y0 = y_lo, best_y1 = y_lo + min_slope_px
    cdef int best_white = 0, best_total = 0
    cdef int y0, y1, white, total
    cdef double bx = <double>(w - 1)
    # Coarse loop
    y0 = y_lo
    while y0 <= y_hi:
        y1 = y_lo
        while y1 <= y_hi:
            if abs(y1 - y0) >= min_slope_px and abs(y1 - y0) <= max_slope_int:
                white, total = sloping_line_white_count_cy(
                    edges_flat, w, h, 0.0, <double>y0, bx, <double>y1,
                    half_width, sample_max,
                )
                if total > 0 and white > best_count:
                    best_count = white
                    best_y0, best_y1 = y0, y1
                    best_white, best_total = white, total
            y1 += coarse_step
        if y1 - coarse_step != y_hi and y1 <= y_hi + coarse_step:
            y1 = y_hi
            if abs(y1 - y0) >= min_slope_px and abs(y1 - y0) <= max_slope_int:
                white, total = sloping_line_white_count_cy(
                    edges_flat, w, h, 0.0, <double>y0, bx, <double>y1,
                    half_width, sample_max,
                )
                if total > 0 and white > best_count:
                    best_count = white
                    best_y0, best_y1 = y0, y1
                    best_white, best_total = white, total
        y0 += coarse_step
    if y0 - coarse_step != y_hi and y0 <= y_hi + coarse_step:
        y0 = y_hi
        y1 = y_lo
        while y1 <= y_hi:
            if abs(y1 - y0) >= min_slope_px and abs(y1 - y0) <= max_slope_int:
                white, total = sloping_line_white_count_cy(
                    edges_flat, w, h, 0.0, <double>y0, bx, <double>y1,
                    half_width, sample_max,
                )
                if total > 0 and white > best_count:
                    best_count = white
                    best_y0, best_y1 = y0, y1
                    best_white, best_total = white, total
            y1 += coarse_step
    if best_count < 0:
        return -1, -1, -1, -1
    # Refine
    cdef int y0_ref_min = max(y_lo, best_y0 - refine_radius)
    cdef int y0_ref_max = min(y_hi, best_y0 + refine_radius)
    cdef int y1_ref_min = max(y_lo, best_y1 - refine_radius)
    cdef int y1_ref_max = min(y_hi, best_y1 + refine_radius)
    y0 = y0_ref_min
    while y0 <= y0_ref_max:
        y1 = y1_ref_min
        while y1 <= y1_ref_max:
            if abs(y1 - y0) >= min_slope_px and abs(y1 - y0) <= max_slope_int:
                white, total = sloping_line_white_count_cy(
                    edges_flat, w, h, 0.0, <double>y0, bx, <double>y1,
                    half_width, sample_max,
                )
                if total > 0 and white > best_count:
                    best_count = white
                    best_y0, best_y1 = y0, y1
                    best_white, best_total = white, total
            y1 += refine_step
        y0 += refine_step
    return best_y0, best_y1, best_white, best_total


@cython.boundscheck(False)
@cython.wraparound(False)
def search_vertical_in_area_cy(
    np.ndarray[np.uint8_t, ndim=1] edges_flat,
    int w,
    int h,
    int x_lo,
    int x_hi,
    double max_slope,
    int coarse_step,
    int refine_radius,
    int refine_step,
    int min_slope_px,
    int half_width,
    int sample_max,
):
    """
    Search vertical sloping line in [x_lo, x_hi]. Returns (best_x0, best_x1, best_white, best_total)
    or (-1, -1, -1, -1) if no valid candidate.
    """
    if x_hi <= x_lo + min_slope_px:
        return -1, -1, -1, -1
    cdef int max_slope_int = max(min_slope_px, <int>max_slope)
    cdef int best_count = -1
    cdef int best_x0 = x_lo, best_x1 = x_lo + min_slope_px
    cdef int best_white = 0, best_total = 0
    cdef int x0, x1, white, total
    cdef double by = <double>(h - 1)
    # Coarse loop
    x0 = x_lo
    while x0 <= x_hi:
        x1 = x_lo
        while x1 <= x_hi:
            if abs(x1 - x0) >= min_slope_px and abs(x1 - x0) <= max_slope_int:
                white, total = sloping_line_white_count_cy(
                    edges_flat, w, h, <double>x0, 0.0, <double>x1, by,
                    half_width, sample_max,
                )
                if total > 0 and white > best_count:
                    best_count = white
                    best_x0, best_x1 = x0, x1
                    best_white, best_total = white, total
            x1 += coarse_step
        if x1 - coarse_step != x_hi and x1 <= x_hi + coarse_step:
            x1 = x_hi
            if abs(x1 - x0) >= min_slope_px and abs(x1 - x0) <= max_slope_int:
                white, total = sloping_line_white_count_cy(
                    edges_flat, w, h, <double>x0, 0.0, <double>x1, by,
                    half_width, sample_max,
                )
                if total > 0 and white > best_count:
                    best_count = white
                    best_x0, best_x1 = x0, x1
                    best_white, best_total = white, total
        x0 += coarse_step
    if x0 - coarse_step != x_hi and x0 <= x_hi + coarse_step:
        x0 = x_hi
        x1 = x_lo
        while x1 <= x_hi:
            if abs(x1 - x0) >= min_slope_px and abs(x1 - x0) <= max_slope_int:
                white, total = sloping_line_white_count_cy(
                    edges_flat, w, h, <double>x0, 0.0, <double>x1, by,
                    half_width, sample_max,
                )
                if total > 0 and white > best_count:
                    best_count = white
                    best_x0, best_x1 = x0, x1
                    best_white, best_total = white, total
            x1 += coarse_step
    if best_count < 0:
        return -1, -1, -1, -1
    # Refine
    cdef int x0_ref_min = max(x_lo, best_x0 - refine_radius)
    cdef int x0_ref_max = min(x_hi, best_x0 + refine_radius)
    cdef int x1_ref_min = max(x_lo, best_x1 - refine_radius)
    cdef int x1_ref_max = min(x_hi, best_x1 + refine_radius)
    x0 = x0_ref_min
    while x0 <= x0_ref_max:
        x1 = x1_ref_min
        while x1 <= x1_ref_max:
            if abs(x1 - x0) >= min_slope_px and abs(x1 - x0) <= max_slope_int:
                white, total = sloping_line_white_count_cy(
                    edges_flat, w, h, <double>x0, 0.0, <double>x1, by,
                    half_width, sample_max,
                )
                if total > 0 and white > best_count:
                    best_count = white
                    best_x0, best_x1 = x0, x1
                    best_white, best_total = white, total
            x1 += refine_step
        x0 += refine_step
    return best_x0, best_x1, best_white, best_total
