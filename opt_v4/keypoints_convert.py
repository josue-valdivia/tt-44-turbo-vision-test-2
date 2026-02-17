#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - environment dependent
    np = None


DEBUG_FLAG = True
BORDER_MARGIN_PX = 50.0
SIMILARITY_DIST_THRESHOLD_PX = 30.0
SIMILARITY_MIN_COMMON_MATCHES = 4

PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
FOOTBALL_KEYPOINTS_CORRECTED: list[tuple[float, float]] = [
    (2.5, 2.5), (2.5, 139.5), (2.5, 249.5), (2.5, 430.5), (2.5, 540.5), (2.5, 678.0),
    (54.5, 249.5), (54.5, 430.5), (110.5, 340.5),
    (164.5, 139.5), (164.5, 269.0), (164.5, 411.0), (164.5, 540.5),
    (525.0, 2.5), (525.0, 249.5), (525.0, 430.5), (525.0, 678.0),
    (886.5, 139.5), (886.5, 269.0), (886.5, 411.0), (886.5, 540.5),
    (940.5, 340.5),
    (998.0, 249.5), (998.0, 430.5),
    (1048.0, 2.5), (1048.0, 139.5), (1048.0, 249.5), (1048.0, 430.5), (1048.0, 540.5), (1048.0, 678.0),
    (434.5, 340.0), (615.5, 340.0),
]
KEYPOINT_CONNECTIONS = [
    [1, 13], [0, 2, 9], [1, 3, 6], [2, 4, 7], [3, 5, 12], [4, 16], [2, 7], [3, 6], [],
    [1, 10], [9, 11], [10, 12], [4, 11],
    [0, 14, 24], [13, 15], [14, 16], [5, 15, 29],
    [18, 25], [17, 19], [18, 20], [19, 28], [],
    [23, 26], [22, 27], [13, 25], [17, 24, 26], [22, 25, 27], [23, 26, 28], [20, 27, 29], [16, 28],
    [], [],
]
KEYPOINT_LABELS = [
    1, 2, 2, 2, 2, 1, 3, 3, 0, 3, 2, 2, 3, 5, 6, 6, 5, 3, 2, 2, 3, 0, 3, 3, 1, 2, 2, 2, 2, 1, 4, 4
]
Y_ORDERING_RULES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7], [9, 10], [10, 11],
    [12, 13], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20],
    [22, 23], [24, 25], [25, 26], [27, 28], [28, 29],
]
_TEMPLATE_PATTERNS_CACHE: dict[tuple[bool, ...], list[list[int]]] | None = None
_EXPANDED_PATTERNS_CACHE: dict[tuple[bool, ...], tuple[tuple[bool, ...], ...]] = {}
_STEP3_PATTERN_CANDS_CACHE: dict[tuple[bool, ...], list[tuple[int, int, int, int]]] = {}
_STEP3_LABEL_FILTER_CACHE: dict[tuple[tuple[bool, ...], tuple[int, int, int, int]], list[tuple[int, int, int, int]]] = {}
STEP3_MAX_PRE_Y_CANDIDATES = 2500


def _debug(msg: str) -> None:
    if DEBUG_FLAG:
        print(f"[DEBUG] {msg}")


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "numpy is required for Step1-4 segment tracking. "
            "Install numpy in your Python environment and rerun."
        )


def _edge_set_from_connections(conns: list[list[int]]) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in conns}


def _pattern_for_quad(quad: list[int], edge_set: set[tuple[int, int]]) -> tuple[bool, ...]:
    return tuple(tuple(sorted((quad[i], quad[j]))) in edge_set for i, j in PAIR_ORDER)


def _build_template_patterns() -> dict[tuple[bool, ...], list[list[int]]]:
    global _TEMPLATE_PATTERNS_CACHE
    if _TEMPLATE_PATTERNS_CACHE is not None:
        return _TEMPLATE_PATTERNS_CACHE
    if DEBUG_FLAG:
        print("[DEBUG] Building template patterns (32P4 permutations)...")
    template_edges = _edge_set_from_connections(
        [[i, nbr] for i, nbrs in enumerate(KEYPOINT_CONNECTIONS) for nbr in nbrs]
    )
    patterns: dict[tuple[bool, ...], list[list[int]]] = {}
    nodes = list(range(len(KEYPOINT_CONNECTIONS)))
    for quad in itertools.permutations(nodes, 4):
        quad_l = list(quad)
        pat = _pattern_for_quad(quad_l, template_edges)
        patterns.setdefault(pat, []).append(quad_l)
    if DEBUG_FLAG:
        total = sum(len(v) for v in patterns.values())
        print(f"[DEBUG] Template patterns ready: pattern_keys={len(patterns)}, candidates={total}")
    _TEMPLATE_PATTERNS_CACHE = patterns
    return patterns


def _homography_from_points(src: list[list[float]], dst: list[list[float]]) -> np.ndarray | None:
    if len(src) < 4 or len(dst) < 4:
        return None
    A: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        A.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    A_np = np.asarray(A, dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(A_np)
    except np.linalg.LinAlgError:
        return None
    h = vh[-1, :]
    if abs(h[-1]) < 1e-9:
        return None
    H = (h / h[-1]).reshape(3, 3)
    return H


def _project_points(H: np.ndarray, pts: list[tuple[float, float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for x, y in pts:
        vec = np.array([x, y, 1.0], dtype=np.float64)
        p = H @ vec
        if abs(p[2]) < 1e-9:
            out.append([0.0, 0.0])
        else:
            out.append([float(p[0] / p[2]), float(p[1] / p[2])])
    return out


def _validate_y_ordering(ordered_keypoints: list[list[float]]) -> tuple[bool, str | None]:
    for rule_i, rule_j in Y_ORDERING_RULES:
        if rule_i >= len(ordered_keypoints) or rule_j >= len(ordered_keypoints):
            continue
        kp_i = ordered_keypoints[rule_i]
        kp_j = ordered_keypoints[rule_j]
        if not kp_i or not kp_j or len(kp_i) < 2 or len(kp_j) < 2:
            continue
        if (abs(kp_i[0]) < 1e-6 and abs(kp_i[1]) < 1e-6) or (abs(kp_j[0]) < 1e-6 and abs(kp_j[1]) < 1e-6):
            continue
        if float(kp_i[1]) >= float(kp_j[1]):
            return False, f"y-ordering violated: y[{rule_i}] >= y[{rule_j}]"
    return True, None


def _validate_y_ordering_partial(candidate: list[int], quad: list[int], frame_keypoints: list[list[float]]) -> bool:
    if len(candidate) < 4 or len(quad) < 4:
        return True
    tpl_to_frame: dict[int, int] = {int(candidate[i]): int(quad[i]) for i in range(4)}
    for rule_i, rule_j in Y_ORDERING_RULES:
        if rule_i not in tpl_to_frame or rule_j not in tpl_to_frame:
            continue
        fi = tpl_to_frame[rule_i]
        fj = tpl_to_frame[rule_j]
        if fi < 0 or fj < 0 or fi >= len(frame_keypoints) or fj >= len(frame_keypoints):
            continue
        pi = frame_keypoints[fi]
        pj = frame_keypoints[fj]
        if not pi or not pj or len(pi) < 2 or len(pj) < 2:
            continue
        if float(pi[1]) >= float(pj[1]):
            return False
    return True


def _extract_frame_kps_labels(frame: dict[str, Any]) -> tuple[list[list[float]], list[int]]:
    labeled = _extract_labeled_points(frame)
    kps: list[list[float]] = []
    labels: list[int] = []
    for item in labeled:
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            continue
        kps.append([float(x), float(y)])
        labels.append(int(_label_to_int(item.get("label")) or 0))
    return kps, labels


def _step1_build_connections_light(
    kps: list[list[float]], labels: list[int], frame_number: int, frame_width: int, frame_height: int
) -> dict[str, Any]:
    n = len(kps)
    if n < 2:
        return {"frame_id": frame_number, "frame_size": [frame_width, frame_height], "keypoints": kps, "labels": labels, "connections": []}
    diag = math.hypot(frame_width, frame_height)
    max_dist = max(80.0, diag * 0.30)
    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        dlist: list[tuple[float, int]] = []
        xi, yi = kps[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = kps[j]
            d = math.hypot(xi - xj, yi - yj)
            if d <= max_dist:
                dlist.append((d, j))
        dlist.sort(key=lambda t: t[0])
        for _, j in dlist[:2]:
            edge_set.add(tuple(sorted((i, j))))
    pair_to_len = {e: math.hypot(kps[e[0]][0] - kps[e[1]][0], kps[e[0]][1] - kps[e[1]][1]) for e in edge_set}
    edges_to_remove: set[tuple[int, int]] = set()
    nodes = sorted({idx for e in edge_set for idx in e})
    for a, b, c in itertools.combinations(nodes, 3):
        tri = [tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))]
        if all(e in pair_to_len for e in tri):
            edges_to_remove.add(max(tri, key=lambda e: pair_to_len[e]))
    connections = sorted([list(e) for e in edge_set if e not in edges_to_remove])
    return {"frame_id": frame_number, "frame_size": [frame_width, frame_height], "keypoints": kps, "labels": labels, "connections": connections}


def _step2_build_four_point_groups_light(step1_entry: dict[str, Any], frame_number: int) -> dict[str, Any]:
    keypoints = step1_entry["keypoints"]
    labels = step1_entry["labels"]
    connections: list[list[int]] = step1_entry.get("connections") or []
    frame_w, frame_h = step1_entry["frame_size"]
    border_margin = 50.0
    valid_nodes = [
        i for i, pt in enumerate(keypoints)
        if pt and len(pt) >= 2 and not (abs(pt[0]) < 1e-6 and abs(pt[1]) < 1e-6)
        and pt[0] >= border_margin and pt[0] < frame_w - border_margin and pt[1] >= border_margin and pt[1] < frame_h - border_margin
    ]
    edge_set = _edge_set_from_connections(connections)
    deg = {i: 0 for i in valid_nodes}
    for a, b in edge_set:
        if a in deg:
            deg[a] += 1
        if b in deg:
            deg[b] += 1

    def _dist_point_line(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        dx, dy = bx - ax, by - ay
        den = math.hypot(dx, dy)
        if den < 1e-6:
            return float("inf")
        return abs(dy * px - dx * py + bx * ay - by * ax) / den

    def _has_collinear_triplet(nodes4: tuple[int, int, int, int], tol: float) -> bool:
        ids = list(nodes4)
        for trip in itertools.combinations(ids, 3):
            a, b, c = trip
            pa, pb, pc = keypoints[a], keypoints[b], keypoints[c]
            if _dist_point_line(pc[0], pc[1], pa[0], pa[1], pb[0], pb[1]) <= tol:
                return True
            if _dist_point_line(pa[0], pa[1], pb[0], pb[1], pc[0], pc[1]) <= tol:
                return True
            if _dist_point_line(pb[0], pb[1], pa[0], pa[1], pc[0], pc[1]) <= tol:
                return True
        return False

    def _spread(nodes4: tuple[int, int, int, int]) -> float:
        ids = list(nodes4)
        dmin = float("inf")
        for i, j in itertools.combinations(ids, 2):
            d = math.hypot(keypoints[i][0] - keypoints[j][0], keypoints[i][1] - keypoints[j][1])
            dmin = min(dmin, d)
        return 0.0 if dmin == float("inf") else dmin

    groups_scored: list[tuple[int, float, list[int]]] = []
    for combo in itertools.combinations(valid_nodes, 4):
        if _has_collinear_triplet(combo, 20.0):
            continue
        conn_count = sum(1 for i in combo if deg.get(i, 0) > 0)
        groups_scored.append((conn_count, _spread(combo), [int(x) for x in combo]))
    if not groups_scored:
        for combo in itertools.combinations(valid_nodes, 4):
            if _has_collinear_triplet(combo, 5.0):
                continue
            conn_count = sum(1 for i in combo if deg.get(i, 0) > 0)
            groups_scored.append((conn_count, _spread(combo), [int(x) for x in combo]))
    groups_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected: list[list[int]] = []
    for _, _, grp in groups_scored:
        if len(selected) >= 3:
            break
        if not selected:
            selected.append(grp)
            continue
        if all(len(set(grp).symmetric_difference(set(sg))) >= 2 for sg in selected):
            selected.append(grp)
    return {
        "frame_id": frame_number,
        "frame_size": step1_entry["frame_size"],
        "connections": [list(e) for e in sorted(edge_set)],
        "keypoints": keypoints,
        "labels": labels,
        "four_points": selected,
    }


def _expanded_patterns(base: tuple[bool, ...]) -> set[tuple[bool, ...]]:
    cached = _EXPANDED_PATTERNS_CACHE.get(base)
    if cached is not None:
        return set(cached)
    base_l = list(base)
    false_idx = [i for i, v in enumerate(base_l) if not v]
    triangles = [(0, 1, 3), (0, 2, 4), (1, 2, 5), (3, 4, 5)]
    out: set[tuple[bool, ...]] = set()
    for mask in range(1 << len(false_idx)):
        edges = base_l[:]
        for bit, idx in enumerate(false_idx):
            if (mask >> bit) & 1:
                edges[idx] = True
        if any(edges[a] and edges[b] and edges[c] for a, b, c in triangles):
            continue
        out.add(tuple(edges))
    _EXPANDED_PATTERNS_CACHE[base] = tuple(sorted(out))
    return out


def _step3_candidates_for_pattern(
    pattern: tuple[bool, ...],
    template_patterns: dict[tuple[bool, ...], list[list[int]]],
) -> list[tuple[int, int, int, int]]:
    cached = _STEP3_PATTERN_CANDS_CACHE.get(pattern)
    if cached is not None:
        return cached
    pattern_set = _expanded_patterns(pattern)
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    for pat in pattern_set:
        for cand in template_patterns.get(pat, []):
            if len(cand) < 4:
                continue
            tup = (int(cand[0]), int(cand[1]), int(cand[2]), int(cand[3]))
            if tup in seen:
                continue
            seen.add(tup)
            out.append(tup)
    _STEP3_PATTERN_CANDS_CACHE[pattern] = out
    return out


def _step3_build_ordered_candidates_light(step2_entry: dict[str, Any], frame_number: int) -> dict[str, Any]:
    frame_connections: list[list[int]] = list(step2_entry.get("connections") or [])
    frame_four_points: list[list[int]] = list(step2_entry.get("four_points") or [])
    frame_keypoints = step2_entry.get("keypoints") or []
    frame_labels = step2_entry.get("labels") or []
    frame_edge_set = _edge_set_from_connections(frame_connections)
    template_patterns = _build_template_patterns()
    matches: list[dict[str, Any]] = []
    for quad in frame_four_points:
        pattern = _pattern_for_quad(quad, frame_edge_set)
        pattern_candidates = _step3_candidates_for_pattern(pattern, template_patterns)
        quad_labels = tuple(
            int(frame_labels[qi]) if qi >= 0 and qi < len(frame_labels) else -1
            for qi in quad[:4]
        )
        label_cache_key = (pattern, quad_labels)
        label_filtered = _STEP3_LABEL_FILTER_CACHE.get(label_cache_key)
        if label_filtered is None:
            tmp: list[tuple[int, int, int, int]] = []
            for cand in pattern_candidates:
                if (
                    KEYPOINT_LABELS[cand[0]] == quad_labels[0]
                    and KEYPOINT_LABELS[cand[1]] == quad_labels[1]
                    and KEYPOINT_LABELS[cand[2]] == quad_labels[2]
                    and KEYPOINT_LABELS[cand[3]] == quad_labels[3]
                ):
                    tmp.append(cand)
            _STEP3_LABEL_FILTER_CACHE[label_cache_key] = tmp
            label_filtered = tmp

        if len(label_filtered) > STEP3_MAX_PRE_Y_CANDIDATES:
            label_filtered = label_filtered[:STEP3_MAX_PRE_Y_CANDIDATES]
            if DEBUG_FLAG:
                _debug(
                    f"Frame {frame_number} quad={quad[:4]} Step3 pre-y capped to "
                    f"{STEP3_MAX_PRE_Y_CANDIDATES} candidates"
                )

        candidates: list[list[int]] = []
        for cand in label_filtered:
            if not _validate_y_ordering_partial(cand, quad, frame_keypoints):
                continue
            candidates.append([int(x) for x in cand[:4]])
        if DEBUG_FLAG and frame_number % 25 == 0:
            _debug(
                f"Frame {frame_number} quad={quad[:4]} Step3 candidates: "
                f"pattern={len(pattern_candidates)} label={len(label_filtered)} final={len(candidates)}"
            )
        matches.append({"four_points": [int(x) for x in quad[:4]], "candidates": candidates, "candidates_count": len(candidates)})
    return {
        "frame_id": frame_number,
        "connections": frame_connections,
        "keypoints": frame_keypoints,
        "frame_size": step2_entry.get("frame_size") or [],
        "labels": frame_labels,
        "decision": None,
        "matches": matches,
    }


def _avg_distance_to_projection(
    cand: list[int], quad: list[int], frame_keypoints: list[list[float]], frame_labels: list[int]
) -> tuple[float, list[list[float]]]:
    src = [[float(FOOTBALL_KEYPOINTS_CORRECTED[int(ci)][0]), float(FOOTBALL_KEYPOINTS_CORRECTED[int(ci)][1])] for ci in cand[:4]]
    dst = [[float(frame_keypoints[int(qi)][0]), float(frame_keypoints[int(qi)][1])] for qi in quad[:4]]
    H = _homography_from_points(src, dst)
    if H is None:
        return float("inf"), [[0.0, 0.0] for _ in FOOTBALL_KEYPOINTS_CORRECTED]
    proj = _project_points(H, FOOTBALL_KEYPOINTS_CORRECTED)
    ordered = [[0.0, 0.0] for _ in FOOTBALL_KEYPOINTS_CORRECTED]
    used: set[int] = set()
    dists: list[float] = []
    for t_idx, p in enumerate(proj):
        label_t = KEYPOINT_LABELS[t_idx] if t_idx < len(KEYPOINT_LABELS) else 0
        best_i = -1
        best_d = float("inf")
        for i, kp in enumerate(frame_keypoints):
            if i in used:
                continue
            if i >= len(frame_labels):
                continue
            if label_t > 0 and int(frame_labels[i]) != int(label_t):
                continue
            d = math.hypot(float(kp[0]) - float(p[0]), float(kp[1]) - float(p[1]))
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d <= 220.0:
            ordered[t_idx] = [float(frame_keypoints[best_i][0]), float(frame_keypoints[best_i][1])]
            used.add(best_i)
            dists.append(best_d)
    if not dists:
        return float("inf"), ordered
    return float(sum(dists) / len(dists)), ordered


def _step4_pick_best_candidate_light(step3_entry: dict[str, Any], frame_number: int) -> dict[str, Any]:
    matches = step3_entry.get("matches") or []
    frame_keypoints = step3_entry.get("keypoints") or []
    frame_labels = step3_entry.get("labels") or []
    best_avg = float("inf")
    best_meta: dict[str, Any] | None = None
    for match_idx, match in enumerate(matches):
        quad = match.get("four_points") or []
        cands = match.get("candidates") or []
        if len(quad) < 4:
            continue
        for cand_idx, cand in enumerate(cands[:50]):
            if len(cand) < 4:
                continue
            avg, ordered = _avg_distance_to_projection(cand, quad, frame_keypoints, frame_labels)
            if avg == float("inf"):
                continue
            y_ok, y_err = _validate_y_ordering(ordered)
            if not y_ok:
                continue
            if avg < best_avg:
                best_avg = avg
                best_meta = {
                    "frame_id": int(frame_number),
                    "match_idx": int(match_idx),
                    "candidate_idx": int(cand_idx),
                    "candidate": [int(x) for x in cand[:4]],
                    "four_points": [int(x) for x in quad[:4]],
                    "avg_distance": float(avg),
                    "reordered_keypoints": ordered,
                    "decision": None,
                    "y_ordering_error": y_err,
                }
    if best_meta is None:
        best_meta = {
            "frame_id": int(frame_number),
            "match_idx": None,
            "candidate_idx": None,
            "candidate": [],
            "four_points": [],
            "avg_distance": float("inf"),
            "reordered_keypoints": [[0.0, 0.0] for _ in FOOTBALL_KEYPOINTS_CORRECTED],
            "decision": None,
            "y_ordering_error": "no_valid_candidate",
        }
    return best_meta


def _normalize_payload(data: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Normalize payload to a dict with a mutable frames list.
    Accepts:
      - {"predictions":{"frames":[...]}}
      - {"frames":[...]}
      - [...]
    """
    if isinstance(data, list):
        out = {"frames": data}
        return out, out["frames"]
    if not isinstance(data, dict):
        raise ValueError("Unsupported payload type. Expected dict or list.")

    if isinstance(data.get("frames"), list):
        return data, data["frames"]

    preds = data.get("predictions")
    if isinstance(preds, dict) and isinstance(preds.get("frames"), list):
        return data, preds["frames"]

    data["frames"] = []
    return data, data["frames"]


def _frame_id(frame: dict[str, Any], default: int) -> int:
    fid = frame.get("frame_id", frame.get("frame_number", default))
    try:
        return int(fid)
    except Exception:
        return int(default)


def _label_to_int(label: Any) -> int | None:
    if label is None:
        return None
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return int(digits) if digits else None


def _label_to_str(label_num: int | None) -> str | None:
    if label_num is None:
        return None
    return f"kpv{int(label_num):02d}"


def _extract_labeled_points(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return points as:
      [{"id": int, "x": float|None, "y": float|None, "label": "kpvNN"|None}, ...]
    Prefer keypoints_labeled when present; otherwise build from keypoints + labels.
    """
    labeled = frame.get("keypoints_labeled")
    if isinstance(labeled, list) and labeled:
        out: list[dict[str, Any]] = []
        for item in labeled:
            if not isinstance(item, dict):
                continue
            idx = item.get("id")
            x = item.get("x")
            y = item.get("y")
            lab = item.get("label")
            try:
                idx_i = int(idx) if idx is not None else -1
            except Exception:
                idx_i = -1
            out.append(
                {
                    "id": idx_i,
                    "x": None if x is None else float(x),
                    "y": None if y is None else float(y),
                    "label": None if lab is None else str(lab),
                }
            )
        return out

    keypoints = frame.get("keypoints") or []
    labels = frame.get("labels") or frame.get("keypoints_labels") or []
    out = []
    for i, pt in enumerate(keypoints):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x = float(pt[0])
        y = float(pt[1])
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            continue
        lab_num = None
        if i < len(labels):
            try:
                lab_num = int(labels[i]) if labels[i] is not None else None
            except Exception:
                lab_num = _label_to_int(labels[i])
        out.append({"id": i, "x": x, "y": y, "label": _label_to_str(lab_num)})
    return out


def _infer_frame_size(frames: list[dict[str, Any]]) -> tuple[int, int]:
    max_x = 0.0
    max_y = 0.0
    for frame in frames:
        for kp in _extract_labeled_points(frame):
            x = kp.get("x")
            y = kp.get("y")
            if x is None or y is None:
                continue
            max_x = max(max_x, float(x))
            max_y = max(max_y, float(y))
    # +1 because coords are pixel indices.
    width = max(1, int(math.ceil(max_x + 1.0)))
    height = max(1, int(math.ceil(max_y + 1.0)))
    return width, height


def _get_frame_size_from_video(video_path: Path) -> tuple[int, int]:
    """
    Read video dimensions from the first frame metadata.
    """
    # Prefer OpenCV if available.
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                cap.release()
            if width > 0 and height > 0:
                return width, height
    except Exception:
        pass

    # Fallback to ffprobe so script works without cv2.
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        obj = json.loads(proc.stdout or "{}")
        streams = obj.get("streams") or []
        if streams:
            width = int(streams[0].get("width") or 0)
            height = int(streams[0].get("height") or 0)
            if width > 0 and height > 0:
                return width, height
    except Exception as exc:
        raise RuntimeError(
            "Failed to read frame size from video. Install OpenCV (cv2) or ffprobe."
        ) from exc

    raise RuntimeError(f"Could not read valid frame size from video: {video_path}")


def _get_fps_from_video(video_path: Path) -> float:
    """Read video frame rate (fps) via ffprobe. Required for segment extraction."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    obj = json.loads(proc.stdout or "{}")
    streams = obj.get("streams") or []
    if not streams:
        return 30.0
    rate = streams[0].get("r_frame_rate") or "30/1"
    if isinstance(rate, (int, float)):
        return float(rate)
    # e.g. "30/1" or "30000/1001"
    parts = str(rate).strip().split("/")
    if len(parts) == 2:
        try:
            num, den = float(parts[0]), float(parts[1])
            if den > 0:
                return num / den
        except ValueError:
            pass
    return 30.0


def _write_segment_video(
    source_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> None:
    """Extract frames [start_frame, end_frame] (inclusive) into a new video file using ffmpeg."""
    n_frames = end_frame - start_frame + 1
    if n_frames <= 0:
        return
    # Frame-accurate: select filter. setpts keeps output timing at same fps.
    fps_val = round(fps, 6) if fps > 0 else 30.0
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        f"select=between(n\\,{start_frame}\\,{end_frame}),setpts=N/{fps_val}/TB",
        "-vsync",
        "cfr",
        "-r",
        str(fps_val),
        "-an",  # drop audio for segment clips
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _output_segment_videos(
    video_path: Path,
    segments: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
) -> list[Path]:
    """Write one video per segment; return list of output paths. Requires ffmpeg."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[WARN] ffmpeg not found; skipping segment video output.")
        return []
    try:
        fps = _get_fps_from_video(video_path)
    except Exception as e:
        print(f"[WARN] Could not get FPS from video ({video_path}), skipping segment videos: {e}")
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for seg in segments:
        seg_id = seg.get("segment_id", len(written))
        start_frame = int(seg.get("start_frame", 0))
        end_frame = int(seg.get("end_frame", 0))
        out_name = f"{stem}_segment_{seg_id:03d}_frames_{start_frame}-{end_frame}.mp4"
        out_path = output_dir / out_name
        try:
            _write_segment_video(video_path, out_path, start_frame, end_frame, fps)
            written.append(out_path)
            print(f"Wrote segment video: {out_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[WARN] Failed to write segment {seg_id}: {e}")
    return written


def _save_debug_keypoints_image(
    *,
    out_path_base: Path,
    frame_bgr: Any,
    annotations: list[tuple[float, float, str]],
) -> None:
    """Save frame image with keypoint markers + red bold text."""
    try:
        import cv2  # type: ignore
    except Exception:
        return

    if frame_bgr is None:
        return
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    for x_f, y_f, text in annotations:
        x = int(round(float(x_f)))
        y = int(round(float(y_f)))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        cv2.circle(img, (x, y), 4, (0, 255, 255), thickness=-1)  # yellow point
        text_pos = (max(0, x + 6), max(16, y - 6))
        # black outline for readability, then red bold text
        cv2.putText(
            img,
            text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),  # red
            2,
            cv2.LINE_AA,
        )
    png_path = out_path_base.with_suffix(".png")
    cv2.imwrite(str(png_path), img)


def _load_frame_bgr(video_path: Path, frame_id: int) -> Any:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def _in_border(x: float, y: float, width: int, height: int, margin: float) -> bool:
    return bool(x < margin or x >= (float(width) - margin) or y < margin or y >= (float(height) - margin))


def _apply_border_filter(
    frames: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    margin: float,
) -> dict[int, int]:
    """
    Zero-out/remove points within image border in all known keypoint containers.
    Returns per-frame removed count.
    """
    removed_by_frame: dict[int, int] = {}

    for index, frame in enumerate(frames):
        fid = _frame_id(frame, index)
        removed = 0

        # 1) keypoints + labels arrays
        keypoints = frame.get("keypoints")
        labels = frame.get("labels")
        if isinstance(keypoints, list):
            for i, pt in enumerate(keypoints):
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                try:
                    x = float(pt[0])
                    y = float(pt[1])
                except Exception:
                    continue
                if abs(x) < 1e-6 and abs(y) < 1e-6:
                    continue
                if _in_border(x, y, width, height, margin):
                    keypoints[i] = [0.0, 0.0]
                    if isinstance(labels, list) and i < len(labels):
                        labels[i] = 0
                    removed += 1

        # 2) keypoints_labeled list
        labeled = frame.get("keypoints_labeled")
        if isinstance(labeled, list):
            for item in labeled:
                if not isinstance(item, dict):
                    continue
                x = item.get("x")
                y = item.get("y")
                if x is None or y is None:
                    continue
                try:
                    xf = float(x)
                    yf = float(y)
                except Exception:
                    continue
                if _in_border(xf, yf, width, height, margin):
                    item["x"] = None
                    item["y"] = None
                    item["label"] = None
                    removed += 1

        removed_by_frame[fid] = removed

    _debug(
        "Border filtering completed: "
        f"margin={margin}px, frame_size=({width}x{height}), "
        f"total_removed={sum(removed_by_frame.values())}"
    )
    return removed_by_frame


def _avg_distance_same_label(
    cur_labeled: list[dict[str, Any]],
    prev_labeled: list[dict[str, Any]],
) -> tuple[float, int, list[tuple[int, int, float]]]:
    """
    Copied behavior from OLD_FILE:
    - group by label
    - greedy nearest match per label
    - return (average_distance, matched_count, matches)
      matches items are (cur_idx, prev_idx, distance),
      with prev_idx=-1 and distance=inf when unmatched.
    """
    prev_by_label: dict[int, list[tuple[int, float, float]]] = {}
    for idx_prev, item in enumerate(prev_labeled or []):
        lx = item.get("x")
        ly = item.get("y")
        lab = _label_to_int(item.get("label"))
        if lab is None or lx is None or ly is None:
            continue
        prev_by_label.setdefault(lab, []).append((idx_prev, float(lx), float(ly)))

    cur_by_label: dict[int, list[tuple[int, float, float]]] = {}
    for idx_cur, item in enumerate(cur_labeled or []):
        cx = item.get("x")
        cy = item.get("y")
        lab = _label_to_int(item.get("label"))
        if lab is None or cx is None or cy is None:
            continue
        cur_by_label.setdefault(lab, []).append((idx_cur, float(cx), float(cy)))

    total = 0.0
    matched_count = 0
    matches: list[tuple[int, int, float]] = []
    valid_cur = 0

    for label, cur_list in cur_by_label.items():
        prev_list = prev_by_label.get(label, [])
        if not prev_list:
            for cur_idx, _, _ in cur_list:
                matches.append((cur_idx, -1, float("inf")))
                total += float("inf")
                valid_cur += 1
            continue

        cur_remaining = cur_list.copy()
        prev_remaining = prev_list.copy()
        used_prev_indices = set()

        while cur_remaining and prev_remaining:
            min_dist = float("inf")
            best_cur_idx = None
            best_prev_idx = None

            for cur_idx, cx, cy in cur_remaining:
                for prev_idx, px, py in prev_remaining:
                    if prev_idx in used_prev_indices:
                        continue
                    dist = float(math.hypot(cx - px, cy - py))
                    if dist < min_dist:
                        min_dist = dist
                        best_cur_idx = cur_idx
                        best_prev_idx = prev_idx

            if best_cur_idx is not None and best_prev_idx is not None:
                matches.append((best_cur_idx, best_prev_idx, min_dist))
                total += min_dist
                matched_count += 1
                valid_cur += 1
                cur_remaining = [(idx, x, y) for idx, x, y in cur_remaining if idx != best_cur_idx]
                used_prev_indices.add(best_prev_idx)
                prev_remaining = [(idx, x, y) for idx, x, y in prev_remaining if idx != best_prev_idx]
            else:
                break

        for cur_idx, _, _ in cur_remaining:
            matches.append((cur_idx, -1, float("inf")))
            total += float("inf")
            valid_cur += 1

    for idx_cur, item in enumerate(cur_labeled or []):
        cx = item.get("x")
        cy = item.get("y")
        lab = _label_to_int(item.get("label"))
        if cx is None or cy is None:
            continue
        if lab is None and not any(m[0] == idx_cur for m in matches):
            matches.append((idx_cur, -1, float("inf")))
            total += float("inf")
            valid_cur += 1

    if valid_cur == 0:
        return float("inf"), 0, matches
    return total / float(valid_cur), matched_count, matches


def _is_similar_frame_pair(
    cur_labeled: list[dict[str, Any]],
    prev_labeled: list[dict[str, Any]],
    *,
    dist_threshold: float,
    min_matches: int,
) -> tuple[bool, dict[str, Any]]:
    avg_dist, _match_count, matches = _avg_distance_same_label(cur_labeled, prev_labeled)
    common_matches = [(ci, pi, d) for (ci, pi, d) in matches if pi != -1 and d != float("inf")]
    all_dist_ok = bool(common_matches) and all(d < dist_threshold for _, _, d in common_matches)
    similar = len(common_matches) >= min_matches and all_dist_ok

    max_dist = max((d for _, _, d in common_matches), default=None)
    fail_reasons: list[str] = []
    if len(common_matches) < min_matches:
        fail_reasons.append(
            f"common_matches {len(common_matches)} < required {min_matches}"
        )
    if not common_matches:
        fail_reasons.append("no common matches with same label")
    elif max_dist is not None and max_dist >= dist_threshold:
        fail_reasons.append(
            f"max_distance {max_dist:.2f}px >= threshold {dist_threshold:.2f}px"
        )

    stats = {
        "avg_distance": avg_dist,
        "common_matches": len(common_matches),
        "max_distance": max_dist,
        "threshold_distance": dist_threshold,
        "threshold_min_matches": min_matches,
        "passed": similar,
        "fail_reasons": fail_reasons,
    }
    return similar, stats


def _build_bidirectional_similarity_report(
    ordered_items: list[tuple[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    For each adjacent pair (A,B):
      - forward: B vs A
      - backward: A vs B
    A pair is connected only if both pass.
    Then build contiguous frame segments from connected pairs.
    """
    pair_results: list[dict[str, Any]] = []
    if len(ordered_items) <= 1:
        return pair_results, [{"segment_id": 0, "start_frame": ordered_items[0][0], "end_frame": ordered_items[0][0], "frame_count": 1}] if ordered_items else []

    for idx in range(len(ordered_items) - 1):
        left_id, left_frame = ordered_items[idx]
        right_id, right_frame = ordered_items[idx + 1]

        left_labeled = _extract_labeled_points(left_frame)
        right_labeled = _extract_labeled_points(right_frame)

        f_ok, f_stats = _is_similar_frame_pair(
            right_labeled,
            left_labeled,
            dist_threshold=SIMILARITY_DIST_THRESHOLD_PX,
            min_matches=SIMILARITY_MIN_COMMON_MATCHES,
        )
        b_ok, b_stats = _is_similar_frame_pair(
            left_labeled,
            right_labeled,
            dist_threshold=SIMILARITY_DIST_THRESHOLD_PX,
            min_matches=SIMILARITY_MIN_COMMON_MATCHES,
        )

        pair_connected = bool(f_ok and b_ok)
        pair_results.append(
            {
                "left_frame_id": left_id,
                "right_frame_id": right_id,
                "forward_pass": f_ok,
                "backward_pass": b_ok,
                "pair_connected": pair_connected,
                "forward_stats": f_stats,
                "backward_stats": b_stats,
            }
        )
        if not f_ok:
            _debug(
                f"Pair {left_id}->{right_id} forward FAILED: "
                + ", ".join(f_stats.get("fail_reasons") or ["unknown"])
            )
        if not b_ok:
            _debug(
                f"Pair {left_id}->{right_id} backward FAILED: "
                + ", ".join(b_stats.get("fail_reasons") or ["unknown"])
            )

    segments: list[dict[str, Any]] = []
    seg_start_idx = 0
    seg_id = 0

    for edge_idx, edge in enumerate(pair_results):
        is_last_edge = edge_idx == len(pair_results) - 1
        should_split = not edge["pair_connected"]
        if should_split:
            start_frame = ordered_items[seg_start_idx][0]
            end_frame = ordered_items[edge_idx][0]
            segments.append(
                {
                    "segment_id": seg_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_count": int((edge_idx - seg_start_idx) + 1),
                }
            )
            seg_id += 1
            seg_start_idx = edge_idx + 1
        if is_last_edge:
            start_frame = ordered_items[seg_start_idx][0]
            end_frame = ordered_items[edge_idx + 1][0]
            segments.append(
                {
                    "segment_id": seg_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_count": int((edge_idx + 1 - seg_start_idx) + 1),
                }
            )

    return pair_results, segments


def run_conversion(
    *,
    input_json: Path,
    output_json: Path,
    report_json: Path,
    video_url: Path | None,
    frame_width: int | None,
    frame_height: int | None,
    border_margin_px: float,
) -> None:
    raw = json.loads(input_json.read_text())
    payload, frames = _normalize_payload(raw)

    if not frames:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2))
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(
            json.dumps(
                {
                    "input_json": str(input_json),
                    "output_json": str(output_json),
                    "frame_count": 0,
                    "border_filter": {"margin_px": border_margin_px, "frame_width": frame_width, "frame_height": frame_height, "removed_by_frame": {}},
                    "pair_similarity": [],
                    "segments": [],
                },
                indent=2,
            )
        )
        return

    if frame_width is None or frame_height is None:
        if video_url is not None:
            v_w, v_h = _get_frame_size_from_video(video_url)
            frame_width = v_w if frame_width is None else frame_width
            frame_height = v_h if frame_height is None else frame_height
            _debug(f"Frame size from video: {frame_width}x{frame_height}")
        else:
            inf_w, inf_h = _infer_frame_size(frames)
            frame_width = inf_w if frame_width is None else frame_width
            frame_height = inf_h if frame_height is None else frame_height
            _debug(f"Inferred frame size from keypoints: {frame_width}x{frame_height}")

    removed_by_frame = _apply_border_filter(
        frames,
        width=int(frame_width),
        height=int(frame_height),
        margin=float(border_margin_px),
    )

    ordered_items = sorted(((_frame_id(fr, idx), fr) for idx, fr in enumerate(frames)), key=lambda x: x[0])
    pair_similarity, segments = _build_bidirectional_similarity_report(ordered_items)

    # Segment-aware Step 1-4 processing (ported from OLD_FILE in lightweight form).
    _require_numpy()
    frame_lookup: dict[int, dict[str, Any]] = {int(fid): fr for fid, fr in ordered_items}
    step1_outputs: list[dict[str, Any]] = []
    step2_outputs: list[dict[str, Any]] = []
    step3_outputs: list[dict[str, Any]] = []
    step4_outputs: list[dict[str, Any]] = []
    debug_unordered_dir = output_json.parent / "debug_keypoints" / "unordered"
    debug_ordered_dir = output_json.parent / "debug_keypoints" / "ordered"
    if DEBUG_FLAG:
        debug_unordered_dir.mkdir(parents=True, exist_ok=True)
        debug_ordered_dir.mkdir(parents=True, exist_ok=True)
    total_step_frames = sum(int(s.get("frame_count", 0)) for s in segments)
    processed_step_frames = 0
    for seg in segments:
        seg_id = int(seg.get("segment_id", -1))
        start_f = int(seg.get("start_frame", 0))
        end_f = int(seg.get("end_frame", -1))
        if DEBUG_FLAG:
            _debug(f"Processing segment {seg_id}: frames {start_f}-{end_f}")
        for fid in range(start_f, end_f + 1):
            processed_step_frames += 1
            frame_obj = frame_lookup.get(fid)
            if frame_obj is None:
                continue
            if DEBUG_FLAG and (processed_step_frames == 1 or processed_step_frames % 10 == 0 or fid == end_f):
                print(
                    f"[DEBUG] StepPipeline progress: {processed_step_frames}/{total_step_frames} "
                    f"(segment={seg_id}, frame={fid})"
                )
            kps, labels = _extract_frame_kps_labels(frame_obj)
            t0 = time.perf_counter() if DEBUG_FLAG else 0.0
            step1 = _step1_build_connections_light(
                kps=kps,
                labels=labels,
                frame_number=int(fid),
                frame_width=int(frame_width),
                frame_height=int(frame_height),
            )
            t1 = time.perf_counter() if DEBUG_FLAG else 0.0
            step2 = _step2_build_four_point_groups_light(step1, frame_number=int(fid))
            t2 = time.perf_counter() if DEBUG_FLAG else 0.0
            step3 = _step3_build_ordered_candidates_light(step2, frame_number=int(fid))
            t3 = time.perf_counter() if DEBUG_FLAG else 0.0
            step4 = _step4_pick_best_candidate_light(step3, frame_number=int(fid))
            if DEBUG_FLAG and (processed_step_frames % 10 == 0 or fid == end_f):
                s1 = (t1 - t0) * 1000.0
                s2 = (t2 - t1) * 1000.0
                s3 = (t3 - t2) * 1000.0
                s4 = (time.perf_counter() - t3) * 1000.0
                print(
                    f"[DEBUG] StepPipeline timings frame={fid}: "
                    f"s1={s1:.1f}ms s2={s2:.1f}ms s3={s3:.1f}ms s4={s4:.1f}ms"
                )
            step4["segment_id"] = seg_id

            # Write ordered keypoints back into frame output.
            frame_obj["keypoints"] = step4.get("reordered_keypoints") or []
            frame_obj["ordered_keypoints"] = step4.get("reordered_keypoints") or []

            if DEBUG_FLAG:
                unordered_base = debug_unordered_dir / f"frame_{int(fid):05d}_unordered"
                ordered_base = debug_ordered_dir / f"frame_{int(fid):05d}_ordered"
                frame_bgr = _load_frame_bgr(video_url, int(fid)) if video_url is not None else None
                unordered_labeled = _extract_labeled_points(frame_obj)
                unordered_annotations: list[tuple[float, float, str]] = []
                for item in unordered_labeled:
                    x = item.get("x")
                    y = item.get("y")
                    if x is None or y is None:
                        continue
                    kp_id = int(item.get("id", -1))
                    kp_lab = str(item.get("label"))
                    unordered_annotations.append((float(x), float(y), f"id:{kp_id} {kp_lab}"))

                ordered_annotations: list[tuple[float, float, str]] = []
                ordered_kps = step4.get("reordered_keypoints") or []
                for ord_idx, pt in enumerate(ordered_kps):
                    if not pt or len(pt) < 2:
                        continue
                    x = float(pt[0])
                    y = float(pt[1])
                    if abs(x) < 1e-6 and abs(y) < 1e-6:
                        continue
                    tpl_lab = int(KEYPOINT_LABELS[ord_idx]) if ord_idx < len(KEYPOINT_LABELS) else 0
                    ordered_annotations.append((x, y, f"ord:{ord_idx} kpv{tpl_lab:02d}"))

                _save_debug_keypoints_image(
                    out_path_base=unordered_base,
                    frame_bgr=frame_bgr,
                    annotations=unordered_annotations,
                )
                _save_debug_keypoints_image(
                    out_path_base=ordered_base,
                    frame_bgr=frame_bgr,
                    annotations=ordered_annotations,
                )

            step1_outputs.append(step1)
            step2_outputs.append(step2)
            step3_outputs.append(step3)
            step4_outputs.append(step4)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    report = {
        "input_json": str(input_json),
        "output_json": str(output_json),
        "frame_count": len(frames),
        "border_filter": {
            "margin_px": border_margin_px,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "removed_by_frame": removed_by_frame,
            "removed_total": int(sum(removed_by_frame.values())),
        },
        "pair_similarity": pair_similarity,
        "segments": segments,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2))

    # Step output dumps (aligned with OLD_FILE naming).
    step1_path = output_json.parent / "keypoint_step_1_pairs.json"
    step2_path = output_json.parent / "keypoint_step_2_four_points.json"
    step3_path = output_json.parent / "keypoint_step_3_four_points_orderd.json"
    step4_path = output_json.parent / "keypoint_step_4_best_points.json"
    step1_path.write_text(json.dumps(step1_outputs, indent=2))
    step2_path.write_text(json.dumps(step2_outputs, indent=2))
    step3_path.write_text(json.dumps(step3_outputs, indent=2))
    step4_path.write_text(json.dumps(step4_outputs, indent=2))

    print(f"Wrote filtered payload: {output_json}")
    print(f"Wrote segment report: {report_json}")
    print(f"Wrote step outputs: {step1_path}, {step2_path}, {step3_path}, {step4_path}")
    if DEBUG_FLAG:
        print(
            f"Wrote debug keypoint images: {debug_unordered_dir} and {debug_ordered_dir}"
        )
    print(f"Detected segments: {len(segments)}")

    # Output one video per segment when video URL is provided
    if video_url is not None and segments and DEBUG_FLAG:
        stem = input_json.stem
        segment_videos_dir = output_json.parent / f"{stem}_segments"
        _output_segment_videos(video_url, segments, segment_videos_dir, stem)
    elif video_url is not None and segments and not DEBUG_FLAG:
        print("Skipping segment video output because DEBUG_FLAG is False.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone keypoint conversion (border filtering + bidirectional frame similarity segmentation)."
    )
    parser.add_argument("--video-url", type=Path, default=None, help="Video path used to read frame width/height automatically.")
    parser.add_argument("--unsorted-json", "--input-json", dest="unsorted_json", type=Path, required=True, help="Unordered keypoints JSON path (same schema as OLD_FILE).")
    parser.add_argument("--output-json", type=Path, default=None, help="Output path for filtered payload JSON. Default: opt_v4/<unsorted_name>.filtered.json")
    parser.add_argument("--report-json", type=Path, default=None, help="Output path for similarity/segment report JSON. Default: opt_v4/<unsorted_name>.segment_report.json")
    parser.add_argument("--frame-width", type=int, default=None, help="Frame width. If omitted, inferred from keypoints.")
    parser.add_argument("--frame-height", type=int, default=None, help="Frame height. If omitted, inferred from keypoints.")
    parser.add_argument("--border-margin-px", type=float, default=BORDER_MARGIN_PX, help="Border margin in pixels to remove keypoints.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    unsorted_json: Path = args.unsorted_json
    default_stem = unsorted_json.stem
    output_json: Path = args.output_json or (Path("opt_v4") / f"{default_stem}.filtered.json")
    report_json: Path = args.report_json or (Path("opt_v4") / f"{default_stem}.segment_report.json")

    run_conversion(
        input_json=unsorted_json,
        output_json=output_json,
        report_json=report_json,
        video_url=args.video_url,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        border_margin_px=float(args.border_margin_px),
    )


if __name__ == "__main__":
    main()
