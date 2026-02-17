#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any


DEBUG_FLAG = True
BORDER_MARGIN_PX = 50.0
SIMILARITY_DIST_THRESHOLD_PX = 30.0
SIMILARITY_MIN_COMMON_MATCHES = 3


def _debug(msg: str) -> None:
    if DEBUG_FLAG:
        print(f"[DEBUG] {msg}")


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

    print(f"Wrote filtered payload: {output_json}")
    print(f"Wrote segment report: {report_json}")
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
