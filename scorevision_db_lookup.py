#!/usr/bin/env python3
"""
Look up sub-URLs by two HOTKEYs (me, other) from Scorevision R2 index databases.
Finds last N sub-URLs per DB for each hotkey, fetches evaluation JSONs,
and reports comparison of evaluation.score for common task_ids.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# --- Hardcoded: change these and run the script ---
# 5HE5eQUC5zd9G87WazpzFjYeoE2LQg8hKt9y88hsBhquLYz5
# 5C88LjaemXPRWREGDyF4g7VN7qnnuBQuhwFnSUMK9sfbiqJU
HOTKEY_ME = "5HE5eQUC5zd9G87WazpzFjYeoE2LQg8hKt9y88hsBhquLYz5"
# 5Ca2CVevqG4s7iu3vLMNq2e2tGz9PMLLbzLeF9Pqneh4zsrm
# 5HdX72NK2sdUes3UMvUy37EJHNKkuGuu6w2ratvWLvki4ab8
HOTKEY_OTHER = "5Ca2CVevqG4s7iu3vLMNq2e2tGz9PMLLbzLeF9Pqneh4zsrm"
N = 5  # From each DB, take the last N sub-URLs (search from the end of the index)

# Database index URLs
DB_INDEX_URLS = [
    "https://pub-b2fdfa5b7a5344f384b6f6015a19a24c.r2.dev/scorevision/index.json",
    "https://pub-7b4130b6af75472f800371248bca15b6.r2.dev/scorevision/index.json",
    "https://pub-5c8278d7febb4036b4bc7062c4c828c0.r2.dev/scorevision/index.json",
    "https://pub-76c893c4f28248be963f7571f3d8ffa9.r2.dev/scorevision/index.json",
]


def get_base_url(index_url: str) -> str:
    """Strip '/scorevision/index.json' to get the base URL for building full URLs."""
    if index_url.endswith("/scorevision/index.json"):
        return index_url[: -len("/scorevision/index.json")] + "/"
    # Fallback: strip last path component
    return index_url.rsplit("/", 1)[0] + "/"


def fetch_index(index_url: str) -> List[str]:
    """Fetch index JSON from URL and return the array of sub-URL strings."""
    req = Request(index_url, headers={"User-Agent": "scorevision-db-lookup/1.0"})
    with urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")
    return [str(item) for item in data]


def find_suburls_with_hotkey(suburls: List[str], hotkey: str, n: int) -> List[str]:
    """Filter sub-URLs that contain hotkey; return the last N from the list (search from the end)."""
    if n <= 0:
        return []
    matching = [s for s in suburls if hotkey in s]
    return matching[-n:] if len(matching) >= n else matching


def fetch_json(url: str) -> Any:
    """Fetch URL and parse as JSON."""
    req = Request(url, headers={"User-Agent": "scorevision-db-lookup/1.0"})
    with urlopen(req, timeout=60) as resp:
        return json.load(resp)


def extract_payload_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract task_id, run.*, evaluation.keypoints.*, evaluation.objects.*, evaluation.score."""
    run = payload.get("run") or {}
    ev = payload.get("evaluation") or {}
    acc_breakdown = ev.get("acc_breakdown") or {}
    keypoints = acc_breakdown.get("keypoints") or {}
    # JSON uses "objects" (plural); support "object" as fallback
    objects = acc_breakdown.get("objects") or acc_breakdown.get("object") or {}

    latency_ms = run.get("latency_ms")
    if latency_ms is not None:
        try:
            latency_ms = int(round(float(latency_ms)))
        except (TypeError, ValueError):
            latency_ms = 0
    else:
        latency_ms = 0

    def _round4(v: Any) -> Any:
        if v is None:
            return None
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return v

    obj_bbox = objects.get("bbox_placement")
    obj_cat = objects.get("categorisation")
    obj_team = objects.get("team")
    obj_enum = objects.get("enumeration")
    obj_track = objects.get("tracking_stability")
    obj_scores = [obj_bbox, obj_cat, obj_team, obj_enum, obj_track]
    obj_nums = [float(x) for x in obj_scores if x is not None]
    objects_avg = round(sum(obj_nums) / len(obj_nums), 4) if obj_nums else None

    return {
        "task_id": payload.get("task_id"),
        "run.error": run.get("error"),
        "run.responses_key": run.get("responses_key"),
        "run.latency_ms": latency_ms,
        "evaluation.keypoints.floor_markings_alignment": _round4(keypoints.get("floor_markings_alignment")),
        "evaluation.objects.bbox_placement": _round4(obj_bbox),
        "evaluation.objects.categorisation": _round4(obj_cat),
        "evaluation.objects.team": _round4(obj_team),
        "evaluation.objects.enumeration": _round4(obj_enum),
        "evaluation.objects.tracking_stability": _round4(obj_track),
        "evaluation.objects.score": objects_avg,
        "evaluation.score": _round4(ev.get("score")),
    }


def collect_url_pairs(hotkey: str, n: int) -> List[Tuple[str, str]]:
    """Collect (evaluation full URL, base URL) for the last N sub-URLs from each DB that contain hotkey."""
    url_pairs: List[Tuple[str, str]] = []
    for index_url in DB_INDEX_URLS:
        base_url = get_base_url(index_url)
        try:
            suburls = fetch_index(index_url)
        except (URLError, HTTPError, ValueError, json.JSONDecodeError):
            continue
        last_n = find_suburls_with_hotkey(suburls, hotkey, n)
        for sub in last_n:
            full_url = base_url + sub if not sub.startswith("http") else sub
            url_pairs.append((full_url, base_url))
    return url_pairs


def fetch_task_scores(
    url_pairs: List[Tuple[str, str]], logger: logging.Logger, label: str
) -> Dict[int, Dict[str, Any]]:
    """
    Fetch each evaluation JSON from url_pairs; return task_id -> {score, evaluation_url, ...}.
    If same task_id appears multiple times, later entry overwrites.
    """
    by_task: Dict[int, Dict[str, Any]] = {}
    for full_url, base_url in url_pairs:
        try:
            data = fetch_json(full_url)
        except (URLError, HTTPError, ValueError, json.JSONDecodeError) as e:
            logger.info("[%s] ERROR fetching %s: %s", label, full_url[:80], e)
            continue
        if not isinstance(data, list) or len(data) == 0:
            continue
        payload = data[0].get("payload") if isinstance(data[0], dict) else None
        if not payload:
            continue
        fields = extract_payload_fields(payload)
        task_id = fields.get("task_id")
        if task_id is None:
            continue
        try:
            task_id = int(task_id)
        except (TypeError, ValueError):
            continue
        score = fields.get("evaluation.score")
        video_url = None
        responses_key = fields.get("run.responses_key")
        if responses_key and isinstance(responses_key, str):
            try:
                responses_full = (base_url + responses_key) if not responses_key.startswith("http") else responses_key
                resp_data = fetch_json(responses_full)
                if isinstance(resp_data, dict):
                    video_url = resp_data.get("video_url")
            except (URLError, HTTPError, ValueError, json.JSONDecodeError):
                pass
        by_task[task_id] = {
            "score": score,
            "evaluation_url": full_url,
            "base_url": base_url,
            "fields": fields,
            "video_url": video_url,
        }
    return by_task


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation scores for two hotkeys on common task_ids."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="scorevision_db_lookup.log",
        help="Log file path (default: scorevision_db_lookup.log)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log to stderr as well",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--keypoints",
        action="store_true",
        help="Compare keypoint scores (floor_markings_alignment) instead of total",
    )
    mode_group.add_argument(
        "--bbox",
        action="store_true",
        help="Compare bbox (objects) scores and show all 5 factor scores",
    )
    args = parser.parse_args()

    compare_mode = "bbox" if args.bbox else ("keypoints" if args.keypoints else "total")

    hotkey_me = HOTKEY_ME.strip()
    hotkey_other = HOTKEY_OTHER.strip()
    n = max(0, N)
    log_path = args.output

    # Setup file logging
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter("%(message)s")
    file_handler.setFormatter(file_fmt)

    logger = logging.getLogger("scorevision_lookup")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    if args.verbose:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.INFO)
        console.setFormatter(file_fmt)
        logger.addHandler(console)

    logger.info("")
    logger.info("=== Scorevision DB lookup %s ===", datetime.now(timezone.utc).isoformat())
    logger.info("HOTKEY_ME:    %s", hotkey_me)
    logger.info("HOTKEY_OTHER: %s", hotkey_other)
    logger.info("N per DB:     %d (last N links from each database)", n)
    logger.info("Compare mode: %s", compare_mode)
    logger.info("")

    # Collect URL pairs for both hotkeys
    logger.info("--- Collecting evaluation URLs ---")
    url_pairs_me = collect_url_pairs(hotkey_me, n)
    url_pairs_other = collect_url_pairs(hotkey_other, n)
    logger.info("Me:    %d evaluation URLs", len(url_pairs_me))
    logger.info("Other: %d evaluation URLs", len(url_pairs_other))
    logger.info("")

    # Fetch evaluation JSONs and build task_id -> record
    logger.info("--- Fetching evaluation JSONs ---")
    me_by_task = fetch_task_scores(url_pairs_me, logger, "me")
    other_by_task = fetch_task_scores(url_pairs_other, logger, "other")
    logger.info("Me:    %d tasks with scores", len(me_by_task))
    logger.info("Other: %d tasks with scores", len(other_by_task))
    logger.info("")

    # Common task_ids
    common_task_ids = sorted(set(me_by_task) & set(other_by_task))
    logger.info("--- Comparison report (common task_id) ---")
    logger.info("Common task_ids: %d", len(common_task_ids))
    logger.info("")

    # When no common tasks, output simple one-line per task so user can manually check
    if not common_task_ids:
        def _score4(v):
            return "%.4f" % v if v is not None else "None"
        logger.info("--- Me: all task_ids (score, evaluation_url) ---")
        for tid in sorted(me_by_task.keys()):
            rec = me_by_task[tid]
            logger.info("  task_id %s  score=%s  %s", tid, _score4(rec["score"]), rec.get("evaluation_url", ""))
        logger.info("")
        logger.info("--- Other: all task_ids (score, evaluation_url) ---")
        for tid in sorted(other_by_task.keys()):
            rec = other_by_task[tid]
            logger.info("  task_id %s  score=%s  %s", tid, _score4(rec["score"]), rec.get("evaluation_url", ""))
        zero_me = [(tid, rec) for tid, rec in me_by_task.items() if rec.get("score") is None or rec.get("score") == 0]
        zero_other = [(tid, rec) for tid, rec in other_by_task.items() if rec.get("score") is None or rec.get("score") == 0]
        if zero_me or zero_other:
            logger.info("")
            logger.info("--- 0-scoring cases ---")
            for tid, rec in sorted(zero_me, key=lambda x: x[0]):
                err = (rec.get("fields") or {}).get("run.error")
                logger.info("  me   task_id %s  %s  %s", tid, ("[Error] %s" % err) if err else "", rec.get("evaluation_url", ""))
            for tid, rec in sorted(zero_other, key=lambda x: x[0]):
                err = (rec.get("fields") or {}).get("run.error")
                logger.info("  other task_id %s  %s  %s", tid, ("[Error] %s" % err) if err else "", rec.get("evaluation_url", ""))
        logger.info("")
        logger.info("--- Summary ---")
        logger.info("Common tasks: 0 (no overlap - see above for full task lists)")
        logger.info("")
        logger.info("Log file: %s", log_path)
        return 0

    me_wins = 0
    other_wins = 0
    ties = 0
    sum_me = 0.0
    sum_other = 0.0
    valid_count = 0  # tasks where both sides have non-zero score (excluded from average/wins)

    for task_id in common_task_ids:
        me_rec = me_by_task[task_id]
        other_rec = other_by_task[task_id]
        me_f = me_rec.get("fields") or {}
        other_f = other_rec.get("fields") or {}
        # Score to compare depends on mode
        if compare_mode == "keypoints":
            score_me = me_f.get("evaluation.keypoints.floor_markings_alignment")
            score_other = other_f.get("evaluation.keypoints.floor_markings_alignment")
        elif compare_mode == "bbox":
            score_me = me_f.get("evaluation.objects.score")
            score_other = other_f.get("evaluation.objects.score")
        else:
            score_me = me_rec["score"]
            score_other = other_rec["score"]
        # Compare; treat None as -1 so it loses
        s_me = score_me if score_me is not None else -1.0
        s_other = score_other if score_other is not None else -1.0
        both_valid = (score_me is not None and score_me != 0 and
                      score_other is not None and score_other != 0)
        if both_valid:
            if s_me > s_other:
                winner = "me"
                me_wins += 1
            elif s_other > s_me:
                winner = "other"
                other_wins += 1
            else:
                winner = "tie"
                ties += 1
            sum_me += s_me
            sum_other += s_other
            valid_count += 1
        else:
            winner = "me" if s_me > s_other else ("other" if s_other > s_me else "tie")

        # Outcome: You win / You lose / Tie (me = you) with icon
        outcome = ("✅ You win" if winner == "me" else
                   "❌ You lose" if winner == "other" else
                   "🤝 Tie")
        logger.info("task_id %s  ->  %s", task_id, outcome)
        me_responses_key = me_f.get("run.responses_key")
        other_responses_key = other_f.get("run.responses_key")
        me_responses_url = (me_rec.get("base_url", "") + me_responses_key) if me_responses_key and not str(me_responses_key).startswith("http") else (me_responses_key or "")
        other_responses_url = (other_rec.get("base_url", "") + other_responses_key) if other_responses_key and not str(other_responses_key).startswith("http") else (other_responses_key or "")
        logger.info("  me responses URL:     %s", me_responses_url or "(not found)")
        logger.info("  other responses URL:  %s", other_responses_url or "(not found)")
        video_url = me_rec.get("video_url") or other_rec.get("video_url")
        logger.info("  video URL:            %s", video_url if video_url else "(not found)")

        def _cell(v):
            return "None" if v is None else str(v)

        keypoint_me = me_f.get("evaluation.keypoints.floor_markings_alignment")
        keypoint_other = other_f.get("evaluation.keypoints.floor_markings_alignment")
        bbox_me = me_f.get("evaluation.objects.score")
        bbox_other = other_f.get("evaluation.objects.score")
        total_me = me_f.get("evaluation.score")
        total_other = other_f.get("evaluation.score")
        run_error_me = me_f.get("run.error")
        run_error_other = other_f.get("run.error")
        zero_me = total_me is None or total_me == 0
        zero_other = total_other is None or total_other == 0
        err_suffix_me = " <- [Error] %s" % (run_error_me,) if (zero_me and run_error_me) else ""
        err_suffix_other = " <- [Error] %s" % (run_error_other,) if (zero_other and run_error_other) else ""

        # Table depends on compare mode
        if compare_mode == "bbox":
            # All 5 bbox factor scores + bbox (avg) score
            bbox_place_me = me_f.get("evaluation.objects.bbox_placement")
            bbox_place_other = other_f.get("evaluation.objects.bbox_placement")
            cat_me = me_f.get("evaluation.objects.categorisation")
            cat_other = other_f.get("evaluation.objects.categorisation")
            team_me = me_f.get("evaluation.objects.team")
            team_other = other_f.get("evaluation.objects.team")
            enum_me = me_f.get("evaluation.objects.enumeration")
            enum_other = other_f.get("evaluation.objects.enumeration")
            track_me = me_f.get("evaluation.objects.tracking_stability")
            track_other = other_f.get("evaluation.objects.tracking_stability")
            logger.info("  +--------+----------------+----------------+----------+--------------+--------------------+-------------+")
            logger.info("  |        | bbox_placement | categorisation | team     | enumeration  | tracking_stability | objects_avg |")
            logger.info("  +--------+----------------+----------------+----------+--------------+--------------------+-------------+")
            logger.info("  | you    | %-14s | %-14s | %-8s | %-12s | %-18s | %-11s |%s", _cell(bbox_place_me), _cell(cat_me), _cell(team_me), _cell(enum_me), _cell(track_me), _cell(bbox_me), err_suffix_me)
            logger.info("  | other  | %-14s | %-14s | %-8s | %-12s | %-18s | %-11s |%s", _cell(bbox_place_other), _cell(cat_other), _cell(team_other), _cell(enum_other), _cell(track_other), _cell(bbox_other), err_suffix_other)
            logger.info("  +--------+----------------+----------------+----------+--------------+--------------------+-------------+")
        elif compare_mode == "keypoints":
            logger.info("  +----------------+-----------+")
            logger.info("  |                | keypoint  |")
            logger.info("  +----------------+-----------+")
            logger.info("  | you            | %-9s |%s", _cell(keypoint_me), err_suffix_me)
            logger.info("  | other          | %-9s |%s", _cell(keypoint_other), err_suffix_other)
            logger.info("  +----------------+-----------+")
        else:
            logger.info("  +----------------+-----------+-----------+-----------+")
            logger.info("  |                | keypoint  | bbox      | total     |")
            logger.info("  +----------------+-----------+-----------+-----------+")
            logger.info("  | you            | %-9s | %-9s | %-9s |%s", _cell(keypoint_me), _cell(bbox_me), _cell(total_me), err_suffix_me)
            logger.info("  | other          | %-9s | %-9s | %-9s |%s", _cell(keypoint_other), _cell(bbox_other), _cell(total_other), err_suffix_other)
            logger.info("  +----------------+-----------+-----------+-----------+")
        if task_id != common_task_ids[-1]:
            logger.info("  " + "=" * 76)

    logger.info("")
    logger.info("--- Summary ---")
    logger.info("Common tasks: %d", len(common_task_ids))
    logger.info("Me wins:      %d", me_wins)
    logger.info("Other wins:   %d", other_wins)
    logger.info("Ties:         %d", ties)
    if common_task_ids and valid_count > 0:
        avg_me = sum_me / valid_count
        avg_other = sum_other / valid_count
        mode_label = {"total": "total", "keypoints": "keypoints", "bbox": "bbox"}[compare_mode]
        logger.info("Average (%s) score - you: %s, other: %s (excl. %d tasks with 0-score)", mode_label, round(avg_me, 4), round(avg_other, 4), len(common_task_ids) - valid_count)
        if avg_me > avg_other:
            logger.info("  -> You win on average")
        elif avg_other > avg_me:
            logger.info("  -> Other wins on average")
        else:
            logger.info("  -> Tie on average")
    logger.info("")

    # Extract 0-scoring cases at the end (total evaluation.score is 0 or None)
    zero_me = [(tid, rec) for tid, rec in me_by_task.items() if rec.get("score") is None or rec.get("score") == 0]
    zero_other = [(tid, rec) for tid, rec in other_by_task.items() if rec.get("score") is None or rec.get("score") == 0]
    if zero_me or zero_other:
        logger.info("--- 0-scoring cases ---")
        for tid, rec in sorted(zero_me, key=lambda x: x[0]):
            err = (rec.get("fields") or {}).get("run.error")
            logger.info("  me   task_id %s  %s  %s", tid, ("[Error] %s" % err) if err else "", rec.get("evaluation_url", ""))
        for tid, rec in sorted(zero_other, key=lambda x: x[0]):
            err = (rec.get("fields") or {}).get("run.error")
            logger.info("  other task_id %s  %s  %s", tid, ("[Error] %s" % err) if err else "", rec.get("evaluation_url", ""))
        logger.info("")

    logger.info("Log file: %s", log_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
