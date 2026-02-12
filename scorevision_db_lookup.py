#!/usr/bin/env python3
"""
Look up sub-URLs by HOTKEY from Scorevision R2 index databases.
Given a HOTKEY and N, finds the last up to N sub-URLs containing that HOTKEY
from each of the 4 databases, fetches each evaluation JSON, extracts payload
fields, and writes full URLs and extracted data to a log file.
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
HOTKEY = "5CArRkhHGAUw8B4EZZWWwCBMWNQ7WMWQUDi8cQVCZYK4Ui95"
N = 2  # Max number of sub-URLs to take per database (last N)

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
    """Filter sub-URLs that contain hotkey and return the last up to N."""
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
    keypoints = ev.get("keypoints") or {}
    # JSON uses "objects" (plural); support "object" as fallback
    objects = ev.get("objects") or ev.get("object") or {}

    latency_ms = run.get("latency_ms")
    if latency_ms is not None:
        try:
            latency_ms = float(latency_ms)
        except (TypeError, ValueError):
            latency_ms = 0.0
    else:
        latency_ms = 0.0

    return {
        "task_id": payload.get("task_id"),
        "run.error": run.get("error"),
        "run.responses_key": run.get("responses_key"),
        "run.latency_ms": latency_ms,
        "evaluation.keypoints.floor_markings_alignment": keypoints.get("floor_markings_alignment"),
        "evaluation.objects.bbox_placement": objects.get("bbox_placement"),
        "evaluation.objects.categorisation": objects.get("categorisation"),
        "evaluation.objects.team": objects.get("team"),
        "evaluation.objects.enumeration": objects.get("enumeration"),
        "evaluation.objects.tracking_stability": objects.get("tracking_stability"),
        "evaluation.score": ev.get("score"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find last N sub-URLs containing HOTKEY from each Scorevision DB and log full URLs."
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
    args = parser.parse_args()

    hotkey = HOTKEY.strip()
    n = max(0, N)
    log_path = args.output

    # Setup file logging
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
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
    logger.info("HOTKEY: %s  N: %d", hotkey, n)
    logger.info("")

    # Collect (evaluation full URL, base URL for this DB) so we can build responses_key full URL
    url_pairs: List[Tuple[str, str]] = []

    for db_idx, index_url in enumerate(DB_INDEX_URLS, start=1):
        base_url = get_base_url(index_url)
        try:
            suburls = fetch_index(index_url)
        except (URLError, HTTPError, ValueError, json.JSONDecodeError) as e:
            logger.info("[DB%d] ERROR fetching index: %s", db_idx, e)
            continue

        last_n = find_suburls_with_hotkey(suburls, hotkey, n)
        logger.info("[DB%d] Found %d sub-URLs containing HOTKEY (requested last N=%d)", db_idx, len(last_n), n)

        for sub in last_n:
            full_url = base_url + sub if not sub.startswith("http") else sub
            url_pairs.append((full_url, base_url))
            logger.info("%s", full_url)

        if not last_n:
            logger.info("[DB%d] (none)", db_idx)
        logger.info("")

    # Fetch each evaluation JSON and extract payload fields
    logger.info("--- Extracted payload fields ---")
    for full_url, base_url in url_pairs:
        logger.info("")
        logger.info("Evaluation URL: %s", full_url)
        try:
            data = fetch_json(full_url)
        except (URLError, HTTPError, ValueError, json.JSONDecodeError) as e:
            logger.info("  ERROR fetching: %s", e)
            continue
        if not isinstance(data, list) or len(data) == 0:
            logger.info("  ERROR: expected non-empty JSON array")
            continue
        payload = data[0].get("payload") if isinstance(data[0], dict) else None
        if not payload:
            logger.info("  ERROR: no payload in first element")
            continue
        fields = extract_payload_fields(payload)
        for k, v in fields.items():
            logger.info("  %s: %s", k, v)
        # run.responses_key as full URL
        responses_key = fields.get("run.responses_key")
        if responses_key and isinstance(responses_key, str):
            responses_full = (base_url + responses_key) if not responses_key.startswith("http") else responses_key
            logger.info("  run.responses_key (full URL): %s", responses_full)

    logger.info("")
    logger.info("Total evaluation URLs: %d", len(url_pairs))
    logger.info("Log file: %s", log_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
