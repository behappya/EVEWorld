#!/usr/bin/env python3
"""Gemini-assisted MLR occlusion and persistence audit (optional semantic review layer).

Deterministic MLR remains the primary metric; this script is a reproducible semantic
adjudication layer on top of it. It re-implements the *audit* protocol from
``paper/repair/07_Gemini辅助MLR遮挡与持续性复核方案.md`` while keeping the frozen
MLR event definition from the paper (Appendix "Model Laziness and MLR"): the count is
anchored to N0, all over-counts are violation evidence, an under-count is exempted only
when every missing instance satisfies r_occ >= 0.15, and a deviation counts as an event
only when it persists at two consecutive MLR sampling timestamps.

Protocol (three passes, all frames labeled in-image):

* Pass A (global): ``--repeats`` requests per view over ``mlr24`` (the 24-point MLR
  sampling axis) and ``dense49`` (49 uniformly sampled frames), with the fixed
  conditioning frame, the instruction, the target query and N0. Pass A only proposes
  candidates: it never produces the final label.
* Pass B (dense verify): for each candidate window, ``--repeats`` requests over the
  dense original-frame window ``[onset - before, onset + after]`` plus the previous /
  current / next MLR sampling points.
* Pass C (refutation): the same window and a refutation prompt that asks for benign
  explanations, ``--repeats`` requests.

``--repeats N`` repeats *each* view and *each* verify/refute pass N times, so a video
issues ``2 * N`` global calls and ``2 * N`` calls per candidate.

Consensus (frozen, see ``run_config.json``):

* ``task_count_conserving``: strict majority over all valid global votes.
* candidate support: >= 2 global votes for the same candidate window, or >= 1 global
  vote plus a detector candidate in the same window, or a detector-only candidate.
* Pass B: strict majority of ``confirmed_mlr``; Pass C: any ``rejected`` blocks the
  event, and any high-confidence benign explanation (``benign_explanation`` not in
  ``{none, uncertain}`` at ``confidence == high``) blocks the event as well.
* ``persistence_mlr_samples >= 2`` (median over the confirming Pass B records) and onset
  agreement within 6 original frames or 2 MLR sampling points.

Limitations that this script must not hide (doc section 10):

1. The VLM is not objective ground truth; it does not replace the frozen detector MLR.
2. The protocol must stay frozen: model, frame sampling, JPEG quality, prompts, schema,
   repeat count and aggregation rule are recorded in ``run_config.json``; a resume with
   a different protocol is refused instead of silently mixing runs.
3. Denominators are never mixed: detector rates use the detector-eligible set, Gemini
   rates use the eligible set with a valid Gemini verdict, and VLM-only cases are
   reported in their own bucket.
4. Identity drift, deformation and teleportation at unchanged count stay auxiliary
   failures; they never become MLR count events.
5. ``uncertain`` is not ``negative``: uncertainty is reported as its own rate.
6. No API token is ever written to disk; only the endpoint *name* is recorded.

Manifest contract (JSONL, one row per video):

    {"key": "...", "video_path": "/abs/generated.mp4",
     "instruction": "put the red cup into the tray", "target_query": "red cup",
     "conditioning_frame_path": "/abs/input_frame.png", "n0": 1,
     "detector_trace_path": "/abs/mlr_trace.json", "eligible": true, "frame_count": 93}

Only ``key`` and ``video_path`` are required. The conditioning frame is the fixed input
frame so that eligibility does not depend on the generated rollout.

Detector record contract (``--detector-records`` JSONL, or per-row
``detector_trace_path``). Accepted keys, first match wins::

    key                     # or video / video_path (file stem fallback)
    eligible                # bool
    n0 / initial_count      # int
    detector_mlr / mlr_event / mlr
    detector_counts / counts                # per MLR sample N_t
    detector_adjusted_counts / adjusted_counts  # per MLR sample \tilde N_t
    detector_occlusion_flags / occlusion_flags  # per MLR sample occlusion gate
    occlusion_overlaps      # per MLR sample, per missing instance r_occ (re-computed
                            # with tau=0.15 when present)
    detector_onset_frame / onset_frame
    detector_onset_sample / onset_sample
    max_count, error, detector_trace_path

When counts are present the detector event is re-computed with Algorithm 1 instead of
trusted blindly; ``detector_event_source`` records which path was used. Detector traces
are passed to the model as hints only, never as ground truth.

Outputs (``--output-dir``): ``run_config.json``, ``global_pass_records.jsonl``,
``dense_verify_records.jsonl``, ``refutation_records.jsonl``,
``per_video_consensus.jsonl``, ``per_video_consensus.csv``, ``summary.json``, optional
``frame_cache/``, and ``dry_run_requests.jsonl`` for ``--dry-run``.

Usage::

    conda activate <your-env>                 # needs google-genai (+ opencv for video frames)
    cd "$EVEWORLD_ROOT"                       # this repository
    python eveworld/evaluation/eval_gemini_mlr_audit.py \
      --manifest /path/to/mlr_audit_manifest.jsonl \
      --output-dir /path/to/gemini_mlr_audit_v1 \
      --model gemini-3.5-flash \
      --global-frame-count 49 --mlr-sample-count 24 \
      --dense-window-before 6 --dense-window-after 10 \
      --jpeg-quality 85 --temperature 0 --thinking-level low \
      --repeats 3 --concurrency 64 --resume

Offline checks: ``--help``, ``--self-test`` and ``--dry-run`` need no API key; only real
runs read ``DIFROST_API_TOKEN`` (repo difrost endpoint) or ``GEMINI_API_KEY`` /
``GOOGLE_API_KEY`` (direct google-genai).

Known, deliberate deviations from the design note: under-count local crops of the robot /
target (doc 4.2, marked optional) are not implemented; the conditioning frame is labeled
``conditioning=1`` as a documented extension of the label protocol; Pass B/C run only for
candidates that pass the global support gate (everything else is reported as
``insufficient_global_support`` instead of being quietly dropped).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # optional: only needed to decode frames
    import cv2  # type: ignore

    _CV2_IMPORT_ERROR: str | None = None
except Exception as _exc:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

try:  # optional: only needed for real model calls
    from google import genai  # type: ignore
    from google.genai import types as gt  # type: ignore

    _GENAI_IMPORT_ERROR: str | None = None
except Exception as _exc:  # pragma: no cover - environment dependent
    genai = None  # type: ignore
    gt = None  # type: ignore
    _GENAI_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
try:  # repo helper: shared difrost endpoint client (imports lance/openai/PIL, so optional)
    import gemini_consensus_judge as gemini_ref  # type: ignore
except Exception:  # pragma: no cover - optional convenience import
    gemini_ref = None


# --------------------------------------------------------------------------------------
# Frozen protocol constants
# --------------------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("DIFROST_MODEL") or (
    getattr(gemini_ref, "DEFAULT_MODEL", None) if gemini_ref is not None else None
) or "gemini-3.5-flash"

PROTOCOL_NAME = "gemini_mlr_audit"
PROTOCOL_VERSION = "v1"

SCHEMA_VERSION_GLOBAL = "mlr_audit_global_v1"
SCHEMA_VERSION_VERIFY = "mlr_audit_verify_v1"

GLOBAL_VIEWS = ("mlr24", "dense49")

OCCLUSION_TAU = 0.15
REQUIRED_CONSECUTIVE_SAMPLES = 2
ONSET_TOLERANCE_FRAMES = 6
ONSET_TOLERANCE_SAMPLES = 2
CANDIDATE_CLUSTER_TOLERANCE_FRAMES = 6
DEFAULT_MAX_CANDIDATES_PER_VIDEO = 3
DEFAULT_REPEATS = 3

GLOBAL_RECORDS_FILE = "global_pass_records.jsonl"
VERIFY_RECORDS_FILE = "dense_verify_records.jsonl"
REFUTE_RECORDS_FILE = "refutation_records.jsonl"
CONSENSUS_JSONL_FILE = "per_video_consensus.jsonl"
CONSENSUS_CSV_FILE = "per_video_consensus.csv"
SUMMARY_FILE = "summary.json"
RUN_CONFIG_FILE = "run_config.json"
DRY_RUN_FILE = "dry_run_requests.jsonl"
FRAME_CACHE_DIR = "frame_cache"

# Column order of per_video_consensus.csv (doc section 8).
CONSENSUS_COLUMNS = [
    "key",
    "eligible",
    "n0",
    "primary_detector_mlr",
    "gemini_confirmed_mlr",
    "verdict",
    "event_type",
    "onset_frame",
    "end_frame",
    "persistence_original_frames",
    "persistence_mlr_samples",
    "first_confirmed_mlr_pair",
    "occlusion_assessment",
    "confidence",
    "detector_gemini_agree",
    "uncertainty_reason",
]

VERDICTS = (
    "confirmed_mlr",
    "rejected",
    "uncertain_need_human",
    "detector_positive_vlm_rejected",
    "vlm_positive_detector_negative",
    "detector_only_unverified",
    "ineligible",
)


SYSTEM_PROMPT = """You are an occlusion-aware Model Laziness auditor for embodied robot video rollouts.
Use only the provided frames, frame labels, instruction, and target query.
Return only JSON matching the requested schema.

Definition:
Model Laziness for this audit means a persistent target-instance count violation
relative to the initial observed count N0. Unsupported extra target instances are
duplication events. Missing target instances are disappearance events only when
the missing target is not plausibly explained by robot-induced occlusion.

Important rules:
- Do not infer correctness from the final frame alone. Inspect the temporal process.
- Use the initial/conditioning frame as the reference inventory.
- Count only the manipulated target specified by the target query.
- Over-counts are retained as violation evidence.
- Under-counts caused by robot gripper/arm occlusion are not violation evidence.
- If the target is outside the observable region, heavily blurred, or hidden by
  non-robot clutter, mark uncertainty rather than forcing a violation.
- Identity drift, deformation, or teleportation at the correct count should be
  recorded as auxiliary failures, but should not by itself become an MLR count event.
- Frame indices are 0-based original video frame indices.

Additional protocol rules (frozen):
- The MLR sampling axis is round(i * (F - 1) / 23) for i = 0..23, where F is the total
  number of frames of the original video. All frame indices you report must be 0-based
  original video frame indices in that coordinate system.
- A count deviation is an MLR event only when the adjusted count differs from N0 at two
  consecutive MLR sampling timestamps. Isolated single-timestamp deviations are not
  persistent events.
- Over-count (N_t > N0) is always violation evidence and is never exempted by occlusion.
- An under-count (N_t < N0) is occlusion-supported, and therefore not violation evidence,
  only if EVERY missing target instance satisfies r_occ >= 0.15, where r_occ is the
  fraction of that missing instance's target reference region covered by the robot
  occluder mask in that frame. If only some missing instances are covered, the under-count
  remains violation evidence.
- Identity drift, deformation and teleportation without a count change are auxiliary
  failures: fill the auxiliary_failures fields but do not turn them into an MLR count event.
- Every image carries a label burned into its top-left corner: "key=<video id>
  frame=<0-based original frame index>", plus " sample=<i>" (i in 0..23) when the image is
  one of the 24 MLR sampling timestamps. The conditioning/initial image is labeled
  "key=<video id> frame=0 conditioning=1". Trust the burned-in label, not the image order.
- Detector traces, when provided, are hints only. They are not ground truth and you may
  disagree with them when the visual evidence supports disagreement.
- Set schema_version to "{schema_version}" in your JSON answer.
"""

GLOBAL_USER_TEMPLATE = """Instruction:
{instruction}

Target query:
{target_query}

Initial target count N0:
{n0}

You will receive the conditioning/initial frame followed by uniformly sampled
video frames with visible frame labels. Determine whether the manipulated target
count persistently deviates from N0.

Return:
1. whether this task should conserve the target count;
2. the visible target count and occlusion status at important frames;
3. candidate MLR event windows, if any;
4. whether each candidate is over-count, under-count, or uncertain;
5. whether a local dense-frame check is needed.

If a detector trace is provided, treat it only as a hint. You may disagree with it
when the visual evidence supports disagreement.
"""

VERIFY_USER_TEMPLATE = """Instruction:
{instruction}

Target query:
{target_query}

Initial target count N0:
{n0}

Candidate event from the previous pass:
{candidate_summary}

You will receive the initial frame and a dense temporal window around the candidate
onset. Decide whether this candidate is a real MLR event.

Answer conservatively:
- Confirm duplication only if an extra target instance is visible while the original
  target instance remains or should still exist.
- Confirm disappearance only if the target remains missing after the robot occluder
  has moved away, or if there is no plausible robot-induced occlusion.
- Reject or mark uncertain if the evidence is explainable by robot occlusion,
  non-robot occlusion, out-of-frame motion, severe blur, or ambiguous target identity.

Return exact 0-based frame indices for onset and end when possible.
"""

REFUTE_USER_TEMPLATE = """Instruction:
{instruction}

Target query:
{target_query}

Initial target count N0:
{n0}

Proposed MLR event:
{candidate_summary}

Your task is to try to refute the proposed MLR event. Look for benign explanations:
robot occlusion, legal manipulation, out-of-view target, camera crop, target identity
ambiguity, or detector-like visual artifacts.

Return JSON stating whether the event remains confirmed, rejected, or uncertain.
Do not confirm unless the count violation is visibly unsupported.
"""

DENSE_WINDOW_PROMPT_NOTE = """Frame layout of this request:
- The first image is the conditioning/initial frame (label "conditioning=1").
- The remaining images are the local dense window around the candidate onset, in
  chronological order, followed by any previous/current/next MLR sampling point that
  is not already inside the dense window. Images whose labels contain "sample=<i>" are
  MLR sampling timestamps; the others are raw original frames inside the dense window.
- Report onset_frame / end_frame as 0-based original frame indices, and
  persistence_mlr_samples as the number of consecutive MLR sampling timestamps whose
  adjusted count deviates from N0.
"""


# --------------------------------------------------------------------------------------
# Frozen JSON schemas (verbatim from the design note, section 6)
# --------------------------------------------------------------------------------------

MLR_GLOBAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "task_count_conserving": {"type": "integer"},
        "target_query_used": {"type": "string"},
        "initial_count_n0": {"type": "integer"},
        "initial_count_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "uncertain"],
        },
        "has_mlr_candidate": {"type": "integer"},
        "candidate_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "event_type": {
                        "type": "string",
                        "enum": ["duplication", "disappearance", "both", "uncertain"],
                    },
                    "first_suspicious_frame": {"type": "integer"},
                    "first_confirming_frame": {"type": "integer"},
                    "last_suspicious_frame": {"type": "integer"},
                    "supporting_frames": {"type": "array", "items": {"type": "integer"}},
                    "opposing_or_ambiguous_frames": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "robot_occlusion_possible": {"type": "integer"},
                    "needs_dense_check": {"type": "integer"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low", "uncertain"],
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "event_id",
                    "event_type",
                    "first_suspicious_frame",
                    "first_confirming_frame",
                    "last_suspicious_frame",
                    "supporting_frames",
                    "opposing_or_ambiguous_frames",
                    "robot_occlusion_possible",
                    "needs_dense_check",
                    "confidence",
                    "reason",
                ],
            },
        },
        "auxiliary_failures": {
            "type": "object",
            "properties": {
                "identity_drift": {"type": "integer"},
                "deformation": {"type": "integer"},
                "teleportation_without_count_change": {"type": "integer"},
                "other": {"type": "string"},
            },
            "required": [
                "identity_drift",
                "deformation",
                "teleportation_without_count_change",
                "other",
            ],
        },
        "overall_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "uncertain"],
        },
        "short_reason": {"type": "string"},
    },
    "required": [
        "schema_version",
        "task_count_conserving",
        "target_query_used",
        "initial_count_n0",
        "initial_count_confidence",
        "has_mlr_candidate",
        "candidate_events",
        "auxiliary_failures",
        "overall_confidence",
        "short_reason",
    ],
}

MLR_VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "event_id": {"type": "string"},
        "verdict": {
            "type": "string",
            "enum": ["confirmed_mlr", "rejected", "uncertain_need_human"],
        },
        "event_type": {
            "type": "string",
            "enum": ["duplication", "disappearance", "both", "none", "uncertain"],
        },
        "onset_frame": {"type": "integer"},
        "end_frame": {"type": "integer"},
        "persistence_original_frames": {"type": "integer"},
        "persistence_mlr_samples": {"type": "integer"},
        "first_confirmed_mlr_pair": {"type": "array", "items": {"type": "integer"}},
        "occlusion_assessment": {
            "type": "string",
            "enum": [
                "no_occlusion",
                "robot_occlusion_explains",
                "robot_occlusion_insufficient",
                "non_robot_occlusion",
                "out_of_view",
                "uncertain",
            ],
        },
        "benign_explanation": {
            "type": "string",
            "enum": [
                "none",
                "robot_occlusion",
                "legal_task_change",
                "out_of_view",
                "camera_or_crop",
                "blur_or_low_resolution",
                "target_identity_ambiguous",
                "other",
                "uncertain",
            ],
        },
        "evidence_frames": {"type": "array", "items": {"type": "integer"}},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "uncertain"],
        },
        "short_reason": {"type": "string"},
    },
    "required": [
        "schema_version",
        "event_id",
        "verdict",
        "event_type",
        "onset_frame",
        "end_frame",
        "persistence_original_frames",
        "persistence_mlr_samples",
        "first_confirmed_mlr_pair",
        "occlusion_assessment",
        "benign_explanation",
        "evidence_frames",
        "confidence",
        "short_reason",
    ],
}


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_name(value: str, limit: int = 60) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return cleaned[:limit] or "item"


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object per line")
            rows.append(row)
    return rows


def load_json_any(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return read_jsonl(path)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = load_json_any(path)
        if isinstance(payload, dict):
            payload = payload.get("items") or payload.get("rows") or []
        if not isinstance(payload, list):
            raise ValueError(f"{path}: manifest must be JSONL or a JSON array")
        rows = [row for row in payload if isinstance(row, dict)]
    else:
        rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def parse_json_object(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            return None
        try:
            value = json.loads(text[left : right + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def response_text_parts(response: Any) -> tuple[list[str], list[str]]:
    if gemini_ref is not None and hasattr(gemini_ref, "response_text_parts"):
        return gemini_ref.response_text_parts(response)
    texts: list[str] = []
    thoughts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if not text:
                continue
            (thoughts if getattr(part, "thought", False) else texts).append(text)
    return texts, thoughts


# --------------------------------------------------------------------------------------
# Frame protocol: sampling axes, labels, extraction
# --------------------------------------------------------------------------------------


def mlr_sample_points(total_frames: int, sample_count: int) -> list[tuple[int, int]]:
    """Return the MLR sampling axis as ``(sample_id, frame_index)`` pairs.

    The frozen axis is ``round(i * (F - 1) / (count - 1))`` for ``i = 0..count-1``, i.e.
    ``round(i * (F - 1) / 23)`` for the default 24 samples. Duplicate frame indices (short
    videos) collapse onto the smallest sample id so that the axis stays strictly ordered.
    """
    if sample_count <= 1:
        return [(0, 0)]
    if total_frames <= 1:
        return [(i, 0) for i in range(sample_count)]
    points: list[tuple[int, int]] = []
    seen: set[int] = set()
    for sample_id in range(sample_count):
        frame_index = int(round(sample_id * (total_frames - 1) / (sample_count - 1)))
        if frame_index in seen:
            continue
        seen.add(frame_index)
        points.append((sample_id, frame_index))
    return points


def mlr_sample_frame_indices(total_frames: int, sample_count: int) -> list[int]:
    return [frame for _, frame in mlr_sample_points(total_frames, sample_count)]


def uniform_frame_indices(total_frames: int, frame_count: int) -> list[int]:
    """Uniformly sampled frame indices (same ``round`` formula), sorted and deduplicated."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if total_frames <= 0:
        return list(range(frame_count))
    if frame_count == 1:
        return [total_frames // 2]
    return sorted({round(i * (total_frames - 1) / (frame_count - 1)) for i in range(frame_count)})


def nearest_mlr_sample(total_frames: int, sample_count: int, frame_index: int) -> int | None:
    """Nearest MLR sample id for an original frame index (ties -> smaller sample id)."""
    points = mlr_sample_points(total_frames, sample_count)
    if not points:
        return None
    return min(points, key=lambda item: (abs(item[1] - frame_index), item[0]))[0]


def mlr_sample_frame(total_frames: int, sample_count: int, sample_id: int) -> int | None:
    for sid, frame in mlr_sample_points(total_frames, sample_count):
        if sid == sample_id:
            return frame
    return None


def dense_window_bounds(
    total_frames: int, onset_frame: int, before: int, after: int
) -> tuple[int, int]:
    return (
        clamp(onset_frame - max(0, before), 0, max(0, total_frames - 1)),
        clamp(onset_frame + max(0, after), 0, max(0, total_frames - 1)),
    )


def frame_label(
    key: str, frame_index: int, sample_id: int | None = None, conditioning: bool = False
) -> str:
    label = f"key={key} frame={frame_index}"
    if conditioning:
        return label + " conditioning=1"
    if sample_id is not None:
        return label + f" sample={sample_id}"
    return label


def merge_frame_requests(*groups: list[tuple[int, int | None]]) -> list[tuple[int, int | None]]:
    """Merge ``(frame_index, sample_id)`` groups in chronological order without duplicates."""
    merged: dict[int, int | None] = {}
    for group in groups:
        for frame_index, sample_id in group:
            if frame_index not in merged:
                merged[frame_index] = sample_id
            elif merged[frame_index] is None and sample_id is not None:
                merged[frame_index] = sample_id
    return sorted(merged.items(), key=lambda item: item[0])


def require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError(
            "opencv (cv2) is required to extract frames; install opencv-python in the active "
            f"environment ({_CV2_IMPORT_ERROR})"
        )
    return cv2


def draw_frame_label(frame: Any, label: str) -> Any:
    cv = require_cv2()
    height, width = frame.shape[:2]
    scale = max(0.4, min(1.1, width / 960.0))
    thickness = max(1, int(round(scale * 1.7)))
    (text_w, text_h), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = max(2, int(round(scale * 3)))
    cv.rectangle(
        frame, (pad - 2, pad - 2), (pad + text_w + 2, pad + text_h + baseline + 2), (24, 24, 24), -1
    )
    cv.putText(
        frame,
        label,
        (pad, pad + text_h),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv.LINE_AA,
    )
    return frame


def resize_frame(frame: Any, max_image_side: int) -> Any:
    cv = require_cv2()
    if max_image_side <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_image_side:
        return frame
    scale = max_image_side / float(longest)
    return cv.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv.INTER_AREA,
    )


def encode_jpeg(frame: Any, jpeg_quality: int) -> bytes:
    cv = require_cv2()
    ok, encoded = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for a labeled frame")
    return encoded.tobytes()


def probe_frame_count(video_path: Path) -> int:
    cv = require_cv2()
    if not video_path.is_file():
        raise FileNotFoundError(str(video_path))
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        total = int(capture.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            return total
        total = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            total += 1
    finally:
        capture.release()
    if total <= 0:
        raise RuntimeError(f"no decodable frames: {video_path}")
    return total


def frame_cache_path(cache_dir: Path, key: str, label: str) -> Path:
    return cache_dir / f"{sanitize_name(key)}__{sha256_text(label)[:10]}.jpg"


def extract_labeled_frames(
    video_path: Path,
    labeled_frames: list[tuple[int, str]],
    *,
    jpeg_quality: int,
    max_image_side: int,
    cache_dir: Path | None,
) -> dict[int, bytes]:
    """Decode every requested frame in a single forward pass and return index -> JPEG bytes.

    ``labeled_frames`` is a list of ``(frame_index, label_text)``; the label is burned into
    the top-left corner of the image (doc section 3.2) before JPEG encoding.
    """
    cv = require_cv2()
    labels: dict[int, str] = {}
    for frame_index, label in labeled_frames:
        labels.setdefault(int(frame_index), label)
    wanted = sorted(labels)
    if not wanted:
        return {}

    payloads: dict[int, bytes] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for frame_index in wanted:
            cached = frame_cache_path(cache_dir, str(video_path), labels[frame_index])
            if cached.is_file():
                payloads[frame_index] = cached.read_bytes()

    remaining = [frame_index for frame_index in wanted if frame_index not in payloads]
    if remaining:
        pending = set(remaining)
        capture = cv.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        try:
            cursor = 0
            while pending:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if cursor in pending:
                    labeled = draw_frame_label(frame, labels[cursor])
                    resized = resize_frame(labeled, max_image_side)
                    data = encode_jpeg(resized, jpeg_quality)
                    payloads[cursor] = data
                    if cache_dir is not None:
                        target = frame_cache_path(cache_dir, str(video_path), labels[cursor])
                        tmp = target.with_name(target.name + ".tmp")
                        tmp.write_bytes(data)
                        os.replace(tmp, target)
                    pending.discard(cursor)
                cursor += 1
        finally:
            capture.release()
        if pending:
            raise RuntimeError(
                f"could not decode frames {sorted(pending)[:8]} (of {len(wanted)}) from {video_path}"
            )
    return payloads


def encode_conditioning_frame(
    frame_path: Path,
    *,
    key: str,
    jpeg_quality: int,
    max_image_side: int,
    cache_dir: Path | None,
) -> tuple[bytes, str]:
    cv = require_cv2()
    label = frame_label(key, 0, conditioning=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = frame_cache_path(cache_dir, key, label + "_cond")
        if cached.is_file():
            return cached.read_bytes(), label
    image = cv.imread(str(frame_path))
    if image is None:
        raise RuntimeError(f"could not read conditioning frame: {frame_path}")
    labeled = draw_frame_label(image, label)
    data = encode_jpeg(resize_frame(labeled, max_image_side), jpeg_quality)
    if cache_dir is not None:
        target = frame_cache_path(cache_dir, key, label + "_cond")
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
    return data, label


def frame_layout_note(conditioning_label: str, labels: list[str]) -> str:
    ordered = [conditioning_label, *labels]
    body = "; ".join(f"{position + 1}: {label}" for position, label in enumerate(ordered[:24]))
    if len(ordered) > 24:
        tail = "; ".join(f"{position + 1}: {label}" for position, label in enumerate(ordered))
        return (
            "Frame order of this request (image order, each label is also burned into the "
            f"image):\n{tail}"
        )
    return f"Frame order of this request (image order): {body}"


# --------------------------------------------------------------------------------------
# Prompt builders
# --------------------------------------------------------------------------------------


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(schema_version=SCHEMA_VERSION_GLOBAL)


def build_global_prompt(
    *,
    instruction: str,
    target_query: str,
    n0: int,
    conditioning_label: str,
    labels: list[str],
    detector_hint: str | None,
) -> str:
    prompt = GLOBAL_USER_TEMPLATE.format(
        instruction=instruction, target_query=target_query, n0=n0
    ).replace("{schema_version}", SCHEMA_VERSION_GLOBAL)
    prompt += "\n" + frame_layout_note(conditioning_label, labels) + "\n"
    if detector_hint:
        prompt += (
            "\nDetector trace (hint only, not ground truth):\n" + detector_hint + "\n"
        )
    prompt += (
        f"\nSet schema_version to \"{SCHEMA_VERSION_GLOBAL}\". Every integer field must be "
        "an integer, and has_mlr_candidate / task_count_conserving must be 0 or 1.\n"
    )
    return prompt


def build_candidate_summary(candidate: dict[str, Any], detector: dict[str, Any] | None) -> str:
    views = ",".join(sorted(candidate.get("views") or [])) or "detector_only"
    lines = [
        f"candidate_id: {candidate.get('candidate_id')}",
        f"event_type: {candidate.get('event_type')}",
        f"candidate onset (0-based original frame): {candidate.get('onset_frame')}",
        f"candidate onset (nearest MLR sample id): {candidate.get('onset_sample_id')}",
        f"candidate window (original frames): "
        f"{candidate.get('first_suspicious_frame')}..{candidate.get('last_suspicious_frame')}",
        f"supporting global votes: {candidate.get('vote_count')} (views: {views})",
    ]
    reasons = candidate.get("reasons") or []
    if reasons:
        lines.append("reported reasons: " + " | ".join(str(reason) for reason in reasons[:3]))
    if detector:
        lines.append(
            "detector trace (hint only, not ground truth): "
            f"mlr_event={detector.get('detector_mlr')}, "
            f"onset_sample={detector.get('detector_onset_sample')}, "
            f"onset_frame={detector.get('detector_onset_frame')}, "
            f"n0={detector.get('n0')}, max_count={detector.get('max_count')}, "
            f"event_source={detector.get('detector_event_source')}"
        )
    else:
        lines.append("detector trace: unavailable (hint only, not ground truth)")
    return "\n".join(lines)


def build_verify_prompt(
    *,
    event_id: str,
    instruction: str,
    target_query: str,
    n0: int,
    candidate_summary: str,
    conditioning_label: str,
    labels: list[str],
) -> str:
    prompt = VERIFY_USER_TEMPLATE.format(
        instruction=instruction, target_query=target_query, n0=n0, candidate_summary=candidate_summary
    )
    prompt += "\n" + DENSE_WINDOW_PROMPT_NOTE + "\n"
    prompt += frame_layout_note(conditioning_label, labels) + "\n"
    prompt += (
        f'\nUse event_id "{event_id}" (identical to the candidate summary above) and set '
        f'schema_version to "{SCHEMA_VERSION_VERIFY}".\n'
    )
    return prompt


def build_refute_prompt(
    *,
    event_id: str,
    instruction: str,
    target_query: str,
    n0: int,
    candidate_summary: str,
    conditioning_label: str,
    labels: list[str],
) -> str:
    prompt = REFUTE_USER_TEMPLATE.format(
        instruction=instruction, target_query=target_query, n0=n0, candidate_summary=candidate_summary
    )
    prompt += "\n" + DENSE_WINDOW_PROMPT_NOTE + "\n"
    prompt += frame_layout_note(conditioning_label, labels) + "\n"
    prompt += (
        "\nSearch actively for a benign explanation. If a benign explanation is plausible, "
        "fill benign_explanation with it; use \"none\" only when no benign explanation "
        "survives. Use the same candidate event_id shown above and set schema_version to "
        f'"{SCHEMA_VERSION_VERIFY}".\n'
    )
    return prompt


# --------------------------------------------------------------------------------------
# Detector trace handling (Algorithm 1 replication)
# --------------------------------------------------------------------------------------


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _as_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                return None
        return out
    return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return 1
        if text in {"0", "false", "no", "n"}:
            return 0
    return None


def occlusion_overlaps_by_sample(value: Any) -> list[list[float]] | None:
    """Normalize ``occlusion_overlaps`` (per sample, per missing instance) into nested lists."""
    if value is None:
        return None
    if isinstance(value, dict):
        keys = sorted(value, key=lambda item: int(item))
        rows = [value[key] for key in keys]
    elif isinstance(value, list):
        rows = value
    else:
        return None
    out: list[list[float]] = []
    for row in rows:
        if row is None:
            out.append([])
            continue
        if not isinstance(row, (list, tuple)):
            row = [row]
        values: list[float] = []
        for item in row:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                return None
        out.append(values)
    return out


def under_count_occlusion_supported(overlaps: list[float] | None, tau: float = OCCLUSION_TAU) -> bool:
    """A missing instance is occlusion-supported only if every missing instance has r_occ >= tau."""
    if not overlaps:
        return False
    return all(value >= tau for value in overlaps)


def adjusted_counts_from_raw(
    n0: int,
    counts: list[int],
    occlusion_flags: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Replicate Algorithm 1 steps 2-4.

    Returns ``(adjusted_counts, exempt_flags)``. Over-counts are always retained; an
    under-count is exempted (``tilde N_t = N0``) only when its occlusion gate is set.
    """
    adjusted: list[int] = []
    exempt: list[int] = []
    for index, count in enumerate(counts):
        flag = bool(occlusion_flags[index]) if occlusion_flags and index < len(occlusion_flags) else False
        if count < n0 and flag:
            adjusted.append(n0)
            exempt.append(1)
        else:
            adjusted.append(count)
            exempt.append(0)
    return adjusted, exempt


def first_persistent_deviation(
    n0: int, adjusted: list[int], required: int = REQUIRED_CONSECUTIVE_SAMPLES
) -> dict[str, Any]:
    """Algorithm 1 step 5: event iff the adjusted count deviates at ``required`` consecutive samples."""
    deviation = [1 if count != n0 else 0 for count in adjusted]
    longest = 0
    current = 0
    for value in deviation:
        current = current + 1 if value else 0
        longest = max(longest, current)
    onset_sample: int | None = None
    pair: list[int] | None = None
    for index in range(len(adjusted) - required + 1):
        window = deviation[index : index + required]
        if all(window):
            onset_sample = index
            pair = [index, index + required - 1]
            break
    return {
        "mlr_event": onset_sample is not None,
        "onset_sample": onset_sample,
        "first_confirmed_pair": pair,
        "max_consecutive_deviations": longest,
        "persistence_samples": longest,
    }


def normalize_detector_record(row: dict[str, Any], key: str) -> dict[str, Any]:
    n0 = _as_int(_first_present(row, ("n0", "initial_count", "initial_count_n0")))
    counts = _as_int_list(_first_present(row, ("detector_counts", "counts", "n_t", "N_t")))
    adjusted = _as_int_list(
        _first_present(row, ("detector_adjusted_counts", "adjusted_counts", "tilde_n_t"))
    )
    flags = _as_int_list(
        _first_present(row, ("detector_occlusion_flags", "occlusion_flags", "detector_occlusion"))
    )
    overlaps = occlusion_overlaps_by_sample(
        _first_present(row, ("occlusion_overlaps", "detector_occlusion_overlaps", "r_occ"))
    )
    record: dict[str, Any] = {
        "key": key,
        "eligible": _as_bool_int(_first_present(row, ("eligible", "is_eligible"))),
        "n0": n0,
        "counts": counts,
        "adjusted_counts": adjusted,
        "occlusion_flags": flags,
        "occlusion_overlaps": overlaps,
        "max_count": _first_present(row, ("max_count", "detector_max_count")),
        "detector_mlr_given": _as_bool_int(
            _first_present(row, ("detector_mlr", "mlr_event", "mlr", "primary_detector_mlr"))
        ),
        "detector_onset_frame_given": _first_present(
            row, ("detector_onset_frame", "onset_frame")
        ),
        "detector_onset_sample_given": _first_present(
            row, ("detector_onset_sample", "onset_sample")
        ),
        "detector_trace_path": _first_present(row, ("detector_trace_path", "trace_path")),
        "error": row.get("error"),
        "available": True,
    }
    return record


def detector_event_from_trace(
    record: dict[str, Any], *, mlr_sample_count: int, total_frames: int | None
) -> dict[str, Any]:
    """Re-compute the frozen detector event (Algorithm 1) or trust the recorded flag."""
    out = dict(record)
    n0 = record.get("n0")
    counts = record.get("counts")
    overlaps = record.get("occlusion_overlaps")
    flags = record.get("occlusion_flags")
    source = "given_mlr_event"
    event: dict[str, Any] = {
        "mlr_event": None,
        "onset_sample": None,
        "first_confirmed_pair": None,
        "persistence_samples": None,
        "max_consecutive_deviations": None,
    }

    if n0 is not None and counts:
        derived_flags = flags
        if overlaps is not None and len(overlaps) == len(counts):
            derived_flags = [
                1 if under_count_occlusion_supported(row) else 0 for row in overlaps
            ]
        adjusted, exempt = adjusted_counts_from_raw(int(n0), counts, derived_flags)
        event = first_persistent_deviation(int(n0), adjusted)
        source = "recomputed_from_counts" if overlaps is None else "recomputed_from_overlaps"
        out["adjusted_counts"] = adjusted
        out["occlusion_flags"] = derived_flags if derived_flags is not None else exempt
    elif record.get("adjusted_counts") and n0 is not None:
        event = first_persistent_deviation(int(n0), record["adjusted_counts"])
        source = "recomputed_from_adjusted_counts"
    elif record.get("detector_mlr_given") is not None:
        event = {
            "mlr_event": bool(record["detector_mlr_given"]),
            "onset_sample": record.get("detector_onset_sample_given"),
            "first_confirmed_pair": None,
            "persistence_samples": None,
            "max_consecutive_deviations": None,
        }
    else:
        source = "unavailable"

    out["detector_event_source"] = source
    out["detector_mlr"] = event["mlr_event"]
    onset_sample = event["onset_sample"]
    if onset_sample is None:
        onset_sample = record.get("detector_onset_sample_given")
    onset_frame = record.get("detector_onset_frame_given")
    if onset_frame is None and onset_sample is not None and total_frames:
        onset_frame = mlr_sample_frame(total_frames, mlr_sample_count, int(onset_sample))
    out["detector_onset_sample"] = onset_sample
    out["detector_onset_frame"] = onset_frame
    out["detector_first_confirmed_pair"] = event["first_confirmed_pair"]
    out["detector_persistence_samples"] = event["persistence_samples"]
    out["detector_max_consecutive_deviations"] = event["max_consecutive_deviations"]
    if record.get("counts") and out.get("max_count") in (None, ""):
        out["max_count"] = max(record["counts"])
    return out


def detector_hint_text(detector: dict[str, Any] | None) -> str | None:
    if not detector or not detector.get("available"):
        return None
    parts = [f"detector_mlr={_fmt(detector.get('detector_mlr'))}"]
    if detector.get("n0") is not None:
        parts.append(f"n0={detector['n0']}")
    if detector.get("counts"):
        parts.append(f"N_t={detector['counts']}")
        if detector.get("adjusted_counts"):
            parts.append(f"adjusted_N_t={detector['adjusted_counts']}")
    if detector.get("occlusion_flags"):
        parts.append(f"occlusion_gate={detector['occlusion_flags']}")
    if detector.get("detector_onset_sample") is not None:
        parts.append(f"onset_sample={detector['detector_onset_sample']}")
    if detector.get("detector_onset_frame") is not None:
        parts.append(f"onset_frame={detector['detector_onset_frame']}")
    parts.append(f"event_source={detector.get('detector_event_source')}")
    parts.append("This trace is a hint only; disagree when the frames support disagreement.")
    return "\n".join(parts)


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    return str(value)


# --------------------------------------------------------------------------------------
# Candidate discovery and clustering
# --------------------------------------------------------------------------------------


def _types_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    if "uncertain" in (left, right):
        return True
    return left == "both" or right == "both"


def global_candidate_votes(global_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract per-record candidate votes from parsed Pass A responses."""
    votes: list[dict[str, Any]] = []
    for record in global_records:
        parsed = record.get("parsed")
        if not isinstance(parsed, dict):
            continue
        for event in parsed.get("candidate_events") or []:
            if not isinstance(event, dict):
                continue
            confirming = event.get("first_confirming_frame")
            suspicious = event.get("first_suspicious_frame")
            onset = confirming if isinstance(confirming, (int, float)) and confirming > 0 else suspicious
            if not isinstance(onset, (int, float)):
                continue
            votes.append(
                {
                    "view": record.get("view"),
                    "repeat": record.get("repeat"),
                    "record_id": record.get("record_id"),
                    "event_type": str(event.get("event_type") or "uncertain"),
                    "onset_frame": int(onset),
                    "first_suspicious_frame": event.get("first_suspicious_frame"),
                    "last_suspicious_frame": event.get("last_suspicious_frame"),
                    "supporting_frames": event.get("supporting_frames") or [],
                    "opposing_or_ambiguous_frames": event.get("opposing_or_ambiguous_frames") or [],
                    "robot_occlusion_possible": event.get("robot_occlusion_possible"),
                    "needs_dense_check": event.get("needs_dense_check"),
                    "confidence": str(event.get("confidence") or "uncertain"),
                    "reason": str(event.get("reason") or ""),
                }
            )
    return votes


def cluster_candidates(votes: list[dict[str, Any]], tolerance: int = CANDIDATE_CLUSTER_TOLERANCE_FRAMES) -> list[dict[str, Any]]:
    """Greedy clustering by onset proximity plus compatible event types."""
    clusters: list[dict[str, Any]] = []
    ordered = sorted(votes, key=lambda vote: (vote["onset_frame"], str(vote.get("view")), vote.get("repeat") or 0))
    for vote in ordered:
        target: dict[str, Any] | None = None
        for cluster in clusters:
            if _types_compatible(cluster["event_type"], vote["event_type"]) and abs(
                int(cluster["onset_frame"]) - int(vote["onset_frame"])
            ) <= tolerance:
                target = cluster
                break
        if target is None:
            clusters.append(
                {
                    "event_type": vote["event_type"],
                    "onset_frame": int(vote["onset_frame"]),
                    "first_suspicious_frame": vote.get("first_suspicious_frame"),
                    "last_suspicious_frame": vote.get("last_suspicious_frame"),
                    "votes": [],
                    "robot_occlusion_possible": 0,
                }
            )
            target = clusters[-1]
        target["votes"].append(vote)
        onsets = sorted(int(item["onset_frame"]) for item in target["votes"])
        target["onset_frame"] = int(statistics.median(onsets))
        suspicion = [
            int(item[key])
            for item in target["votes"]
            for key in ("first_suspicious_frame",)
            if isinstance(item.get(key), (int, float))
        ]
        last = [
            int(item[key])
            for item in target["votes"]
            for key in ("last_suspicious_frame",)
            if isinstance(item.get(key), (int, float))
        ]
        if suspicion:
            target["first_suspicious_frame"] = min(suspicion)
        if last:
            target["last_suspicious_frame"] = max(last)
        target["event_type"] = _mode([item["event_type"] for item in target["votes"]]) or "uncertain"
        occlusion = [
            int(item["robot_occlusion_possible"])
            for item in target["votes"]
            if isinstance(item.get("robot_occlusion_possible"), (int, float))
        ]
        if occlusion:
            target["robot_occlusion_possible"] = int(round(statistics.mean(occlusion)))
    for cluster in clusters:
        cluster["vote_count"] = len(cluster["votes"])
        cluster["views"] = sorted({str(item.get("view")) for item in cluster["votes"] if item.get("view")})
        cluster["repeats"] = sorted(
            {int(item["repeat"]) for item in cluster["votes"] if isinstance(item.get("repeat"), int)}
        )
        cluster["reasons"] = [
            str(item["reason"]) for item in cluster["votes"] if str(item.get("reason") or "").strip()
        ]
        cluster["detector_supported"] = False
        cluster["source"] = "vlm"
    return clusters


def detector_candidate(detector: dict[str, Any] | None) -> dict[str, Any] | None:
    if not detector or not detector.get("available") or not detector.get("detector_mlr"):
        return None
    counts = detector.get("counts") or []
    adjusted = detector.get("adjusted_counts") or counts
    n0 = detector.get("n0")
    has_over = any(count > n0 for count in counts) if n0 is not None else False
    has_under = any(count < n0 for count in counts) if n0 is not None else False
    if has_over and has_under:
        event_type = "both"
    elif has_over:
        event_type = "duplication"
    elif has_under:
        event_type = "disappearance"
    else:
        event_type = "uncertain"
    onset_frame = detector.get("detector_onset_frame")
    onset_sample = detector.get("detector_onset_sample")
    pair = detector.get("detector_first_confirmed_pair")
    if onset_frame is None:
        onset_frame = 0 if onset_sample is None else onset_sample
    return {
        "event_type": event_type,
        "onset_frame": int(onset_frame),
        "onset_sample_id": onset_sample,
        "first_suspicious_frame": None,
        "last_suspicious_frame": None,
        "votes": [],
        "vote_count": 0,
        "views": [],
        "repeats": [],
        "reasons": [
            "detector trace: "
            f"mlr_event=1 source={detector.get('detector_event_source')} "
            f"onset_sample={onset_sample} pair={pair} "
            f"N_t={counts} adjusted={adjusted}"
        ],
        "robot_occlusion_possible": 0,
        "detector_supported": True,
        "source": "detector",
        "detector_onset_sample": onset_sample,
        "detector_first_confirmed_pair": pair,
    }


def build_candidates(
    global_records: list[dict[str, Any]],
    detector: dict[str, Any] | None,
    *,
    key: str,
    total_frames: int,
    mlr_sample_count: int,
    max_candidates: int,
    tolerance: int = CANDIDATE_CLUSTER_TOLERANCE_FRAMES,
) -> list[dict[str, Any]]:
    """Union the Pass A clusters with the detector candidate and apply the support gate."""
    clusters = cluster_candidates(global_candidate_votes(global_records), tolerance=tolerance)
    det_candidate = detector_candidate(detector)
    if det_candidate is not None:
        attached = False
        for cluster in clusters:
            if _types_compatible(cluster["event_type"], det_candidate["event_type"]) and abs(
                int(cluster["onset_frame"]) - int(det_candidate["onset_frame"])
            ) <= tolerance:
                cluster["detector_supported"] = True
                cluster["source"] = "vlm+detector"
                cluster["detector_onset_sample"] = det_candidate.get("detector_onset_sample")
                cluster["detector_first_confirmed_pair"] = det_candidate.get(
                    "detector_first_confirmed_pair"
                )
                cluster["reasons"] = cluster["reasons"] + det_candidate["reasons"]
                attached = True
                break
        if not attached:
            clusters.append(det_candidate)

    for cluster in clusters:
        cluster["key"] = key
        cluster["total_frames"] = total_frames
        cluster["mlr_sample_count"] = mlr_sample_count
        cluster["onset_sample_id"] = cluster.get("onset_sample_id") or nearest_mlr_sample(
            total_frames, mlr_sample_count, int(cluster["onset_frame"])
        )
        vote_count = int(cluster.get("vote_count") or 0)
        detector_supported = bool(cluster.get("detector_supported"))
        cluster["support_gate"] = (
            vote_count >= 2
            or (vote_count >= 1 and detector_supported)
            or cluster.get("source") == "detector"
        )
        if detector_supported and cluster.get("detector_onset_sample") is not None:
            cluster["onset_sample_id"] = int(cluster["detector_onset_sample"])

    ranked = sorted(
        clusters,
        key=lambda cluster: (
            not cluster["support_gate"],
            not bool(cluster.get("detector_supported")),
            -int(cluster.get("vote_count") or 0),
            int(cluster["onset_frame"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for position, cluster in enumerate(ranked):
        cluster["candidate_id"] = f"{key}__c{position:02d}"
        if cluster["support_gate"] and len(selected) < max(1, max_candidates):
            selected.append(cluster)
        else:
            cluster["drop_reason"] = (
                "max_candidates_per_video"
                if cluster["support_gate"]
                else "insufficient_global_support"
            )
            dropped.append(cluster)
    for cluster in dropped:
        cluster["verdict"] = "insufficient_global_support"
        cluster["assessment"] = {
            "candidate_id": cluster["candidate_id"],
            "verdict": "insufficient_global_support",
            "reason": cluster["drop_reason"],
            "n_verify": 0,
            "n_refute": 0,
            "global_vote_count": int(cluster.get("vote_count") or 0),
            "detector_supported": bool(cluster.get("detector_supported")),
        }
    return selected + dropped


def candidate_window(
    candidate: dict[str, Any],
    *,
    total_frames: int,
    mlr_sample_count: int,
    before: int,
    after: int,
) -> dict[str, Any]:
    onset = int(candidate["onset_frame"])
    start, end = dense_window_bounds(total_frames, onset, before, after)
    onset_sample = candidate.get("onset_sample_id")
    if onset_sample is None:
        onset_sample = nearest_mlr_sample(total_frames, mlr_sample_count, onset)
    anchors: list[tuple[int, int | None]] = [(-1, onset_sample)]
    for sample_id in (onset_sample - 1, onset_sample + 1):
        frame = mlr_sample_frame(total_frames, mlr_sample_count, sample_id)
        if frame is not None:
            anchors.append((frame, sample_id))
    anchors = [(frame, sample) for frame, sample in anchors if frame >= 0]
    dense = [(frame, None) for frame in range(start, end + 1)]
    merged = merge_frame_requests(dense, anchors)
    return {
        "dense_start": start,
        "dense_end": end,
        "anchors": sorted(anchors, key=lambda item: (item[1] is None, item[1])),
        "frames": merged,
    }


def global_view_frames(
    *,
    key: str,
    total_frames: int,
    view: str,
    global_frame_count: int,
    mlr_sample_count: int,
) -> list[tuple[int, int | None]]:
    """Frame request list for one global view; the conditioning frame is handled separately."""
    if view == "mlr24":
        return mlr_sample_points(total_frames, mlr_sample_count)
    if view == "dense49":
        return [(frame, None) for frame in uniform_frame_indices(total_frames, global_frame_count)]
    raise ValueError(f"unknown global view: {view}")


# --------------------------------------------------------------------------------------
# Model calls
# --------------------------------------------------------------------------------------


def require_genai() -> Any:
    if gt is None or genai is None:
        raise SystemExit(
            "google-genai is required for real model calls; install it in the active "
            f"environment (import error: {_GENAI_IMPORT_ERROR})"
        )
    return gt


def endpoint_name() -> str:
    if os.getenv("DIFROST_API_TOKEN"):
        return "difrost_genai"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "google_genai"
    return "unset"


def require_credentials() -> str:
    """Fail fast (before any frame is decoded) when no credential is available."""
    endpoint = endpoint_name()
    if endpoint == "unset":
        raise SystemExit(
            "no Gemini credentials found: set DIFROST_API_TOKEN (repo difrost endpoint) or "
            "GEMINI_API_KEY / GOOGLE_API_KEY (direct google-genai). Tokens are never written to disk."
        )
    return endpoint


_CLIENT_TLS = threading.local()


def _difrost_client(timeout_seconds: float) -> Any:
    import ssl
    import uuid

    require_genai()
    base_url = os.getenv("DIFROST_GENAI_BASE_URL", "https://llm-api.cc/api/v1")
    host = os.getenv("DIFROST_HOST", "llm-api.cc")
    token = os.getenv("DIFROST_API_TOKEN")
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    timeout_ms = int(timeout_seconds * 1000) if timeout_seconds and timeout_seconds > 0 else None
    http_opts = gt.HttpOptions(
        base_url=base_url,
        api_version="genai",
        timeout=timeout_ms,
        headers={
            "Authorization": f"Bearer {token}",
            "Host": host,
            "X-Difrost-Session-Affinity": uuid.uuid4().hex,
        },
        async_client_args={"ssl": ssl_ctx},
        client_args={"verify": False},
    )
    return genai.Client(vertexai=False, api_key="placeholder", http_options=http_opts)


def get_client(timeout_seconds: float) -> Any:
    """Thread-local client: difrost endpoint when DIFROST_API_TOKEN is set, else google-genai."""
    endpoint = require_credentials()
    cached = getattr(_CLIENT_TLS, "client_cache", None)
    cache_key = (endpoint, float(timeout_seconds or 0))
    if cached is None or cached[0] != cache_key:
        if endpoint == "difrost_genai" and gemini_ref is not None:
            client = gemini_ref.get_gemini_client(timeout_seconds)
        elif endpoint == "difrost_genai":
            client = _difrost_client(timeout_seconds)
        else:
            require_genai()
            timeout_ms = (
                int(timeout_seconds * 1000) if timeout_seconds and timeout_seconds > 0 else None
            )
            client = genai.Client(http_options=gt.HttpOptions(timeout=timeout_ms))
        cached = (cache_key, client)
        _CLIENT_TLS.client_cache = cached
    return cached[1]


def call_model(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: list[bytes],
    schema: dict[str, Any],
    temperature: float,
    max_output_tokens: int,
    thinking_level: str,
    include_thoughts: bool,
    retries: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """One JSON-schema-constrained Gemini call with retries; returns parsed + raw text."""
    types = require_genai()
    parts: list[Any] = [types.Part.from_text(text=user_prompt)]
    parts.extend(
        types.Part.from_bytes(data=data, mime_type="image/jpeg") for data in images
    )
    contents = [types.Content(role="user", parts=parts)]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
        ),
    )
    started = time.time()
    raw_text = ""
    last_error = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            client = get_client(timeout_seconds)
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            texts, thoughts = response_text_parts(response)
            raw_text = "\n".join(texts) or getattr(response, "text", "") or ""
            parsed = parse_json_object(raw_text)
            return {
                "parsed": parsed,
                "raw_text": raw_text,
                "thought_text": "\n".join(thoughts) if thoughts else "",
                "attempts": attempt,
                "latency_sec": round(time.time() - started, 4),
                "error": None if parsed is not None else "unparsable_model_response",
            }
        except Exception as exc:  # retry transient API failures
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max(1, retries):
                time.sleep(min(2.0 * attempt, 8.0))
    return {
        "parsed": None,
        "raw_text": raw_text,
        "thought_text": "",
        "attempts": max(1, retries),
        "latency_sec": round(time.time() - started, 4),
        "error": f"model failed after {max(1, retries)} attempts: {last_error}",
    }


# --------------------------------------------------------------------------------------
# Verification and consensus aggregation
# --------------------------------------------------------------------------------------


def _mode(values: list[Any]) -> Any:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    counts: dict[Any, int] = {}
    for value in cleaned:
        counts[value] = counts.get(value, 0) + 1
    best = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    if len(best) > 1 and best[0][1] == best[1][1]:
        return None
    return best[0][0]


def _median_int(values: list[Any]) -> int | None:
    numbers = [int(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return int(round(statistics.median(numbers)))


def _valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if isinstance(record.get("parsed"), dict)]


def assess_candidate(
    candidate: dict[str, Any],
    verify_records: list[dict[str, Any]],
    refute_records: list[dict[str, Any]],
    *,
    task_conserving: bool,
) -> dict[str, Any]:
    """Apply the frozen event-level acceptance rules (doc section 7.2)."""
    assessment: dict[str, Any] = {
        "candidate_id": candidate.get("candidate_id"),
        "event_type": candidate.get("event_type"),
        "onset_frame": candidate.get("onset_frame"),
        "onset_sample_id": candidate.get("onset_sample_id"),
        "global_vote_count": int(candidate.get("vote_count") or 0),
        "detector_supported": bool(candidate.get("detector_supported")),
        "source": candidate.get("source"),
        "support_gate": bool(candidate.get("support_gate")),
        "n_verify": len(verify_records),
        "n_refute": len(refute_records),
        "task_count_conserving": bool(task_conserving),
    }
    if not candidate.get("support_gate"):
        assessment.update(
            {
                "verdict": "insufficient_global_support",
                "reason": candidate.get("drop_reason", "insufficient_global_support"),
            }
        )
        return assessment

    verify = _valid_records(verify_records)
    refute = _valid_records(refute_records)
    verify_verdicts = [str(record["parsed"].get("verdict") or "") for record in verify]
    refute_verdicts = [str(record["parsed"].get("verdict") or "") for record in refute]
    benign_high = [
        record
        for record in refute
        if str(record["parsed"].get("benign_explanation") or "uncertain") not in {"none", "uncertain"}
        and str(record["parsed"].get("confidence") or "") == "high"
    ]
    confirmed_votes = sum(1 for verdict in verify_verdicts if verdict == "confirmed_mlr")
    rejected_votes = sum(1 for verdict in verify_verdicts if verdict == "rejected")
    b_confirmed = bool(verify) and confirmed_votes * 2 > len(verify)
    b_rejected = bool(verify) and rejected_votes * 2 > len(verify)
    c_rejected = any(verdict == "rejected" for verdict in refute_verdicts)

    confirmed_records = [
        record for record in verify if str(record["parsed"].get("verdict") or "") == "confirmed_mlr"
    ]
    persistence = _median_int(
        [record["parsed"].get("persistence_mlr_samples") for record in confirmed_records]
    )
    onsets: list[int] = [
        int(record["parsed"]["onset_frame"])
        for record in confirmed_records
        if isinstance(record["parsed"].get("onset_frame"), (int, float))
    ]
    onsets.append(int(candidate["onset_frame"]))
    onset_spread_frames = (max(onsets) - min(onsets)) if onsets else None
    sample_ids: list[int] = []
    for onset in onsets:
        sample_id = candidate.get("onset_sample_id")
        if onset == int(candidate["onset_frame"]) and sample_id is not None:
            sample_ids.append(int(sample_id))
            continue
        total_frames = candidate.get("total_frames")
        if total_frames:
            nearest = nearest_mlr_sample(
                int(total_frames), int(candidate.get("mlr_sample_count") or 24), int(onset)
            )
            if nearest is not None:
                sample_ids.append(int(nearest))
    onset_spread_samples = (max(sample_ids) - min(sample_ids)) if sample_ids else None
    onset_consistent = bool(onsets) and (
        (onset_spread_frames is not None and onset_spread_frames <= ONSET_TOLERANCE_FRAMES)
        or (onset_spread_samples is not None and onset_spread_samples <= ONSET_TOLERANCE_SAMPLES)
    )
    persistence_ok = persistence is not None and persistence >= REQUIRED_CONSECUTIVE_SAMPLES

    reasons: list[str] = []
    if not task_conserving:
        reasons.append("task_not_count_conserving")
    if not b_confirmed:
        reasons.append("pass_b_not_confirmed")
    if c_rejected:
        reasons.append("refutation_rejected")
    if benign_high:
        reasons.append(
            "high_confidence_benign_explanation:"
            + ",".join(
                sorted({str(record["parsed"].get("benign_explanation")) for record in benign_high})
            )
        )
    if not persistence_ok:
        reasons.append(f"persistence_mlr_samples={persistence}")
    if not onset_consistent:
        reasons.append("onset_inconsistent")

    if b_confirmed and not c_rejected and not benign_high and persistence_ok and onset_consistent and task_conserving:
        verdict = "confirmed"
    elif b_rejected or c_rejected:
        verdict = "rejected"
    else:
        verdict = "uncertain"

    representative = None
    pool = confirmed_records or verify or refute
    if pool:
        representative = sorted(
            pool,
            key=lambda record: (
                int(record["parsed"].get("onset_frame") or 0),
                str(record.get("record_id") or ""),
            ),
        )[len(pool) // 2]

    assessment.update(
        {
            "verdict": verdict,
            "reason": "; ".join(reasons) if reasons else "all_acceptance_rules_passed",
            "verify_verdicts": verify_verdicts,
            "refute_verdicts": refute_verdicts,
            "pass_b_confirmed_majority": b_confirmed,
            "pass_b_rejected_majority": b_rejected,
            "refutation_rejected": c_rejected,
            "high_confidence_benign": [
                {
                    "benign_explanation": record["parsed"].get("benign_explanation"),
                    "confidence": record["parsed"].get("confidence"),
                    "record_id": record.get("record_id"),
                }
                for record in benign_high
            ],
            "persistence_mlr_samples": persistence,
            "persistence_original_frames": _median_int(
                [record["parsed"].get("persistence_original_frames") for record in confirmed_records]
            ),
            "onset_spread_frames": onset_spread_frames,
            "onset_spread_samples": onset_spread_samples,
            "onset_consistent": onset_consistent,
            "representative_record_id": representative.get("record_id") if representative else None,
            "representative_parsed": representative["parsed"] if representative else None,
        }
    )
    return assessment


def video_consensus(
    *,
    item: dict[str, Any],
    key: str,
    total_frames: int,
    eligible: bool,
    n0: int | None,
    detector: dict[str, Any] | None,
    global_records: list[dict[str, Any]],
    candidate_assessments: list[dict[str, Any]],
    mlr_sample_count: int,
    errors: list[str],
) -> dict[str, Any]:
    """Aggregate global votes, candidate assessments and the detector trace into one row."""
    valid_global = _valid_records(global_records)
    conserving_votes = [
        int(record["parsed"]["task_count_conserving"])
        for record in valid_global
        if isinstance(record["parsed"].get("task_count_conserving"), (int, float))
    ]
    conserving_yes = sum(1 for value in conserving_votes if value == 1)
    conserving_no = sum(1 for value in conserving_votes if value == 0)
    task_conserving = conserving_yes > conserving_no
    task_conserving_tie = conserving_yes == conserving_no and conserving_votes != []
    candidate_votes = sum(
        1
        for record in valid_global
        if isinstance(record["parsed"].get("has_mlr_candidate"), (int, float))
        and int(record["parsed"]["has_mlr_candidate"]) == 1
    )
    detector_known = bool(detector and detector.get("available"))
    detector_positive = int(detector.get("detector_mlr")) if detector_known else None

    confirmed = [item_ for item_ in candidate_assessments if item_.get("verdict") == "confirmed"]
    rejected = [item_ for item_ in candidate_assessments if item_.get("verdict") == "rejected"]
    uncertain = [item_ for item_ in candidate_assessments if item_.get("verdict") == "uncertain"]
    unsupported = [
        item_
        for item_ in candidate_assessments
        if item_.get("verdict") == "insufficient_global_support"
    ]

    reasons: list[str] = []
    gemini_confirmed: int | None
    if not eligible:
        verdict = "ineligible"
        gemini_confirmed = None
        reasons.append("ineligible: detector_eligible=0 or n0=0")
        primary = confirmed[0] if confirmed else None
    elif confirmed:
        primary = confirmed[0]
        if detector_positive is None:
            verdict = "confirmed_mlr"
        elif detector_positive == 1:
            verdict = "confirmed_mlr"
        else:
            verdict = "vlm_positive_detector_negative"
            reasons.append("detector_negative_but_gemini_confirmed")
        gemini_confirmed = 1
    elif detector_positive == 1 and not uncertain:
        primary = rejected[0] if rejected else None
        if rejected:
            verdict = "detector_positive_vlm_rejected"
            gemini_confirmed = 0
            reasons.append("detector_positive_but_pass_b_rejected")
        else:
            verdict = "detector_only_unverified"
            gemini_confirmed = None
            reasons.append("detector_positive_candidate_without_global_support")
    elif uncertain:
        primary = uncertain[0]
        verdict = "uncertain_need_human"
        gemini_confirmed = None
        reasons.append(uncertain[0].get("reason") or "uncertain")
    elif rejected:
        primary = rejected[0]
        verdict = "rejected"
        gemini_confirmed = 0
        reasons.append("pass_b_rejected" if detector_positive != 1 else "detector_positive_pass_b_rejected")
    elif detector_positive == 1:
        primary = None
        verdict = "detector_only_unverified"
        gemini_confirmed = None
        reasons.append("detector_positive_without_vlm_candidate")
    elif unsupported:
        primary = None
        verdict = "uncertain_need_human"
        gemini_confirmed = None
        reasons.append("candidates_without_global_support")
    elif candidate_votes == 0 and valid_global:
        primary = None
        verdict = "rejected"
        gemini_confirmed = 0
        reasons.append("no_mlr_candidate_proposed_by_global_pass")
    else:
        primary = None
        verdict = "uncertain_need_human"
        gemini_confirmed = None
        if not valid_global:
            reasons.append("no_valid_global_votes")
        else:
            reasons.append("global_votes_disagree")

    if errors:
        reasons.append("errors:" + str(len(errors)))

    representative = (primary or {}).get("representative_parsed") or {}
    event_type = str(representative.get("event_type") or (primary or {}).get("event_type") or "")
    if event_type in {"", "none"}:
        event_type = ""
    onset_frame = representative.get("onset_frame", (primary or {}).get("onset_frame"))
    end_frame = representative.get("end_frame")
    persistence_frames = representative.get("persistence_original_frames", (primary or {}).get("persistence_original_frames"))
    persistence_samples = representative.get("persistence_mlr_samples", (primary or {}).get("persistence_mlr_samples"))
    pair = representative.get("first_confirmed_mlr_pair")
    if pair is None:
        pair = (primary or {}).get("first_confirmed_pair") or (
            detector.get("detector_first_confirmed_pair") if detector_known else None
        )
    occlusion = str(representative.get("occlusion_assessment") or "")
    confidence = str(representative.get("confidence") or (primary or {}).get("confidence") or "")
    agree = (
        int(detector_positive == gemini_confirmed)
        if (detector_positive is not None and gemini_confirmed is not None)
        else None
    )

    row: dict[str, Any] = {
        "key": key,
        "video_path": item.get("video_path"),
        "instruction": item.get("instruction") or "",
        "target_query": item.get("target_query") or "",
        "total_frames": total_frames,
        "eligible": int(bool(eligible)),
        "n0": n0,
        "primary_detector_mlr": detector_positive,
        "gemini_confirmed_mlr": gemini_confirmed,
        "verdict": verdict,
        "event_type": event_type,
        "onset_frame": onset_frame,
        "end_frame": end_frame,
        "persistence_original_frames": persistence_frames,
        "persistence_mlr_samples": persistence_samples,
        "first_confirmed_mlr_pair": pair,
        "occlusion_assessment": occlusion,
        "confidence": confidence,
        "detector_gemini_agree": agree,
        "uncertainty_reason": "; ".join(reasons) if reasons else "",
        "task_count_conserving": int(task_conserving),
        "task_count_conserving_votes": {"yes": conserving_yes, "no": conserving_no},
        "task_count_conserving_tie": bool(task_conserving_tie),
        "n_global_records": len(global_records),
        "n_global_valid": len(valid_global),
        "global_candidate_votes": candidate_votes,
        "candidates_total": len(candidate_assessments),
        "confirmed_candidate_ids": [item_.get("candidate_id") for item_ in confirmed],
        "rejected_candidate_ids": [item_.get("candidate_id") for item_ in rejected],
        "uncertain_candidate_ids": [item_.get("candidate_id") for item_ in uncertain],
        "unsupported_candidate_ids": [item_.get("candidate_id") for item_ in unsupported],
        "detector": {
            "available": detector_known,
            "mlr_event": detector_positive,
            "n0": (detector or {}).get("n0"),
            "event_source": (detector or {}).get("detector_event_source"),
            "onset_sample": (detector or {}).get("detector_onset_sample"),
            "onset_frame": (detector or {}).get("detector_onset_frame"),
            "counts": (detector or {}).get("counts"),
            "adjusted_counts": (detector or {}).get("adjusted_counts"),
            "occlusion_flags": (detector or {}).get("occlusion_flags"),
            "max_count": (detector or {}).get("max_count"),
            "error": (detector or {}).get("error"),
        },
        "candidates": candidate_assessments,
        "errors": errors,
        "created_at": now_iso(),
    }
    return row


def auxiliary_failure_summary(global_records: list[dict[str, Any]]) -> dict[str, int]:
    """Count videos whose majority global vote flags each auxiliary failure (doc section 10.4)."""
    totals = {"identity_drift": 0, "deformation": 0, "teleportation_without_count_change": 0}
    for record in _valid_records(global_records):
        auxiliary = record["parsed"].get("auxiliary_failures")
        if not isinstance(auxiliary, dict):
            continue
        for name in totals:
            value = auxiliary.get(name)
            if isinstance(value, (int, float)) and int(value) == 1:
                totals[name] += 1
    return totals


def consensus_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in CONSENSUS_COLUMNS:
        value = row.get(column)
        if column == "first_confirmed_mlr_pair" and value is not None:
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if value is None:
            value = ""
        out[column] = value
    return out
# --------------------------------------------------------------------------------------
# Run configuration, resume and request planning
# --------------------------------------------------------------------------------------

# Fields that must match before a resumed run is allowed to reuse existing records.
PROTOCOL_FIELDS = (
    "model",
    "global_frame_count",
    "mlr_sample_count",
    "dense_window_before",
    "dense_window_after",
    "jpeg_quality",
    "max_image_side",
    "temperature",
    "thinking_level",
    "repeats",
    "prompt_sha256",
    "schema_version",
    "frame_label_protocol",
)

FRAME_LABEL_PROTOCOL = (
    "key=<video key> frame=<0-based original frame index>[ sample=<MLR sample id>]"
    "[ conditioning=1]; burned into the top-left corner of every image"
)

MLR_SAMPLE_FORMULA = (
    "sample i in 0..S-1 -> frame index round(i*(F-1)/(S-1)) on the original frame axis, "
    "duplicates removed (keeps the smallest sample id)"
)

AGGREGATION_SUMMARY = {
    "task_count_conserving": "strict majority over all valid Pass A votes",
    "candidate_support_gate": (
        ">=2 Pass A votes in the same candidate window, or >=1 vote plus a detector "
        "candidate in the same window, or a detector-only candidate"
    ),
    "pass_b": "strict majority of confirmed_mlr over the valid Pass B records of the candidate",
    "pass_c": (
        "any rejected verdict blocks the event; any high-confidence benign explanation "
        "(benign_explanation not in {none, uncertain} at confidence == high) blocks it too"
    ),
    "persistence": "median persistence_mlr_samples of the confirming Pass B records must be >= 2",
    "onset": "onset spread <= 6 original frames or <= 2 MLR sampling points",
    "primary_detector": (
        "recomputed from the detector trace with the frozen Algorithm 1 rules; it is never "
        "overwritten by the VLM"
    ),
    "denominators": (
        "detector rates use the detector-known eligible videos; Gemini rates use the eligible "
        "videos with a valid Gemini verdict; the two denominators are never mixed"
    ),
}

PLACEHOLDER = "(not provided in the manifest)"

RECORD_FILES = (
    GLOBAL_RECORDS_FILE,
    VERIFY_RECORDS_FILE,
    REFUTE_RECORDS_FILE,
    CONSENSUS_JSONL_FILE,
    CONSENSUS_CSV_FILE,
    SUMMARY_FILE,
    DRY_RUN_FILE,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prompt_hashes() -> dict[str, str]:
    return {
        "system": sha256_text(build_system_prompt()),
        "global": sha256_text(GLOBAL_USER_TEMPLATE),
        "verify": sha256_text(VERIFY_USER_TEMPLATE + "\n" + DENSE_WINDOW_PROMPT_NOTE),
        "refute": sha256_text(REFUTE_USER_TEMPLATE + "\n" + DENSE_WINDOW_PROMPT_NOTE),
    }


def build_run_config(
    args: argparse.Namespace,
    manifest_paths: list[str],
    detector_path: str | None,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Frozen protocol description; contains endpoint *names* only, never tokens."""
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": created_at or now_iso(),
        "updated_at": now_iso(),
        "model": args.model,
        "endpoint": endpoint_name(),
        "endpoint_note": (
            "only the endpoint name is recorded; API tokens are never written to disk"
        ),
        "token_env_vars": ["DIFROST_API_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "temperature": args.temperature,
        "thinking_level": args.thinking_level,
        "include_thoughts": bool(args.include_thoughts),
        "model_max_tokens": args.model_max_tokens,
        "model_retries": args.model_retries,
        "model_timeout": args.model_timeout,
        "global_frame_count": args.global_frame_count,
        "mlr_sample_count": args.mlr_sample_count,
        "mlr_sample_formula": MLR_SAMPLE_FORMULA,
        "dense_window_before": args.dense_window_before,
        "dense_window_after": args.dense_window_after,
        "jpeg_quality": args.jpeg_quality,
        "max_image_side": args.max_image_side,
        "frame_label_protocol": FRAME_LABEL_PROTOCOL,
        "repeats": args.repeats,
        "global_views": list(GLOBAL_VIEWS),
        "repeats_semantics": "each global view and each Pass B/C call is issued --repeats times",
        "max_candidates_per_video": args.max_candidates_per_video,
        "frame_cache": bool(args.frame_cache),
        "concurrency": args.concurrency,
        "resume": bool(args.resume),
        "limit": args.limit,
        "manifests": list(manifest_paths),
        "detector_records": str(detector_path) if detector_path else None,
        "prompt_sha256": prompt_hashes(),
        "schema_version": {"global": SCHEMA_VERSION_GLOBAL, "verify": SCHEMA_VERSION_VERIFY},
        "tolerances": {
            "occlusion_tau": OCCLUSION_TAU,
            "required_consecutive_samples": REQUIRED_CONSECUTIVE_SAMPLES,
            "onset_tolerance_frames": ONSET_TOLERANCE_FRAMES,
            "onset_tolerance_samples": ONSET_TOLERANCE_SAMPLES,
            "candidate_cluster_tolerance_frames": CANDIDATE_CLUSTER_TOLERANCE_FRAMES,
        },
        "aggregation": dict(AGGREGATION_SUMMARY),
    }


def protocol_differences(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Names of the frozen protocol fields that differ between two run configs."""
    differences: list[str] = []
    for field in PROTOCOL_FIELDS:
        if previous.get(field) != current.get(field):
            differences.append(field)
    return differences


def read_records(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def load_run_records(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "global": read_records(output_dir / GLOBAL_RECORDS_FILE),
        "verify": read_records(output_dir / VERIFY_RECORDS_FILE),
        "refute": read_records(output_dir / REFUTE_RECORDS_FILE),
        "consensus": read_records(output_dir / CONSENSUS_JSONL_FILE),
    }


def reset_output_records(output_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in RECORD_FILES:
        path = output_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


class JsonlWriter:
    """Thread-safe append-only JSONL writer; every record is flushed immediately."""

    def __init__(self, path: Path, lock: threading.Lock) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = lock
        self._handle = path.open("a", encoding="utf-8")

    def append(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def global_record_id(key: str, view: str, repeat: int) -> str:
    return f"{key}__A__{view}__r{int(repeat):02d}"


def dense_record_id(key: str, which: str, candidate_id: Any, repeat: int) -> str:
    return f"{key}__{which}__{candidate_id}__r{int(repeat):02d}"


def global_dedup_key(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (record.get("key"), record.get("view"), record.get("repeat"))


def dense_dedup_key(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (record.get("key"), record.get("candidate_id"), record.get("repeat"))


def parsed_or_issues(
    parsed: dict[str, Any] | None, required: Any, expected_schema: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Return ``(parsed, issues)``; a response missing required fields is not a valid vote."""
    if parsed is None:
        return None, ["parsed_json_missing"]
    issues = [f"missing_required:{name}" for name in required if name not in parsed]
    version = parsed.get("schema_version")
    if version != expected_schema:
        issues.append(f"schema_version_mismatch:{version}")
    if issues:
        return None, issues
    return parsed, []


def perform_call(
    *,
    args: argparse.Namespace,
    system_prompt: str,
    prompt: str,
    images: list[bytes],
    schema: dict[str, Any],
    required: Any,
    expected_schema: str,
) -> dict[str, Any]:
    result = call_model(
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        images=images,
        schema=schema,
        temperature=args.temperature,
        max_output_tokens=args.model_max_tokens,
        thinking_level=args.thinking_level,
        include_thoughts=bool(args.include_thoughts),
        retries=args.model_retries,
        timeout_seconds=args.model_timeout,
    )
    raw_parsed = result.get("parsed")
    parsed, issues = parsed_or_issues(raw_parsed, required, expected_schema)
    out: dict[str, Any] = {
        "parsed": parsed,
        "schema_issues": issues,
        "raw_text": result.get("raw_text"),
        "thought_text": result.get("thought_text"),
        "attempts": result.get("attempts"),
        "latency_sec": result.get("latency_sec"),
        "error": result.get("error"),
    }
    if issues and raw_parsed is not None:
        out["parsed_raw"] = raw_parsed
    return out


def prompt_text(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else PLACEHOLDER


def materialize_view(
    *,
    key: str,
    video_path: Path,
    requests: list[tuple[int, int | None]],
    conditioning: tuple[bytes, str, str],
    jpeg_quality: int,
    max_image_side: int,
    cache_dir: Path | None,
) -> tuple[list[bytes], list[str], list[dict[str, Any]]]:
    """Decode and label the requested frames; returns (images, labels, frame_records)."""
    labeled = [
        (int(frame), frame_label(key, int(frame), None if sample is None else int(sample)))
        for frame, sample in requests
    ]
    payloads = extract_labeled_frames(
        video_path,
        labeled,
        jpeg_quality=jpeg_quality,
        max_image_side=max_image_side,
        cache_dir=cache_dir,
    )
    images: list[bytes] = [conditioning[0]]
    labels: list[str] = [conditioning[1]]
    frame_records: list[dict[str, Any]] = []
    for frame, sample in requests:
        frame_index = int(frame)
        data = payloads[frame_index]
        label = frame_label(key, frame_index, None if sample is None else int(sample))
        images.append(data)
        labels.append(label)
        frame_records.append(
            {
                "frame_index": frame_index,
                "sample_id": None if sample is None else int(sample),
                "label": label,
                "jpeg_bytes": len(data),
                "jpeg_sha256": sha256_bytes(data),
            }
        )
    return images, labels, frame_records


def conditioning_payload(
    *,
    item: dict[str, Any],
    key: str,
    video_path: Path,
    jpeg_quality: int,
    max_image_side: int,
    cache_dir: Path | None,
) -> tuple[bytes, str, str]:
    """Conditioning frame bytes: the manifest input frame, otherwise original frame 0."""
    conditioning_label = frame_label(key, 0, conditioning=True)
    frame_path = item.get("conditioning_frame_path")
    if frame_path and Path(str(frame_path)).is_file():
        data, label = encode_conditioning_frame(
            Path(str(frame_path)),
            key=key,
            jpeg_quality=jpeg_quality,
            max_image_side=max_image_side,
            cache_dir=cache_dir,
        )
        return data, label, "manifest_conditioning_frame"
    payloads = extract_labeled_frames(
        video_path,
        [(0, conditioning_label)],
        jpeg_quality=jpeg_quality,
        max_image_side=max_image_side,
        cache_dir=cache_dir,
    )
    if 0 not in payloads:
        raise RuntimeError(f"could not decode the conditioning frame from {video_path}")
    return payloads[0], conditioning_label, "video_frame_0"


def task_conserving_from_global(global_records: list[dict[str, Any]]) -> bool:
    """Mirror of the majority rule used by :func:`video_consensus`."""
    values = [
        int(record["parsed"]["task_count_conserving"])
        for record in _valid_records(global_records)
        if isinstance(record["parsed"].get("task_count_conserving"), (int, float))
    ]
    return sum(1 for value in values if value == 1) > sum(1 for value in values if value == 0)


def prepare_video(
    item: dict[str, Any], *, args: argparse.Namespace, detector: dict[str, Any] | None
) -> dict[str, Any]:
    """Resolve frames, N0, eligibility and the detector hint for one manifest row."""
    key = str(item["key"])
    video_path = Path(str(item["video_path"]))
    blocking: list[str] = list(item.get("errors") or [])
    notes: list[str] = []
    detector_known = bool(detector and detector.get("available"))
    n0 = item.get("n0")
    if n0 is None and detector_known:
        n0 = detector.get("n0")
    if n0 is None:
        blocking.append("missing_n0")
    if not video_path.is_file():
        blocking.append("video_missing")

    total_frames = item.get("frame_count")
    if total_frames is None and video_path.is_file():
        try:
            total_frames = probe_frame_count(video_path)
        except Exception as exc:
            blocking.append(f"frame_count_unavailable:{type(exc).__name__}: {exc}")
    if isinstance(total_frames, int) and total_frames <= 0:
        blocking.append("frame_count_zero")

    if detector_known and detector.get("eligible") is not None:
        eligible = bool(detector.get("eligible")) and int(n0 or 0) > 0
    elif n0 is not None:
        eligible = int(n0) > 0
    else:
        eligible = True
        notes.append("eligibility_unknown_assumed_eligible")
    if not detector_known:
        notes.append("detector_trace_unavailable")

    return {
        "key": key,
        "video_path": video_path,
        "total_frames": total_frames,
        "n0": None if n0 is None else int(n0),
        "eligible": eligible,
        "detector": detector,
        "detector_hint": detector_hint_text(detector),
        "blocking_errors": blocking,
        "notes": notes,
        "errors": blocking + notes,
        "usable": not blocking,
    }


def plan_views(
    *, key: str, total_frames: int, global_frame_count: int, mlr_sample_count: int
) -> dict[str, list[tuple[int, int | None]]]:
    return {
        view: global_view_frames(
            key=key,
            total_frames=total_frames,
            view=view,
            global_frame_count=global_frame_count,
            mlr_sample_count=mlr_sample_count,
        )
        for view in GLOBAL_VIEWS
    }


# --------------------------------------------------------------------------------------
# Request execution (one record per model call)
# --------------------------------------------------------------------------------------


def label_requests(key: str, requests: list[tuple[int, int | None]]) -> list[str]:
    """Frame labels for a request list, identical to the ones burned into the images."""
    return [
        frame_label(key, int(frame), None if sample is None else int(sample))
        for frame, sample in requests
    ]


def conditioning_record(conditioning: tuple[bytes, str, str]) -> dict[str, Any]:
    data, label, source = conditioning
    return {
        "label": label,
        "source": source,
        "jpeg_bytes": len(data),
        "jpeg_sha256": sha256_bytes(data),
    }


def fill_detector_onset_frame(
    detector: dict[str, Any] | None, *, mlr_sample_count: int, total_frames: Any
) -> dict[str, Any] | None:
    """Resolve the detector onset frame once the frame count is known."""
    if not detector or not detector.get("available"):
        return detector
    if detector.get("detector_onset_frame") is not None:
        return detector
    sample = detector.get("detector_onset_sample")
    if sample is None or not total_frames:
        return detector
    frame = mlr_sample_frame(int(total_frames), int(mlr_sample_count), int(sample))
    if frame is None:
        return detector
    filled = dict(detector)
    filled["detector_onset_frame"] = frame
    return filled


def run_global_call(
    *,
    args: argparse.Namespace,
    item: dict[str, Any],
    plan: dict[str, Any],
    view: str,
    repeat: int,
    system_prompt: str,
    conditioning_for: Callable[[str], tuple[bytes, str, str]],
    cache_dir: Path | None,
) -> dict[str, Any]:
    """One Pass A request. Failures are captured in the record; this never raises."""
    key = str(plan["key"])
    total_frames = int(plan["total_frames"])
    n0 = plan.get("n0")
    record: dict[str, Any] = {
        "record_id": global_record_id(key, view, repeat),
        "pass": "A",
        "view": view,
        "key": key,
        "video_path": plan.get("video_path"),
        "repeat": int(repeat),
        "model": args.model,
        "expected_schema_version": SCHEMA_VERSION_GLOBAL,
        "system_prompt_sha256": sha256_text(system_prompt),
        "created_at": now_iso(),
        "total_frames": total_frames,
        "n0": None if n0 is None else int(n0),
        "conditioning": None,
        "frames": [],
        "n_images": 0,
        "prompt_sha256": None,
        "parsed": None,
        "schema_issues": [],
        "raw_text": None,
        "thought_text": None,
        "attempts": None,
        "latency_sec": None,
        "error": None,
    }
    try:
        conditioning = conditioning_for(key)
        requests = global_view_frames(
            key=key,
            total_frames=total_frames,
            view=view,
            global_frame_count=int(args.global_frame_count),
            mlr_sample_count=int(args.mlr_sample_count),
        )
        images, labels, frame_records = materialize_view(
            key=key,
            video_path=Path(str(plan["video_path"])),
            requests=requests,
            conditioning=conditioning,
            jpeg_quality=int(args.jpeg_quality),
            max_image_side=int(args.max_image_side),
            cache_dir=cache_dir,
        )
        prompt = build_global_prompt(
            instruction=prompt_text(item.get("instruction")),
            target_query=prompt_text(item.get("target_query")),
            n0=0 if n0 is None else int(n0),
            conditioning_label=conditioning[1],
            labels=labels,
            detector_hint=plan.get("detector_hint"),
        )
        record["conditioning"] = conditioning_record(conditioning)
        record["frames"] = frame_records
        record["n_images"] = len(images)
        record["prompt_sha256"] = sha256_text(prompt)
        record.update(
            perform_call(
                args=args,
                system_prompt=system_prompt,
                prompt=prompt,
                images=images,
                schema=MLR_GLOBAL_SCHEMA,
                required=MLR_GLOBAL_SCHEMA["required"],
                expected_schema=SCHEMA_VERSION_GLOBAL,
            )
        )
    except Exception as exc:  # frame decoding, prompt building, network: record, never raise
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["finished_at"] = now_iso()
    return record


def run_dense_call(
    *,
    args: argparse.Namespace,
    item: dict[str, Any],
    plan: dict[str, Any],
    candidate: dict[str, Any],
    which: str,
    repeat: int,
    system_prompt: str,
    conditioning_for: Callable[[str], tuple[bytes, str, str]],
    cache_dir: Path | None,
) -> dict[str, Any]:
    """One Pass B (``which="B"``) or Pass C (``which="C"``) request for one candidate."""
    key = str(plan["key"])
    total_frames = int(plan["total_frames"])
    n0 = plan.get("n0")
    candidate_id = str(candidate.get("candidate_id") or "")
    record: dict[str, Any] = {
        "record_id": dense_record_id(key, which, candidate_id, repeat),
        "pass": which,
        "which": which,
        "key": key,
        "video_path": plan.get("video_path"),
        "candidate_id": candidate_id,
        "candidate_event_type": candidate.get("event_type"),
        "candidate_onset_frame": candidate.get("onset_frame"),
        "candidate_onset_sample_id": candidate.get("onset_sample_id"),
        "candidate_support_gate": bool(candidate.get("support_gate")),
        "candidate_source": candidate.get("source"),
        "repeat": int(repeat),
        "model": args.model,
        "expected_schema_version": SCHEMA_VERSION_VERIFY,
        "system_prompt_sha256": sha256_text(system_prompt),
        "created_at": now_iso(),
        "total_frames": total_frames,
        "n0": None if n0 is None else int(n0),
        "dense_start": None,
        "dense_end": None,
        "conditioning": None,
        "frames": [],
        "n_images": 0,
        "candidate_summary": None,
        "prompt_sha256": None,
        "parsed": None,
        "schema_issues": [],
        "raw_text": None,
        "thought_text": None,
        "attempts": None,
        "latency_sec": None,
        "error": None,
    }
    try:
        conditioning = conditioning_for(key)
        window = candidate_window(
            candidate,
            total_frames=total_frames,
            mlr_sample_count=int(args.mlr_sample_count),
            before=int(args.dense_window_before),
            after=int(args.dense_window_after),
        )
        record["dense_start"] = window["dense_start"]
        record["dense_end"] = window["dense_end"]
        images, labels, frame_records = materialize_view(
            key=key,
            video_path=Path(str(plan["video_path"])),
            requests=window["frames"],
            conditioning=conditioning,
            jpeg_quality=int(args.jpeg_quality),
            max_image_side=int(args.max_image_side),
            cache_dir=cache_dir,
        )
        summary = build_candidate_summary(candidate, plan.get("detector"))
        builder = build_verify_prompt if which == "B" else build_refute_prompt
        prompt = builder(
            event_id=candidate_id,
            instruction=prompt_text(item.get("instruction")),
            target_query=prompt_text(item.get("target_query")),
            n0=0 if n0 is None else int(n0),
            candidate_summary=summary,
            conditioning_label=conditioning[1],
            labels=labels,
        )
        record["candidate_summary"] = summary
        record["conditioning"] = conditioning_record(conditioning)
        record["frames"] = frame_records
        record["n_images"] = len(images)
        record["prompt_sha256"] = sha256_text(prompt)
        record.update(
            perform_call(
                args=args,
                system_prompt=system_prompt,
                prompt=prompt,
                images=images,
                schema=MLR_VERIFY_SCHEMA,
                required=MLR_VERIFY_SCHEMA["required"],
                expected_schema=SCHEMA_VERSION_VERIFY,
            )
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["finished_at"] = now_iso()
    return record


# --------------------------------------------------------------------------------------
# Protocol orchestration
# --------------------------------------------------------------------------------------


def run_protocol(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    detectors: dict[str, dict[str, Any]],
    output_dir: Path,
    existing: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Run Pass A (global), then Pass B/C per candidate, then aggregate one row per video.

    Scheduling rule: every global call of a video must finish before its candidates can be
    clustered, and every dense call of a candidate must finish before that video is
    aggregated. Records are appended as soon as they complete, so an interrupted run is
    resumable without repeating finished requests.
    """
    repeats = max(1, int(args.repeats))
    concurrency = max(1, int(args.concurrency))
    mlr_sample_count = int(args.mlr_sample_count)
    cache_dir = (output_dir / FRAME_CACHE_DIR) if args.frame_cache else None
    system_prompt = build_system_prompt()

    write_lock = threading.Lock()
    writers = {
        "global": JsonlWriter(output_dir / GLOBAL_RECORDS_FILE, write_lock),
        "verify": JsonlWriter(output_dir / VERIFY_RECORDS_FILE, write_lock),
        "refute": JsonlWriter(output_dir / REFUTE_RECORDS_FILE, write_lock),
        "consensus": JsonlWriter(output_dir / CONSENSUS_JSONL_FILE, write_lock),
    }
    records_by_key: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def buckets_for(key: str) -> dict[str, list[dict[str, Any]]]:
        return records_by_key.setdefault(key, {"global": [], "verify": [], "refute": []})

    for bucket in ("global", "verify", "refute"):
        for record in existing.get(bucket) or []:
            buckets_for(str(record.get("key")))[bucket].append(record)
    existing_consensus = {str(row.get("key")): row for row in (existing.get("consensus") or [])}
    have_global = {global_dedup_key(record) for record in (existing.get("global") or [])}
    have_verify = {dense_dedup_key(record) for record in (existing.get("verify") or [])}
    have_refute = {dense_dedup_key(record) for record in (existing.get("refute") or [])}

    plan_by_key: dict[str, dict[str, Any]] = {}
    plans: list[dict[str, Any]] = []
    for item in items:
        key = str(item["key"])
        raw_detector = detector_for_item(item, detectors)
        detector = None
        if raw_detector is not None and raw_detector.get("available"):
            detector = detector_event_from_trace(
                raw_detector,
                mlr_sample_count=mlr_sample_count,
                total_frames=item.get("frame_count"),
            )
        plan = prepare_video(item, args=args, detector=detector)
        plan["item"] = item
        plan["detector"] = fill_detector_onset_frame(
            detector, mlr_sample_count=mlr_sample_count, total_frames=plan.get("total_frames")
        )
        plan["detector_hint"] = detector_hint_text(plan["detector"])
        plan_by_key[key] = plan
        plans.append(plan)

    remaining: dict[str, int] = {}
    for plan in plans:
        key = plan["key"]
        pending = 0
        if plan["usable"] and plan["eligible"] and key not in existing_consensus:
            for view in GLOBAL_VIEWS:
                for repeat in range(1, repeats + 1):
                    if (key, view, repeat) not in have_global:
                        pending += 1
        remaining[key] = pending

    conditioning_cache: dict[str, tuple[bytes, str, str]] = {}
    conditioning_locks: dict[str, threading.Lock] = {}
    conditioning_guard = threading.Lock()

    def conditioning_for(key: str) -> tuple[bytes, str, str]:
        cached = conditioning_cache.get(key)
        if cached is not None:
            return cached
        with conditioning_guard:
            lock = conditioning_locks.setdefault(key, threading.Lock())
        with lock:
            cached = conditioning_cache.get(key)
            if cached is None:
                plan = plan_by_key[key]
                cached = conditioning_payload(
                    item=plan["item"],
                    key=key,
                    video_path=Path(str(plan["video_path"])),
                    jpeg_quality=int(args.jpeg_quality),
                    max_image_side=int(args.max_image_side),
                    cache_dir=cache_dir,
                )
                conditioning_cache[key] = cached
        return cached

    consensus_rows: dict[str, dict[str, Any]] = {}
    candidates_by_key: dict[str, list[dict[str, Any]]] = {}
    dense_pending: dict[str, int] = {}
    scheduled: set[str] = set()
    finalized: set[str] = set()
    inflight: set[Any] = set()
    future_meta: dict[Any, tuple[str, str]] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:

        def submit(bucket: str, key: str, fn: Callable[..., dict[str, Any]], **kwargs: Any) -> None:
            future = executor.submit(fn, **kwargs)
            future_meta[future] = (bucket, key)
            inflight.add(future)

        def schedule_global(plan: dict[str, Any]) -> None:
            key = plan["key"]
            for view in GLOBAL_VIEWS:
                for repeat in range(1, repeats + 1):
                    if (key, view, repeat) in have_global:
                        continue
                    submit(
                        "global",
                        key,
                        run_global_call,
                        args=args,
                        item=plan["item"],
                        plan=plan,
                        view=view,
                        repeat=repeat,
                        system_prompt=system_prompt,
                        conditioning_for=conditioning_for,
                        cache_dir=cache_dir,
                    )

        def submit_dense(
            key: str, candidate: dict[str, Any], bucket: str, which: str, repeat: int
        ) -> None:
            plan = plan_by_key[key]
            submit(
                bucket,
                key,
                run_dense_call,
                args=args,
                item=plan["item"],
                plan=plan,
                candidate=candidate,
                which=which,
                repeat=repeat,
                system_prompt=system_prompt,
                conditioning_for=conditioning_for,
                cache_dir=cache_dir,
            )

        def schedule_dense(key: str) -> None:
            scheduled.add(key)
            plan = plan_by_key[key]
            candidates = build_candidates(
                buckets_for(key)["global"],
                plan.get("detector"),
                key=key,
                total_frames=int(plan["total_frames"]),
                mlr_sample_count=mlr_sample_count,
                max_candidates=int(args.max_candidates_per_video),
            )
            candidates_by_key[key] = candidates
            total = 0
            for candidate in candidates:
                if not candidate.get("support_gate"):
                    continue
                for bucket, which, seen in (
                    ("verify", "B", have_verify),
                    ("refute", "C", have_refute),
                ):
                    for repeat in range(1, repeats + 1):
                        if (key, candidate.get("candidate_id"), repeat) in seen:
                            continue
                        total += 1
                        submit_dense(key, candidate, bucket, which, repeat)
            dense_pending[key] = total
            if total == 0:
                finalize(key)

        def finalize(key: str) -> None:
            if key in finalized:
                return
            finalized.add(key)
            plan = plan_by_key[key]
            buckets = buckets_for(key)
            global_records = buckets["global"]
            task_conserving = task_conserving_from_global(global_records)
            assessments: list[dict[str, Any]] = []
            for candidate in candidates_by_key.get(key) or []:
                candidate_id = candidate.get("candidate_id")
                verify_records = [
                    record
                    for record in buckets["verify"]
                    if record.get("candidate_id") == candidate_id
                ]
                refute_records = [
                    record
                    for record in buckets["refute"]
                    if record.get("candidate_id") == candidate_id
                ]
                assessments.append(
                    assess_candidate(
                        candidate,
                        verify_records,
                        refute_records,
                        task_conserving=task_conserving,
                    )
                )
            errors: list[str] = list(plan.get("errors") or [])
            for record in [*global_records, *buckets["verify"], *buckets["refute"]]:
                if record.get("error"):
                    errors.append(f"{record.get('record_id')}: {record['error']}")
                errors.extend(
                    f"{record.get('record_id')}: {issue}"
                    for issue in (record.get("schema_issues") or [])
                )
            errors = list(dict.fromkeys(errors))
            total_frames = plan.get("total_frames")
            row = video_consensus(
                item=plan["item"],
                key=key,
                total_frames=int(total_frames) if isinstance(total_frames, int) else 0,
                eligible=bool(plan["eligible"]),
                n0=plan.get("n0"),
                detector=plan.get("detector"),
                global_records=global_records,
                candidate_assessments=assessments,
                mlr_sample_count=mlr_sample_count,
                errors=errors,
            )
            row["notes"] = list(plan.get("notes") or [])
            row["from_previous_run"] = key in existing_consensus
            writers["consensus"].append(row)
            consensus_rows[key] = row

        for plan in plans:
            key = plan["key"]
            if key in existing_consensus:
                continue
            if not plan["usable"] or not plan["eligible"]:
                finalize(key)
                continue
            if remaining.get(key, 0) <= 0:
                schedule_dense(key)
            else:
                schedule_global(plan)

        while inflight:
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for future in done:
                inflight.discard(future)
                bucket, key = future_meta.pop(future)
                record = future.result()  # a bug in a worker must fail loudly, never be swallowed
                writers[bucket].append(record)
                buckets_for(key)[bucket].append(record)
                if bucket == "global":
                    remaining[key] = remaining.get(key, 0) - 1
                    plan = plan_by_key[key]
                    if remaining[key] <= 0 and key not in scheduled:
                        schedule_dense(key)
                else:
                    dense_pending[key] = dense_pending.get(key, 0) - 1
                    if dense_pending.get(key, 0) <= 0:
                        finalize(key)

    for writer in writers.values():
        writer.close()

    ordered: list[dict[str, Any]] = []
    for item in items:
        key = str(item["key"])
        if key in consensus_rows:
            ordered.append(consensus_rows[key])
        elif key in existing_consensus:
            ordered.append(existing_consensus[key])
    for key, row in existing_consensus.items():
        if key not in {str(item["key"]) for item in items}:
            ordered.append(row)
    return ordered


def write_consensus_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONSENSUS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(consensus_csv_row(row))


# --------------------------------------------------------------------------------------
# Detector record loading, dry run, summary and CLI
# --------------------------------------------------------------------------------------

# Deliberate deviations from the design note, recorded in every summary.json.
KNOWN_DEVIATIONS = (
    "Pass B/C are issued only for candidates that pass the global support gate; every other "
    "candidate is reported as insufficient_global_support instead of being dropped silently.",
    "The optional under-count local crops of the robot / target (doc section 4.2) are not "
    "implemented; occlusion support is taken from the detector trace and from the model's own "
    "occlusion_assessment.",
    "The conditioning frame label 'conditioning=1' is a documented extension of the frame "
    "label protocol in the design note.",
    "--max-image-side is an added safety parameter; 0 (the default) means frames are never "
    "resized before JPEG encoding.",
    "Detector traces are hints only: they never overwrite the frozen deterministic MLR and "
    "they never become a model input that is presented as ground truth.",
)


def load_detector_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load detector rows keyed by video key; accepts a JSON mapping, a JSON array or JSONL."""
    payload = load_json_any(path)
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        # A single record (one-line JSONL or a bare JSON object) is detected by its own fields;
        # anything else is read as a {video key: detector row} mapping.
        if _first_present(payload, ("key", "video", "video_path")) is not None:
            rows = [payload]
        else:
            for name, value in payload.items():
                if not isinstance(value, dict):
                    continue
                row = dict(value)
                own = row.get("key")
                if own not in (None, "") and str(own) != str(name):
                    # keep the row reachable under the mapping key as well as its own key
                    alias = dict(row)
                    alias["key"] = str(name)
                    rows.append(alias)
                else:
                    row.setdefault("key", name)
                rows.append(row)
    elif isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    detectors: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = _first_present(row, ("key", "video", "video_path"))
        if name is None:
            continue
        text = str(name)
        detectors[text] = row
        stem = Path(text).stem
        if stem:
            detectors.setdefault(stem, row)
    return detectors


def detector_for_item(
    item: dict[str, Any], detectors: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Detector row for one manifest item: key, then video stem, then detector_trace_path."""
    key = str(item.get("key"))
    row = detectors.get(key)
    if row is None:
        stem = Path(str(item.get("video_path") or "")).stem
        row = detectors.get(stem) if stem else None
    if row is None:
        trace_path = item.get("detector_trace_path")
        if trace_path and Path(str(trace_path)).is_file():
            payload = load_json_any(Path(str(trace_path)))
            row = payload if isinstance(payload, dict) else None
    if row is None:
        return None
    return normalize_detector_record(row, key)


def run_dry_run(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    detectors: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Plan every request a real run would issue, without decoding frames or calling the model.

    The plan is written to ``dry_run_requests.jsonl``; it is a planning aid, not a run record.
    """
    repeats = max(1, int(args.repeats))
    mlr_sample_count = int(args.mlr_sample_count)
    rows: list[dict[str, Any]] = []

    for item in items:
        key = str(item["key"])
        raw_detector = detector_for_item(item, detectors)
        detector = None
        if raw_detector is not None and raw_detector.get("available"):
            detector = detector_event_from_trace(
                raw_detector,
                mlr_sample_count=mlr_sample_count,
                total_frames=item.get("frame_count"),
            )
        try:
            plan: dict[str, Any] = prepare_video(item, args=args, detector=detector)
        except Exception as exc:
            blocking = [f"plan_failed:{type(exc).__name__}: {exc}"]
            plan = {
                "key": key,
                "video_path": item.get("video_path"),
                "total_frames": item.get("frame_count"),
                "n0": item.get("n0"),
                "eligible": False,
                "detector": detector,
                "detector_hint": None,
                "blocking_errors": blocking,
                "notes": [],
                "errors": blocking,
                "usable": False,
            }
        plan["item"] = item
        plan["detector"] = fill_detector_onset_frame(
            detector, mlr_sample_count=mlr_sample_count, total_frames=plan.get("total_frames")
        )
        plan["detector_hint"] = detector_hint_text(plan["detector"])

        base: dict[str, Any] = {
            "key": key,
            "video_path": item.get("video_path"),
            "pass": None,
            "view": None,
            "which": None,
            "candidate_id": None,
            "repeat": None,
            "model": args.model,
            "temperature": args.temperature,
            "thinking_level": args.thinking_level,
            "jpeg_quality": args.jpeg_quality,
            "max_image_side": args.max_image_side,
            "global_frame_count": args.global_frame_count,
            "mlr_sample_count": mlr_sample_count,
            "dense_window_before": args.dense_window_before,
            "dense_window_after": args.dense_window_after,
            "total_frames": plan.get("total_frames"),
            "n0": plan.get("n0"),
            "eligible": bool(plan.get("eligible")),
            "n_frames": 0,
            "n_images": 0,
            "frame_indices": [],
            "mlr_sample_ids": [],
            "frame_layout_note": None,
            "prompt_sha256": None,
            "prompt_error": None,
            "schema_version": None,
            "dense_start": None,
            "dense_end": None,
            "detector_available": bool(plan["detector"] and plan["detector"].get("available")),
            "detector_hint": plan.get("detector_hint"),
            "note": None,
        }

        if not plan["usable"]:
            row = dict(base)
            row["note"] = "video_not_usable:" + "|".join(plan.get("blocking_errors") or [])
            rows.append(row)
            continue
        if not plan["eligible"]:
            row = dict(base)
            row["note"] = "ineligible_no_calls_planned"
            rows.append(row)
            continue

        total_frames = int(plan["total_frames"])
        conditioning_label = frame_label(key, 0, conditioning=True)

        for view in GLOBAL_VIEWS:
            requests = global_view_frames(
                key=key,
                total_frames=total_frames,
                view=view,
                global_frame_count=int(args.global_frame_count),
                mlr_sample_count=mlr_sample_count,
            )
            labels = label_requests(key, requests)
            row = dict(base)
            row.update(
                {
                    "pass": "A",
                    "view": view,
                    "n_frames": len(requests),
                    "n_images": len(requests) + 1,
                    "frame_indices": [int(frame) for frame, _ in requests],
                    "mlr_sample_ids": [
                        None if sample is None else int(sample) for _, sample in requests
                    ],
                    "frame_layout_note": frame_layout_note(conditioning_label, labels),
                    "schema_version": SCHEMA_VERSION_GLOBAL,
                    "note": "pass_a",
                }
            )
            try:
                prompt = build_global_prompt(
                    instruction=prompt_text(item.get("instruction")),
                    target_query=prompt_text(item.get("target_query")),
                    n0=0 if plan.get("n0") is None else int(plan["n0"]),
                    conditioning_label=conditioning_label,
                    labels=labels,
                    detector_hint=plan.get("detector_hint"),
                )
                row["prompt_sha256"] = sha256_text(prompt)
            except Exception as exc:
                row["prompt_error"] = f"{type(exc).__name__}: {exc}"
            for repeat in range(1, repeats + 1):
                line = dict(row)
                line["repeat"] = repeat
                rows.append(line)

        # Pass B/C are planned from the detector trace only: the global votes of a dry run
        # do not exist yet, so no candidate can pass the VLM part of the support gate here.
        candidates = build_candidates(
            [],
            plan.get("detector"),
            key=key,
            total_frames=total_frames,
            mlr_sample_count=mlr_sample_count,
            max_candidates=int(args.max_candidates_per_video),
        )
        planned_dense = 0
        for candidate in candidates:
            if not candidate.get("support_gate"):
                continue
            window = candidate_window(
                candidate,
                total_frames=total_frames,
                mlr_sample_count=mlr_sample_count,
                before=int(args.dense_window_before),
                after=int(args.dense_window_after),
            )
            requests = window["frames"]
            labels = label_requests(key, requests)
            for which, builder in (("B", build_verify_prompt), ("C", build_refute_prompt)):
                row = dict(base)
                row.update(
                    {
                        "pass": which,
                        "which": which,
                        "candidate_id": candidate.get("candidate_id"),
                        "n_frames": len(requests),
                        "n_images": len(requests) + 1,
                        "frame_indices": [int(frame) for frame, _ in requests],
                        "mlr_sample_ids": [
                            None if sample is None else int(sample) for _, sample in requests
                        ],
                        "frame_layout_note": frame_layout_note(conditioning_label, labels),
                        "schema_version": SCHEMA_VERSION_VERIFY,
                        "dense_start": window["dense_start"],
                        "dense_end": window["dense_end"],
                        "note": "pass_b_c_planned_from_detector_trace_only",
                    }
                )
                try:
                    prompt = builder(
                        event_id=str(candidate.get("candidate_id") or ""),
                        instruction=prompt_text(item.get("instruction")),
                        target_query=prompt_text(item.get("target_query")),
                        n0=0 if plan.get("n0") is None else int(plan["n0"]),
                        candidate_summary=build_candidate_summary(candidate, plan.get("detector")),
                        conditioning_label=conditioning_label,
                        labels=labels,
                    )
                    row["prompt_sha256"] = sha256_text(prompt)
                except Exception as exc:
                    row["prompt_error"] = f"{type(exc).__name__}: {exc}"
                for repeat in range(1, repeats + 1):
                    line = dict(row)
                    line["repeat"] = repeat
                    rows.append(line)
                planned_dense += 1
        if planned_dense == 0:
            row = dict(base)
            row["note"] = (
                "no_pass_b_c_planned:detector_trace_unavailable_or_no_detector_candidate"
            )
            rows.append(row)

    path = output_dir / DRY_RUN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def build_summary(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    detectors: dict[str, dict[str, Any]],
    global_records: list[dict[str, Any]],
    run_config: dict[str, Any],
    resume_stats: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate run-level counts; denominators are kept explicit and never mixed."""
    manifest_keys = [str(item["key"]) for item in items]
    manifest_key_set = set(manifest_keys)
    rows_by_key = {str(row.get("key")): row for row in rows}
    manifest_rows = [rows_by_key[key] for key in manifest_keys if key in rows_by_key]
    outside_rows = [row for row in rows if str(row.get("key")) not in manifest_key_set]

    eligible_rows = [row for row in manifest_rows if int(row.get("eligible") or 0) == 1]
    detector_known_rows = [
        row for row in eligible_rows if (row.get("detector") or {}).get("available")
    ]
    detector_positive_rows = [
        row for row in detector_known_rows if (row.get("detector") or {}).get("mlr_event") == 1
    ]
    gemini_vote_rows = [
        row
        for row in eligible_rows
        if row.get("gemini_confirmed_mlr") is not None
        or row.get("verdict") == "uncertain_need_human"
    ]
    gemini_positive_rows = [
        row for row in gemini_vote_rows if row.get("gemini_confirmed_mlr") == 1
    ]
    uncertain_rows = [row for row in gemini_vote_rows if row.get("verdict") == "uncertain_need_human"]
    comparable_rows = [
        row for row in eligible_rows if row.get("detector_gemini_agree") is not None
    ]
    disagreement_rows = [
        row for row in comparable_rows if int(row.get("detector_gemini_agree") or 0) == 0
    ]

    verdict_counts = {verdict: 0 for verdict in VERDICTS}
    verdict_counts["(other)"] = 0
    for row in rows:
        verdict = str(row.get("verdict") or "")
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1
        else:
            verdict_counts["(other)"] += 1
    by_event_type: dict[str, int] = {}
    for row in rows:
        event_type = str(row.get("event_type") or "")
        if event_type:
            by_event_type[event_type] = by_event_type.get(event_type, 0) + 1

    candidate_counts = {
        "total": 0,
        "confirmed": 0,
        "rejected": 0,
        "uncertain": 0,
        "insufficient_global_support": 0,
    }
    for row in rows:
        for candidate in row.get("candidates") or []:
            candidate_counts["total"] += 1
            verdict = str(candidate.get("verdict") or "")
            if verdict in candidate_counts:
                candidate_counts[verdict] += 1

    detector_source_counts: dict[str, int] = {}
    for row in manifest_rows:
        source = str((row.get("detector") or {}).get("event_source") or "")
        if source:
            detector_source_counts[source] = detector_source_counts.get(source, 0) + 1

    error_rows = [row for row in rows if row.get("errors")]
    error_messages = [
        f"{row.get('key')}: {message}" for row in error_rows for message in (row.get("errors") or [])
    ]

    def rate(numerator: int, denominator: int, definition: str) -> dict[str, Any]:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "percent": (round(100.0 * numerator / denominator, 2) if denominator else None),
            "definition": definition,
        }

    summary: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": now_iso(),
        "primary_mlr_detector": (
            "the frozen deterministic MLR (Algorithm 1) recomputed from the detector trace; "
            "the VLM layer never overwrites it"
        ),
        "counts": {
            "manifest_videos": len(items),
            "consensus_rows": len(rows),
            "consensus_rows_outside_manifest": len(outside_rows),
            "eligible": len(eligible_rows),
            "ineligible": len(manifest_rows) - len(eligible_rows),
            "videos_with_errors": len(error_rows),
            "gemini_confirmed_mlr": len(
                [row for row in manifest_rows if row.get("gemini_confirmed_mlr") == 1]
            ),
            "detector_positive": len(detector_positive_rows),
        },
        "verdict_counts": verdict_counts,
        "by_event_type": by_event_type,
        "candidates": candidate_counts,
        "vlm_only": {
            "vlm_positive_detector_negative": verdict_counts.get("vlm_positive_detector_negative", 0),
            "detector_positive_vlm_rejected": verdict_counts.get("detector_positive_vlm_rejected", 0),
            "detector_only_unverified": verdict_counts.get("detector_only_unverified", 0),
            "detector_negative_vlm_uncertain": len(
                [
                    row
                    for row in manifest_rows
                    if row.get("verdict") == "uncertain_need_human"
                    and (row.get("detector") or {}).get("mlr_event") == 0
                ]
            ),
        },
        "detector_coverage": {
            "detector_rows_loaded": len(detectors),
            "eligible_with_detector_trace": len(detector_known_rows),
            "eligible_without_detector_trace": len(eligible_rows) - len(detector_known_rows),
            "eligible_with_detector_verdict": len(
                [
                    row
                    for row in detector_known_rows
                    if (row.get("detector") or {}).get("mlr_event") is not None
                ]
            ),
            "event_source_counts": detector_source_counts,
        },
        "auxiliary_failures": auxiliary_failure_summary(global_records),
        "rates_detail": {
            "detector_mlr_rate": rate(
                len(detector_positive_rows),
                len(detector_known_rows),
                "detector MLR events / eligible videos with a detector trace",
            ),
            "gemini_confirmed_rate": rate(
                len(gemini_positive_rows),
                len(gemini_vote_rows),
                "valid Pass B confirmed_mlr verdicts / eligible videos with a valid Gemini verdict",
            ),
            "uncertain_rate": rate(
                len(uncertain_rows),
                len(gemini_vote_rows),
                "uncertain_need_human verdicts / eligible videos with a valid Gemini verdict",
            ),
            "detector_gemini_disagreement_rate": rate(
                len(disagreement_rows),
                len(comparable_rows),
                "videos where the detector and the Gemini verdict differ / videos where both are known",
            ),
            "denominator_note": AGGREGATION_SUMMARY["denominators"],
        },
        "run": {
            "model": run_config.get("model"),
            "endpoint": run_config.get("endpoint"),
            "protocol_version": run_config.get("protocol_version"),
            "created_at": run_config.get("created_at"),
            "updated_at": run_config.get("updated_at"),
            "repeats": run_config.get("repeats"),
            "global_frame_count": run_config.get("global_frame_count"),
            "mlr_sample_count": run_config.get("mlr_sample_count"),
            "dense_window_before": run_config.get("dense_window_before"),
            "dense_window_after": run_config.get("dense_window_after"),
            "jpeg_quality": run_config.get("jpeg_quality"),
            "max_image_side": run_config.get("max_image_side"),
            "temperature": run_config.get("temperature"),
            "thinking_level": run_config.get("thinking_level"),
            "concurrency": run_config.get("concurrency"),
            "resume": run_config.get("resume"),
            "limit": run_config.get("limit"),
            "manifests": run_config.get("manifests"),
            "detector_records": run_config.get("detector_records"),
            "prompt_sha256": run_config.get("prompt_sha256"),
            "schema_version": run_config.get("schema_version"),
            "tolerances": run_config.get("tolerances"),
            "aggregation": run_config.get("aggregation"),
        },
        "errors": {
            "videos_with_errors": len(error_rows),
            "messages_total": len(error_messages),
            "messages": error_messages[:20],
        },
        "resume": dict(resume_stats),
        "notes": [
            "The VLM layer is a semantic adjudication layer, not objective ground truth.",
            "uncertain is reported as its own rate; it is never counted as a negative.",
            *KNOWN_DEVIATIONS,
        ],
    }
    return summary


def load_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    """Load and normalize manifest rows; duplicate keys are skipped with a warning."""
    items: list[dict[str, Any]] = []
    paths: list[str] = []
    seen: set[str] = set()
    for raw_path in args.manifest:
        path = Path(str(raw_path))
        paths.append(str(path))
        if not path.is_file():
            raise SystemExit(f"manifest not found: {path}")
        try:
            rows = read_manifest(path)
        except ValueError as exc:
            raise SystemExit(f"{path}: {exc}") from exc
        for row in rows:
            if not isinstance(row, dict):
                raise SystemExit(f"{path}: every manifest row must be a JSON object")
            key = row.get("key")
            video_path = row.get("video_path")
            if key in (None, "") or video_path in (None, ""):
                raise SystemExit(
                    f"{path}: every manifest row needs both 'key' and 'video_path'"
                )
            key = str(key)
            if key in seen:
                print(f"[warn] duplicate manifest key skipped: {key}", file=sys.stderr)
                continue
            seen.add(key)
            item = dict(row)
            item["key"] = key
            item["video_path"] = str(video_path)
            item["n0"] = _as_int(_first_present(row, ("n0", "initial_count", "initial_count_n0")))
            item["frame_count"] = _as_int(
                _first_present(row, ("frame_count", "total_frames", "num_frames"))
            )
            if not str(row.get("instruction") or "").strip() and not str(
                row.get("target_query") or ""
            ).strip():
                item["errors"] = ["missing_instruction_and_target_query"]
            items.append(item)
    if args.limit and int(args.limit) > 0:
        items = items[: int(args.limit)]
    return items, paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description=(
            "Gemini-assisted MLR occlusion and persistence audit (optional semantic layer on "
            "top of the frozen deterministic detector MLR)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="PATH",
        help="JSONL manifest of videos; repeat the flag to pass several manifests",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="directory for run_config.json, records, consensus and summary",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id")
    parser.add_argument(
        "--global-frame-count",
        type=int,
        default=49,
        help="frames sampled from the whole video for the dense49 Pass A view",
    )
    parser.add_argument(
        "--mlr-sample-count",
        type=int,
        default=24,
        help="number of MLR sampling timestamps (doc protocol: 24)",
    )
    parser.add_argument(
        "--dense-window-before", type=int, default=6, help="Pass B/C window before the onset"
    )
    parser.add_argument(
        "--dense-window-after", type=int, default=10, help="Pass B/C window after the onset"
    )
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality of every frame")
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=0,
        help="downscale frames so the longest side is at most this; 0 means no resizing",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    parser.add_argument(
        "--thinking-level",
        choices=("low", "medium", "high"),
        default="low",
        help="Gemini thinking level",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="repeats per global view and per Pass B/C candidate",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="maximum number of in-flight requests"
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse finished records; --no-resume clears the record files of the output dir",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="only process the first N manifest rows (0 = all)"
    )
    parser.add_argument(
        "--max-candidates-per-video",
        type=int,
        default=DEFAULT_MAX_CANDIDATES_PER_VIDEO,
        help="maximum number of candidates that get Pass B/C calls per video",
    )
    parser.add_argument(
        "--detector-records",
        default=None,
        metavar="PATH",
        help="optional detector MLR records (JSON/JSONL) used as hints and for the primary metric",
    )
    parser.add_argument("--model-retries", type=int, default=4, help="attempts per model call")
    parser.add_argument(
        "--model-timeout", type=float, default=1200.0, help="per-attempt timeout in seconds"
    )
    parser.add_argument(
        "--model-max-tokens", type=int, default=4000, help="max output tokens per model call"
    )
    parser.add_argument(
        "--include-thoughts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ask the model to return its thinking summaries as well",
    )
    parser.add_argument(
        "--frame-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cache labeled JPEG frames under frame_cache/ inside the output dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan every request offline (no API key, no frame decoding) and write dry_run_requests.jsonl",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the offline self test of the sampling, clustering, aggregation and schema rules",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if not args.manifest:
        parser.error("--manifest is required (pass at least one JSONL manifest)")
    if not args.output_dir:
        parser.error("--output-dir is required")
    if int(args.repeats) < 1:
        parser.error("--repeats must be >= 1")
    if int(args.concurrency) < 1:
        parser.error("--concurrency must be >= 1")
    if int(args.mlr_sample_count) < 2:
        parser.error("--mlr-sample-count must be >= 2")
    if int(args.global_frame_count) < 1:
        parser.error("--global-frame-count must be >= 1")
    if not args.dry_run:
        require_genai()
        require_credentials()

    items, manifest_paths = load_items(args)
    if not items:
        raise SystemExit("no manifest rows were loaded")
    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_path = Path(str(args.detector_records)) if args.detector_records else None
    if detector_path is not None and not detector_path.is_file():
        raise SystemExit(f"detector records not found: {detector_path}")
    detectors = load_detector_records(detector_path) if detector_path is not None else {}

    config_path = output_dir / RUN_CONFIG_FILE
    previous: dict[str, Any] | None = None
    if config_path.is_file():
        payload = load_json_any(config_path)
        previous = payload if isinstance(payload, dict) else None
    created_at = previous.get("created_at") if (args.resume and previous) else None
    run_config = build_run_config(
        args,
        manifest_paths,
        str(detector_path) if detector_path is not None else None,
        created_at=created_at if isinstance(created_at, str) else None,
    )

    if args.dry_run:
        rows = run_dry_run(args=args, items=items, detectors=detectors, output_dir=output_dir)
        if config_path.is_file():
            print(
                f"[dry-run] {config_path.name} already exists and was left untouched",
                file=sys.stderr,
            )
        else:
            write_json(config_path, run_config)
        notes: dict[str, int] = {}
        for row in rows:
            note = str(row.get("note") or "")
            notes[note] = notes.get(note, 0) + 1
        print(f"[dry-run] {len(rows)} planned request(s) -> {output_dir / DRY_RUN_FILE}")
        for note, count in sorted(notes.items()):
            print(f"[dry-run]   {note}: {count}")
        return 0

    record_files_present = [name for name in RECORD_FILES if (output_dir / name).is_file()]
    if args.resume:
        if record_files_present and previous is None:
            raise SystemExit(
                f"{output_dir} already contains {', '.join(record_files_present)} but no "
                f"{RUN_CONFIG_FILE}; re-run with --no-resume (or move the files) so that records "
                "of different protocols are never mixed"
            )
        if previous is not None:
            differences = protocol_differences(previous, run_config)
            if differences:
                raise SystemExit(
                    "refusing to resume: the frozen protocol fields changed ("
                    + ", ".join(differences)
                    + "); re-run with --no-resume in a fresh output dir instead"
                )
    else:
        removed = reset_output_records(output_dir)
        if removed:
            print(
                f"[resume] --no-resume: removed {len(removed)} record file(s): "
                + ", ".join(removed),
                file=sys.stderr,
            )
    write_json(config_path, run_config)

    existing = (
        load_run_records(output_dir)
        if args.resume
        else {"global": [], "verify": [], "refute": [], "consensus": []}
    )
    resume_stats = {
        "reused_global_records": len(existing["global"]),
        "reused_verify_records": len(existing["verify"]),
        "reused_refute_records": len(existing["refute"]),
        "videos_already_aggregated": len(existing["consensus"]),
    }
    print(
        f"[run] model={args.model} endpoint={run_config.get('endpoint')} "
        f"repeats={args.repeats} concurrency={args.concurrency} videos={len(items)}"
    )
    print(f"[run] output-dir={output_dir}")
    if any(resume_stats.values()):
        print(
            "[resume] reused "
            + ", ".join(f"{name}={count}" for name, count in resume_stats.items())
        )

    rows = run_protocol(
        args=args,
        items=items,
        detectors=detectors,
        output_dir=output_dir,
        existing=existing,
    )
    write_consensus_csv(output_dir / CONSENSUS_CSV_FILE, rows)
    summary = build_summary(
        args=args,
        items=items,
        rows=rows,
        detectors=detectors,
        global_records=read_records(output_dir / GLOBAL_RECORDS_FILE),
        run_config=run_config,
        resume_stats=resume_stats,
    )
    write_json(output_dir / SUMMARY_FILE, summary)

    counts = summary["counts"]
    print(
        "[done] "
        + " ".join(f"{name}={value}" for name, value in counts.items())
        + " | verdicts="
        + ", ".join(
            f"{verdict}={count}"
            for verdict, count in summary["verdict_counts"].items()
            if count
        )
    )
    print(f"[done] wrote {CONSENSUS_JSONL_FILE}, {CONSENSUS_CSV_FILE} and {SUMMARY_FILE}")
    return 0


# --------------------------------------------------------------------------------------
# Offline self test
# --------------------------------------------------------------------------------------


def run_self_test() -> int:
    """Offline checks of the frozen sampling, clustering, aggregation and schema rules."""
    checks: list[tuple[str, bool | None, str]] = []

    def check(name: str, condition: Any, detail: str) -> None:
        checks.append((name, None if condition is None else bool(condition), detail))

    points = mlr_sample_points(93, 24)
    check(
        "mlr_sample_points(93,24)",
        len(points) == 24 and points[0] == (0, 0) and points[-1] == (23, 92),
        f"len={len(points)} first={points[0] if points else None} last={points[-1] if points else None}",
    )

    uniform = uniform_frame_indices(93, 49)
    check(
        "uniform_frame_indices(93,49)",
        len(uniform) == 49
        and uniform[0] == 0
        and uniform[-1] == 92
        and all(left < right for left, right in zip(uniform, uniform[1:])),
        f"len={len(uniform)} first={uniform[0] if uniform else None} last={uniform[-1] if uniform else None}",
    )

    head = dense_window_bounds(93, 2, 6, 10)
    tail = dense_window_bounds(93, 91, 6, 10)
    check(
        "dense_window_bounds clamping",
        head == (0, 12) and tail == (85, 92),
        f"onset=2 -> {head}; onset=91 -> {tail}",
    )

    votes = [
        {"view": "mlr24", "repeat": 1, "onset_frame": 30, "event_type": "duplication", "reason": "a"},
        {"view": "dense49", "repeat": 2, "onset_frame": 33, "event_type": "duplication", "reason": "b"},
        {"view": "mlr24", "repeat": 3, "onset_frame": 40, "event_type": "duplication", "reason": "c"},
    ]
    clusters = cluster_candidates(votes, tolerance=CANDIDATE_CLUSTER_TOLERANCE_FRAMES)
    check(
        "cluster_candidates(tolerance=6)",
        len(clusters) == 2 and max(cluster["vote_count"] for cluster in clusters) == 2,
        f"clusters={[cluster['vote_count'] for cluster in clusters]}",
    )

    def make_candidate(**overrides: Any) -> dict[str, Any]:
        candidate: dict[str, Any] = {
            "candidate_id": "self_test__c00",
            "event_type": "duplication",
            "onset_frame": 30,
            "first_suspicious_frame": 28,
            "last_suspicious_frame": 40,
            "onset_sample_id": 7,
            "vote_count": 2,
            "views": ["mlr24", "dense49"],
            "repeats": [1, 2],
            "reasons": ["self test"],
            "detector_supported": False,
            "source": "vlm",
            "support_gate": True,
            "total_frames": 93,
            "mlr_sample_count": 24,
        }
        candidate.update(overrides)
        return candidate

    def verify_record(verdict: str, **fields: Any) -> dict[str, Any]:
        parsed: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION_VERIFY,
            "event_id": "self_test__c00",
            "verdict": verdict,
            "event_type": "duplication",
            "onset_frame": 30,
            "end_frame": 40,
            "persistence_original_frames": 6,
            "persistence_mlr_samples": 3,
            "first_confirmed_mlr_pair": [7, 8],
            "occlusion_assessment": "no_occlusion",
            "benign_explanation": "none",
            "evidence_frames": [30, 32, 34],
            "confidence": "high",
            "short_reason": "self test",
        }
        parsed.update(fields)
        return {
            "record_id": f"self_test__B__{verdict}",
            "candidate_id": "self_test__c00",
            "parsed": parsed,
        }

    confirmed_assessment = assess_candidate(
        make_candidate(), [verify_record("confirmed_mlr")], [], task_conserving=True
    )
    rejected_assessment = assess_candidate(
        make_candidate(),
        [verify_record("rejected")],
        [verify_record("rejected", benign_explanation="robot_occlusion")],
        task_conserving=True,
    )
    uncertain_assessment = assess_candidate(
        make_candidate(), [verify_record("uncertain_need_human")], [], task_conserving=True
    )
    check(
        "assess_candidate verdicts",
        confirmed_assessment["verdict"] == "confirmed"
        and rejected_assessment["verdict"] == "rejected"
        and uncertain_assessment["verdict"] == "uncertain",
        f"confirmed={confirmed_assessment['verdict']} rejected={rejected_assessment['verdict']} "
        f"uncertain={uncertain_assessment['verdict']}",
    )
    check(
        "assess_candidate persistence and onset",
        confirmed_assessment["persistence_mlr_samples"] == 3
        and confirmed_assessment["onset_consistent"] is True,
        f"persistence={confirmed_assessment['persistence_mlr_samples']} "
        f"onset_spread_frames={confirmed_assessment['onset_spread_frames']}",
    )

    consensus_item = {"key": "self_test", "video_path": "/tmp/self_test.mp4"}
    detector_positive = {
        "available": True,
        "n0": 1,
        "detector_mlr": 1,
        "detector_event_source": "given_mlr_event",
        "counts": [1, 2, 2],
    }
    detector_negative = {
        "available": True,
        "n0": 1,
        "detector_mlr": 0,
        "detector_event_source": "recomputed_from_counts",
        "counts": [1, 1, 1],
    }
    detector_positive_row = video_consensus(
        item=consensus_item,
        key="self_test",
        total_frames=93,
        eligible=True,
        n0=1,
        detector=detector_positive,
        global_records=[],
        candidate_assessments=[rejected_assessment],
        mlr_sample_count=24,
        errors=[],
    )
    detector_negative_row = video_consensus(
        item=consensus_item,
        key="self_test",
        total_frames=93,
        eligible=True,
        n0=1,
        detector=detector_negative,
        global_records=[],
        candidate_assessments=[confirmed_assessment],
        mlr_sample_count=24,
        errors=[],
    )
    check(
        "video_consensus verdicts",
        detector_positive_row["verdict"] == "detector_positive_vlm_rejected"
        and detector_negative_row["verdict"] == "vlm_positive_detector_negative"
        and detector_negative_row["gemini_confirmed_mlr"] == 1,
        f"detector_positive={detector_positive_row['verdict']} "
        f"detector_negative={detector_negative_row['verdict']} "
        f"gemini_confirmed_mlr={detector_negative_row['gemini_confirmed_mlr']}",
    )

    global_required = [
        "schema_version",
        "task_count_conserving",
        "target_query_used",
        "initial_count_n0",
        "initial_count_confidence",
        "has_mlr_candidate",
        "candidate_events",
        "auxiliary_failures",
        "overall_confidence",
        "short_reason",
    ]
    verify_required = [
        "schema_version",
        "event_id",
        "verdict",
        "event_type",
        "onset_frame",
        "end_frame",
        "persistence_original_frames",
        "persistence_mlr_samples",
        "first_confirmed_mlr_pair",
        "occlusion_assessment",
        "benign_explanation",
        "evidence_frames",
        "confidence",
        "short_reason",
    ]
    check(
        "frozen JSON schemas",
        list(MLR_GLOBAL_SCHEMA["required"]) == global_required
        and list(MLR_VERIFY_SCHEMA["required"]) == verify_required
        and MLR_VERIFY_SCHEMA["properties"]["verdict"]["enum"]
        == ["confirmed_mlr", "rejected", "uncertain_need_human"]
        and MLR_GLOBAL_SCHEMA["properties"]["candidate_events"]["items"]["properties"][
            "event_type"
        ]["enum"]
        == ["duplication", "disappearance", "both", "uncertain"],
        f"global_required={len(MLR_GLOBAL_SCHEMA['required'])} "
        f"verify_required={len(MLR_VERIFY_SCHEMA['required'])}",
    )
    check(
        "consensus columns and verdicts",
        list(CONSENSUS_COLUMNS)
        == [
            "key",
            "eligible",
            "n0",
            "primary_detector_mlr",
            "gemini_confirmed_mlr",
            "verdict",
            "event_type",
            "onset_frame",
            "end_frame",
            "persistence_original_frames",
            "persistence_mlr_samples",
            "first_confirmed_mlr_pair",
            "occlusion_assessment",
            "confidence",
            "detector_gemini_agree",
            "uncertainty_reason",
        ]
        and list(VERDICTS)
        == [
            "confirmed_mlr",
            "rejected",
            "uncertain_need_human",
            "detector_positive_vlm_rejected",
            "vlm_positive_detector_negative",
            "detector_only_unverified",
            "ineligible",
        ],
        f"columns={len(CONSENSUS_COLUMNS)} verdicts={len(VERDICTS)}",
    )

    labels = label_requests("self_test", points)
    conditioning_label = frame_label("self_test", 0, conditioning=True)
    try:
        prompt_ok = bool(build_system_prompt()) and bool(
            build_global_prompt(
                instruction="put the red cup into the tray",
                target_query="red cup",
                n0=1,
                conditioning_label=conditioning_label,
                labels=labels,
                detector_hint=detector_hint_text(detector_positive),
            )
        )
        prompt_ok = prompt_ok and bool(
            build_verify_prompt(
                event_id="self_test__c00",
                instruction="put the red cup into the tray",
                target_query="red cup",
                n0=1,
                candidate_summary="candidate_id: self_test__c00",
                conditioning_label=conditioning_label,
                labels=labels,
            )
        )
        prompt_ok = prompt_ok and bool(
            build_refute_prompt(
                event_id="self_test__c00",
                instruction="put the red cup into the tray",
                target_query="red cup",
                n0=1,
                candidate_summary="candidate_id: self_test__c00",
                conditioning_label=conditioning_label,
                labels=labels,
            )
        )
    except Exception as exc:
        check("prompt builders", False, f"{type(exc).__name__}: {exc}")
    else:
        check("prompt builders", prompt_ok, "system + global + verify + refute prompts built")

    layout_lines = frame_layout_note(conditioning_label, labels).splitlines()
    short_layout = frame_layout_note(conditioning_label, labels[:2])
    check(
        "frame layout note numbering",
        len(layout_lines) == 2
        and layout_lines[1].startswith(f"1: {conditioning_label};")
        and layout_lines[1].endswith(f"{len(labels) + 1}: {labels[-1]}")
        and short_layout.startswith(
            f"Frame order of this request (image order): 1: {conditioning_label}; 2: "
        ),
        f"long_entries={len(layout_lines[1].split('; '))} labels={len(labels)}",
    )

    detector_detail = ""
    detector_ok = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            single_path = Path(tmp) / "single.jsonl"
            mapping_path = Path(tmp) / "mapping.json"
            single_path.write_text(
                json.dumps({"key": "self_test", "n0": 1, "detector_counts": [1, 0, 0]}),
                encoding="utf-8",
            )
            mapping_path.write_text(
                json.dumps({"self_test": {"n0": 1, "detector_counts": [1, 0, 0]}}),
                encoding="utf-8",
            )
            single = load_detector_records(single_path).get("self_test")
            mapped = load_detector_records(mapping_path).get("self_test")
            detector_ok = bool(single) and bool(mapped)
            detector_detail = (
                f"single_row={normalize_detector_record(single, 'self_test')['n0'] if single else None} "
                f"mapping_row={normalize_detector_record(mapped, 'self_test')['n0'] if mapped else None}"
            )
    except Exception as exc:
        detector_detail = f"{type(exc).__name__}: {exc}"
    check("detector record loading (JSONL row and mapping)", detector_ok, detector_detail)

    check(
        "occlusion gate (tau=0.15)",
        under_count_occlusion_supported([0.2, 0.1]) is False
        and under_count_occlusion_supported([0.2, 0.3]) is True
        and under_count_occlusion_supported(None) is False,
        "partially covered under-count stays a violation",
    )
    adjusted, exempt = adjusted_counts_from_raw(1, [0, 2], [1, 1])
    adjusted_plain, exempt_plain = adjusted_counts_from_raw(1, [0], [0])
    check(
        "adjusted_counts_from_raw",
        adjusted == [1, 2] and exempt == [1, 0] and adjusted_plain == [0] and exempt_plain == [0],
        f"exempted={adjusted} exempt={exempt} unexempted={adjusted_plain}",
    )

    event = first_persistent_deviation(1, [0, 0, 1])
    isolated = first_persistent_deviation(1, [1, 0, 1, 1])
    check(
        "first_persistent_deviation",
        event["mlr_event"] is True
        and event["onset_sample"] == 0
        and event["first_confirmed_pair"] == [0, 1]
        and isolated["mlr_event"] is False,
        f"persistent={event} isolated={isolated['mlr_event']}",
    )

    if cv2 is None:
        check("labeled JPEG smoke test", None, "opencv (cv2) unavailable")
    else:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - numpy ships with opencv
            check("labeled JPEG smoke test", None, f"numpy unavailable: {exc}")
        else:
            try:
                frame = np.zeros((64, 160, 3), dtype=np.uint8)
                frame[:, :] = (24, 32, 48)
                data = encode_jpeg(draw_frame_label(frame, labels[0]), 85)
            except Exception as exc:
                check("labeled JPEG smoke test", False, f"{type(exc).__name__}: {exc}")
            else:
                check(
                    "labeled JPEG smoke test",
                    data[:2] == b"\xff\xd8" and len(data) > 100,
                    f"jpeg_bytes={len(data)} label={labels[0]!r}",
                )

    failures = [name for name, ok, _ in checks if ok is False]
    skips = [name for name, ok, _ in checks if ok is None]
    for name, ok, detail in checks:
        tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {name} - {detail}")
    print(
        f"{len(checks) - len(failures) - len(skips)}/{len(checks) - len(skips)} checks passed"
        + (f", {len(skips)} skipped" if skips else "")
    )
    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
