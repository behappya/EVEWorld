#!/usr/bin/env python3
"""EVE · TEA VLM 层: 用 Qwen3.6-VL 直接评测"过程偷懒"(Model Laziness)。

与已有 Qwen-IF/PA-I(只问"任务完成没/物理对不对", 抓不到过程)不同, 本评测器专门
针对偷懒的【过程忠实】维度设计 prompt: 逼 VLM 逐环节检查"接触->抓取->搬运->释放->
稳定"这条因果链是否被跳过/颠倒/伪造, 而非只看终态是否正确。

普适性设计(方案27 §一的偷懒定义, 不绑定单一任务):
  - 不假设具体物体/任务, 只问通用的操作因果链环节。
  - 每个子问题是"过程是否出现了物理上不可能的捷径", 对应一种偷懒签名:
      Q1 无因移动/瞬移  Q2 未抓取即移动  Q3 缺失抓取/接触阶段
      Q4 终态提前出现    Q5 回退/拿回/数量不守恒
  - 让模型对每个子问题给 0/1 + 一句证据, 最后给一个 0-4 的 laziness 严重度。
  - 强制看整段时序(多帧), 明确提示"逐帧按时间顺序推理, 关注帧间变化而非单帧质量"。

输出: 逐视频 JSON(各子问题 + severity + 证据) + CSV + 汇总(便于 3.8s vs 7.8s 对比)。

复用 eval_dreamgenbench_qwen_api.py 的并发/读帧/OpenAI接口框架。
新增: --crop(side-by-side 取右半生成) + 偷懒专用 prompt + 结构化解析。

用法:
  python3 qwen_laziness.py \
    --video-dir <side_by_side_dir> --crop 0.5,1.0 \
    --qwen-base 127.0.0.1 --concurrency 128 \
    --out-root /data/.../eve_outputs/tea_qwen --run-name sft_3p8s --limit 92
"""
from __future__ import annotations
import argparse, base64, csv, json, os, re, threading, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2

_TLS = threading.local()

# ------------------------------------------------------------------ prompt
# 5 个子问题, 每个对应一种偷懒签名。要求 VLM 输出严格 JSON。
LAZINESS_PROMPT = """You are a strict robotics video auditor. The video shows a robot arm performing a manipulation task described as: "{prompt}".

Your job is NOT to judge visual quality or whether the final state looks correct. Your ONLY job is to detect PROCESS SHORTCUTS ("model laziness"): cases where the video reaches a plausible-looking outcome by SKIPPING, FAKING, or REVERSING the physical process that should have produced it.

Look at the frames in TIME ORDER and reason about FRAME-TO-FRAME CHANGES (how objects move between frames), not about how sharp any single frame is. A physically faithful manipulation must go through: approach -> contact -> grasp -> transport (object moves TOGETHER with the hand/gripper) -> release -> settle.

Answer these five yes/no questions. Answer 1 only if you have clear visual evidence in the frames; when unsure, answer 0.

Q1 no_cause_motion: In some frame, does the target object change position while the hand/gripper is NOT touching it (a gap is visible between hand and object, yet the object still moves, slides, jumps, or appears elsewhere)? Answer 1 if you see object displacement with a visible hand-object gap in any transition.
Q2 move_before_grasp: Look at the moment the object first starts moving toward the goal. At that moment, is the gripper clearly NOT yet closed around the object (fingers still open, or hand only near/above it, not clamping)? Answer 1 if the object begins its goal-directed motion while the gripper is not visibly closed on it. This is common — do not default to 0; check the grasp state at motion onset.
Q3 missing_stage: Is any REQUIRED stage clearly missing (e.g. the object arrives at the goal but we never see it being picked up / grasped / carried)?
Q4 premature_terminal: Independently of Q3, judge TIMING: locate the first frame where the object is at (or almost at) its goal position, and express it as a fraction of the video length. Is that fraction < 0.5 (goal reached in the first half), with later frames showing no further meaningful task progress (arm hovering, repeating, or idle)? Answer 1 if the goal appears in the first half. This can be 1 even when Q3 is also 1 (a skipped process often makes the terminal appear early). Report the timing in the evidence.
Q5 rollback_or_nonconservation: After the goal is reached, is it broken/undone (object taken back, pushed away), OR does the object count/identity change (object disappears, duplicates, morphs)?

Then give an overall laziness_severity from 0 to 4:
0 = faithful full process, no shortcut.
1 = mostly faithful, one minor questionable transition.
2 = one clear shortcut.
3 = multiple shortcuts / a major skip.
4 = terminal essentially fabricated, almost no real process.

Reply with ONLY a compact JSON object, no extra text:
{{"q1":0/1,"q2":0/1,"q3":0/1,"q4":0/1,"q5":0/1,"laziness_severity":0-4,"evidence":"one short sentence citing the frames"}}"""

# ------------------------------------------------------------------ 裁判B(独立)
# 独立裁判用于打破 best-of-N 的循环: 若用裁判A(上面)选优、又用裁判A汇报改善, 取N个带噪
# 测量的最小值天然低于均值(赢家诅咒)。裁判B测同一"语义偷懒"构念, 但用【不同的问题分解】
# (阶段完整度而非5签名), 是结构性独立的第二测量。保持 temperature=0 -> 可复现;
# 独立性来自不同措辞+不同抽帧(--frame-offset), 非采样噪声。
LAZINESS_PROMPT_B = """You are auditing a robot manipulation video for the task: "{prompt}".
Ignore image sharpness or visual quality. Judge ONLY whether the physical PROCESS that produces the outcome is actually shown, step by step, in time order.

A faithful manipulation shows a continuous causal chain: the arm approaches the object, the gripper closes on it, the object travels TOGETHER with the gripper to the goal, then is released and settles. Laziness = the video reaches the goal state by SKIPPING, FAKING, TELEPORTING, or REORDERING parts of this chain, even if the final frame looks correct.

Watch the frames in TIME ORDER (reason about how things move between consecutive frames) and answer:
1. process_completeness: a float 0.0-1.0 = the fraction of the approach->grasp->transport->release chain that is actually visible as continuous motion (1.0 = the entire chain is shown; 0.5 = about half is shown; 0.0 = the goal state simply appears with no visible process).
2. teleport_or_jump: 0/1 = does the target object ever change position WITHOUT the closed gripper carrying it (it jumps, slides on its own, or appears elsewhere across consecutive frames)?
3. goal_first_half: 0/1 = is the object already at its final goal position within the FIRST HALF of the video, with the remaining frames idle, hovering, or repetitive?

Then give an overall laziness_severity from 0 to 4, consistent with the above:
0 = the full faithful process is shown; 1 = mostly shown, one minor gap; 2 = one clear part skipped/faked; 3 = multiple parts skipped or a major skip; 4 = terminal essentially fabricated, almost no real process.

Reply with ONLY a compact JSON object, no extra text:
{{"process_completeness":0.0-1.0,"teleport_or_jump":0/1,"goal_first_half":0/1,"laziness_severity":0-4,"evidence":"one short sentence citing the frames"}}"""

PROMPTS = {"a": LAZINESS_PROMPT, "b": LAZINESS_PROMPT_B}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_qwen_base(base: str) -> str:
    value = str(base or "").strip()
    if not value:
        raise ValueError("empty qwen base")
    if value.startswith("http://") or value.startswith("https://"):
        out = value.rstrip("/")
        parsed = urlparse(out)
        if not parsed.path or parsed.path == "/":
            out = out + "/v1"
        return out
    host_port = value if ":" in value else f"{value}:8000"
    return f"http://{host_port}/v1"


def ensure_no_proxy(base_url: str) -> None:
    host = urlparse(base_url).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        if host not in parts:
            parts.append(host)
        os.environ[key] = ",".join(parts)


def get_client(base_url: str):
    from openai import OpenAI
    if not hasattr(_TLS, "clients"):
        _TLS.clients = {}
    if base_url not in _TLS.clients:
        _TLS.clients[base_url] = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    return _TLS.clients[base_url]


def resolve_model(base_url: str, requested: str, timeout: float) -> str:
    if requested and requested != "auto":
        return requested
    try:
        client = get_client(base_url)
        models = client.models.list(timeout=min(timeout, 20.0))
        for item in getattr(models, "data", []) or []:
            mid = getattr(item, "id", None)
            if mid:
                return str(mid)
    except Exception as exc:
        print(f"auto-detect model failed, fallback: {exc}", flush=True)
    return "Qwen/Qwen3.6-VL"


def prompt_from_video_path(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    return stem.replace("_", " ")


def frame_indices(total: int, k: int, offset: float = 0.0) -> list[int]:
    """均匀取 k 帧; offset(0-1 之间, 以帧间距为单位)让裁判B抽到不同帧, 增强测量独立性。"""
    if k <= 0:
        raise ValueError("frame_count must be positive")
    if total <= 0:
        return list(range(k))
    if k == 1:
        return [total // 2]
    step = (total - 1) / (k - 1)
    idxs = {min(total - 1, max(0, round(i * step + offset * step))) for i in range(k)}
    return sorted(idxs)


def video_frames(video_path: Path, frame_count: int, crop, jpeg_q: int, max_side: int,
                 offset: float = 0.0) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        urls = []
        for idx in frame_indices(total, frame_count, offset):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if crop is not None:
                w0 = frame.shape[1]
                frame = frame[:, int(crop[0] * w0):int(crop[1] * w0)]
            if max_side > 0:
                h, w = frame.shape[:2]
                if max(h, w) > max_side:
                    s = max_side / float(max(h, w))
                    frame = cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
            if not ok:
                continue
            urls.append("data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii"))
        if not urls:
            raise ValueError(f"no frames from {video_path}")
        return urls
    finally:
        cap.release()


def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for it in content:
            t = it.get("text") if isinstance(it, dict) else getattr(it, "text", None)
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def parse_laziness(text: str, judge: str = "a") -> dict[str, Any]:
    """从模型输出抽 JSON; 容错: 找第一个 {...}。裁判A/B 字段不同, 统一映射到
    laziness_severity(0-4) + any_shortcut(0/1), 使下游选优脚本无需区分裁判。"""
    raw = text or ""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    obj = {}
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = {}
    def gi(k):
        try:
            return int(round(float(obj.get(k, 0))))
        except Exception:
            return 0
    def gf(k):
        try:
            return float(obj.get(k, 0))
        except Exception:
            return 0.0
    sev = max(0, min(4, gi("laziness_severity")))
    if judge == "b":
        # 裁判B: 3 个独立签名(不同分解), 统一到 q1..q5 槽位便于同表存储
        pc = max(0.0, min(1.0, gf("process_completeness")))
        tj = 1 if gi("teleport_or_jump") >= 1 else 0
        gh = 1 if gi("goal_first_half") >= 1 else 0
        # 兜底: 未给 severity 时用签名近似(过程越不完整 severity 越高)
        if "laziness_severity" not in obj:
            sev = max(0, min(4, round((1.0 - pc) * 4)))
        q = {"q1": tj, "q2": 0, "q3": 1 if pc < 0.5 else 0, "q4": gh, "q5": 0}
        any_sc = int(tj or gh or pc < 0.5 or sev >= 2)
        return {**q, "process_completeness": round(pc, 3),
                "laziness_severity": sev, "any_shortcut": any_sc,
                "evidence": str(obj.get("evidence", ""))[:300], "parsed_ok": int(bool(m))}
    # 裁判A: 5 签名
    q = {f"q{i}": (1 if gi(f"q{i}") >= 1 else 0) for i in range(1, 6)}
    if "laziness_severity" not in obj and any(q.values()):
        sev = min(4, sum(q.values()))
    return {**q, "process_completeness": "",
            "laziness_severity": sev,
            "any_shortcut": int(any(q.values()) or sev >= 2),
            "evidence": str(obj.get("evidence", ""))[:300],
            "parsed_ok": int(bool(m))}


def chat(base_url, model, messages, max_tokens, timeout, disable_thinking):
    client = get_client(base_url)
    kwargs = {"model": model, "messages": messages, "temperature": 0.0,
              "max_tokens": max_tokens, "timeout": timeout}
    if disable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        return client.chat.completions.create(**kwargs)
    except Exception:
        if "extra_body" in kwargs:
            kwargs.pop("extra_body")
            return client.chat.completions.create(**kwargs)
        raise


def evaluate_one(video_path, frame_urls, base_url, model, args) -> dict[str, Any]:
    started = time.time()
    prompt = prompt_from_video_path(video_path)
    text = PROMPTS[args.judge].format(prompt=prompt)
    content = [{"type": "text", "text": text}]
    content.extend({"type": "image_url", "image_url": {"url": u}} for u in frame_urls)
    messages = [{"role": "user", "content": content}]
    last_err = ""
    for retry in range(1, args.model_retries + 1):
        try:
            resp = chat(base_url, model, messages, args.model_max_tokens, args.model_timeout, args.disable_thinking)
            raw = extract_text(resp.choices[0].message.content)
            parsed = parse_laziness(raw, args.judge)
            return {"video_path": str(video_path), "prompt": prompt, **parsed,
                    "raw_text": raw[:600], "error": None, "retry": retry,
                    "latency_sec": round(time.time() - started, 3), "created_at": now_iso()}
        except Exception as exc:
            last_err = str(exc)
            if retry < args.model_retries:
                time.sleep(min(2.0 * retry, 8.0))
    return {"video_path": str(video_path), "prompt": prompt,
            "q1": 0, "q2": 0, "q3": 0, "q4": 0, "q5": 0, "process_completeness": "",
            "laziness_severity": 0,
            "any_shortcut": 0, "evidence": "", "parsed_ok": 0,
            "raw_text": "", "error": f"failed after {args.model_retries}: {last_err}",
            "retry": args.model_retries, "latency_sec": round(time.time() - started, 3),
            "created_at": now_iso()}


FIELDS = ["video_path", "prompt", "q1", "q2", "q3", "q4", "q5", "process_completeness",
          "laziness_severity", "any_shortcut", "evidence", "parsed_ok", "raw_text", "error",
          "retry", "latency_sec", "created_at"]


def read_done(path: Path, rerun_errors: bool) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if rerun_errors and row.get("error"):
                continue
            if row.get("video_path"):
                done.add(row["video_path"])
    return done


def append_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    # 防列错位: 若已存在文件的表头与当前 FIELDS 不一致(如新增字段后 append),
    # 先按新 FIELDS 迁移旧行(缺列补空), 再追加。否则 DictReader 按旧表头解析新行会整体错位。
    if exists:
        with path.open("r", encoding="utf-8", newline="") as fh:
            old_header = next(csv.reader(fh), [])
        if old_header != FIELDS:
            with path.open("r", encoding="utf-8", newline="") as fh:
                old_rows = list(csv.DictReader(fh))
            with path.open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                for r in old_rows:
                    w.writerow({k: r.get(k, "") for k in FIELDS})
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in FIELDS})


def summarize(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    def col(k):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(k, 0)))
            except Exception:
                vals.append(0.0)
        return vals
    n = max(1, len(rows))
    sev = col("laziness_severity")
    out = {"n": len(rows),
           "mean_severity": sum(sev) / n,
           "any_shortcut_rate": sum(col("any_shortcut")) / n,
           "errors": sum(1 for r in rows if r.get("error"))}
    for i in range(1, 6):
        out[f"q{i}_rate"] = sum(col(f"q{i}")) / n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path,
                    default=Path("/data/datasets/gagi/eve_outputs/tea_qwen"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--qwen-base", default="127.0.0.1")
    ap.add_argument("--qwen-model", default="auto")
    ap.add_argument("--crop", default=None, help="side-by-side 取右半: 0.5,1.0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--frame-count", type=int, default=16)
    ap.add_argument("--judge", choices=["a", "b"], default="a",
                    help="a=5签名(选优用); b=独立裁判(过程完整度分解, 评测/去循环用)")
    ap.add_argument("--frame-offset", type=float, default=0.0,
                    help="抽帧相位偏移(0-1, 裁判B建议0.5), 增强与裁判A的测量独立性")
    ap.add_argument("--max-image-side", type=int, default=512)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--model-retries", type=int, default=3)
    ap.add_argument("--model-timeout", type=float, default=300.0)
    ap.add_argument("--model-max-tokens", type=int, default=1024)
    ap.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rerun-errors", action="store_true")
    args = ap.parse_args()

    base_url = normalize_qwen_base(args.qwen_base)
    ensure_no_proxy(base_url)
    model = resolve_model(base_url, args.qwen_model, args.model_timeout)
    crop = tuple(float(x) for x in args.crop.split(",")) if args.crop else None
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(args.video_dir.expanduser().resolve().glob("**/*.mp4"))
    if args.limit > 0:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"no mp4 in {args.video_dir}")

    csv_path = out_root / f"{args.run_name}_laziness.csv"
    done = read_done(csv_path, args.rerun_errors) if args.resume else set()
    pending = [v for v in videos if str(v) not in done]

    print(f"[qwen-laziness] dir={args.video_dir}", flush=True)
    print(f"  model={model} base={base_url} crop={crop} frames={args.frame_count}", flush=True)
    print(f"  videos={len(videos)} pending={len(pending)} concurrency={args.concurrency}", flush=True)

    frame_cache: dict[Path, list[str]] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        it = iter(pending)

        def submit_next() -> bool:
            try:
                path = next(it)
            except StopIteration:
                return False
            try:
                if path not in frame_cache:
                    frame_cache[path] = video_frames(path, args.frame_count, crop, args.jpeg_quality,
                                                     args.max_image_side, args.frame_offset)
            except Exception as exc:
                print(f"  frame extract failed {path.name}: {exc}", flush=True)
                return submit_next()
            fut = pool.submit(evaluate_one, path, frame_cache[path], base_url, model, args)
            futures[fut] = path
            return True

        for _ in range(args.concurrency):
            if not submit_next():
                break
        completed = len(done)
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            records = []
            for fut in finished:
                path = futures.pop(fut)
                try:
                    records.append(fut.result())
                except Exception as exc:
                    records.append({"video_path": str(path), "prompt": prompt_from_video_path(path),
                                    "error": str(exc), "laziness_severity": 0, "any_shortcut": 0,
                                    "parsed_ok": 0, "created_at": now_iso()})
                completed += 1
                if completed % 10 == 0 or completed == len(videos):
                    print(f"  completed {completed}/{len(videos)}", flush=True)
                submit_next()
            append_csv(csv_path, records)

    summary = summarize(csv_path)
    summary["run_name"] = args.run_name
    summary["video_dir"] = str(args.video_dir)
    (out_root / f"{args.run_name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
