from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np

from eveworld.alternatives.physlatent.prompt_parser import parse_pick_place_prompt


COLOR_RANGES = {
    'red': [((0, 70, 40), (10, 255, 255)), ((170, 70, 40), (179, 255, 255))],
    'orange': [((8, 70, 40), (25, 255, 255))],
    'yellow': [((20, 60, 50), (38, 255, 255))],
    'green': [((38, 45, 35), (85, 255, 255))],
    'blue': [((90, 45, 35), (130, 255, 255))],
    'cyan': [((80, 45, 35), (100, 255, 255))],
    'teal': [((75, 35, 35), (100, 255, 255))],
    'purple': [((125, 40, 35), (160, 255, 255))],
    'pink': [((145, 35, 45), (175, 255, 255))],
    'brown': [((5, 45, 25), (25, 210, 180))],
    'white': [((0, 0, 160), (179, 55, 255))],
    'black': [((0, 0, 0), (179, 255, 80))],
    'gray': [((0, 0, 70), (179, 45, 210))],
    'grey': [((0, 0, 70), (179, 45, 210))],
}

KERNEL = np.ones((5, 5), np.uint8)


def load_labels(packed_data_dir: Path) -> list[dict]:
    with (packed_data_dir / 'labels' / 'data.pkl').open('rb') as handle:
        return pickle.load(handle)


def prompt_to_video_map(raw_data_dir: Path) -> dict[str, Path]:
    mapping = {}
    for txt_path in sorted(raw_data_dir.glob('*.txt')):
        prompt = txt_path.read_text(encoding='utf-8').strip()
        video_path = txt_path.with_suffix('.mp4')
        if video_path.is_file():
            mapping[prompt] = video_path
    return mapping


def read_video_rgb(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f'No frames decoded from {video_path}')
    return frames


def preprocess_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    if float(height) / src_h < float(width) / src_w:
        new_h = int(round(float(width) / src_w * src_h))
        new_w = width
    else:
        new_h = height
        new_w = int(round(float(height) / src_h * src_w))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x1 = max((new_w - width) // 2, 0)
    y1 = max((new_h - height) // 2, 0)
    return resized[y1 : y1 + height, x1 : x1 + width]


def sample_frames(frames: list[np.ndarray], num_frames: int, height: int, width: int) -> list[np.ndarray]:
    indexes = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return [preprocess_frame(frames[int(idx)], height, width) for idx in indexes]


def color_words(text: str) -> list[str]:
    words = set(re.findall(r'[a-z]+', text.lower()))
    return [word for word in COLOR_RANGES if word in words]


def color_mask(frame: np.ndarray, words: list[str]) -> np.ndarray:
    if not words:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for word in words:
        for low, high in COLOR_RANGES[word]:
            mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    return clean_mask(mask)


def motion_mask(frame: np.ndarray, first_frame: np.ndarray, prev_frame: np.ndarray | None) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray, first_gray)
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        diff = np.maximum(diff, cv2.absdiff(gray, prev_gray))
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    return clean_mask(mask)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def component_boxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, int]]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area >= min_area:
            boxes.append((int(x), int(y), int(w), int(h), int(area)))
    return boxes


def choose_box(
    boxes: list[tuple[int, int, int, int, int]],
    previous_center: tuple[float, float] | None,
) -> tuple[int, int, int, int, int] | None:
    if not boxes:
        return None
    if previous_center is None:
        return max(boxes, key=lambda item: item[4])
    px, py = previous_center
    return min(
        boxes,
        key=lambda item: ((item[0] + item[2] / 2 - px) ** 2 + (item[1] + item[3] / 2 - py) ** 2) ** 0.5
        - 0.002 * item[4],
    )


def bbox_to_norm(box: tuple[int, int, int, int, int], height: int, width: int) -> list[float]:
    x, y, w, h, _area = box
    return [
        float((x + w / 2) / width),
        float((y + h / 2) / height),
        float(w / width),
        float(h / height),
    ]


def contact_stages(valid_mask: list[float]) -> list[int]:
    stages = []
    total = max(len(valid_mask) - 1, 1)
    for idx, valid in enumerate(valid_mask):
        if valid <= 0:
            stages.append(0)
            continue
        progress = idx / total
        if progress < 0.25:
            stages.append(0)
        elif progress < 0.45:
            stages.append(1)
        elif progress < 0.85:
            stages.append(2)
        else:
            stages.append(3)
    return stages


def center_distance(a: list[float], b: list[float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def object_motion_from_bboxes(bboxes: list[list[float]], valid_mask: list[float]) -> list[float]:
    motion = []
    previous = None
    for bbox, valid in zip(bboxes, valid_mask):
        if valid <= 0:
            motion.append(0.0)
            previous = None
            continue
        center = bbox[:2]
        if previous is None:
            motion.append(0.0)
        else:
            motion.append(center_distance(center, previous))
        previous = center
    return [float(min(value, 1.0)) for value in motion]


def infer_phase_done_labels(
    bboxes: list[list[float]],
    valid_mask: list[float],
    min_done_bin: int = 4,
) -> dict:
    """Infer weak phase/done labels from an 8-bin object track.

    These labels are intentionally conservative: low-quality or ambiguous
    tracks still keep state/goal/trajectory supervision, but terminal masks are
    down-weighted by ``phys_label_quality``.
    """
    num_bins = len(valid_mask)
    valid_indexes = [idx for idx, valid in enumerate(valid_mask) if valid > 0]
    object_motion = object_motion_from_bboxes(bboxes, valid_mask)
    phase = [0 for _ in range(num_bins)]
    goal_reached = [0.0 for _ in range(num_bins)]
    release = [0.0 for _ in range(num_bins)]
    terminal_mask = [0.0 for _ in range(num_bins)]
    done_bin = -1

    if not valid_indexes:
        return {
            'phys_phase': phase,
            'phys_done_bin': done_bin,
            'phys_done_mask': 0.0,
            'phys_terminal_mask': terminal_mask,
            'phys_goal_reached': goal_reached,
            'phys_release': release,
            'phys_object_motion': object_motion,
            'phys_label_quality': 0.0,
            'phys_phase_mask': [0.0 for _ in range(num_bins)],
            'phys_goal_reached_mask': [0.0 for _ in range(num_bins)],
            'phys_release_mask': [0.0 for _ in range(num_bins)],
            'phys_object_motion_mask': [0.0 for _ in range(num_bins)],
        }

    first_bbox = bboxes[valid_indexes[0]]
    goal_bbox = bboxes[valid_indexes[-1]]
    total_disp = center_distance(first_bbox, goal_bbox)
    goal_size = max(goal_bbox[2], goal_bbox[3], 0.02)
    goal_threshold = max(0.07, 1.5 * goal_size)

    for idx in valid_indexes:
        if idx < min_done_bin:
            continue
        tail = [j for j in valid_indexes if j >= idx]
        if not tail:
            continue
        tail_distances = [center_distance(bboxes[j], goal_bbox) for j in tail]
        tail_motion = [object_motion[j] for j in tail[1:]]
        tail_motion_mean = float(np.mean(tail_motion)) if tail_motion else 0.0
        tail_motion_max = float(np.max(tail_motion)) if tail_motion else 0.0
        if max(tail_distances) <= goal_threshold * 1.35 and tail_motion_mean <= 0.07 and tail_motion_max <= 0.12:
            done_bin = idx
            break

    # If the object only stabilizes at the final bin, keep a done label but no
    # terminal span. It is useful for done CE, but not for terminal denoise loss.
    if done_bin < 0 and valid_indexes:
        done_bin = valid_indexes[-1]

    if done_bin >= 0:
        for idx in range(num_bins):
            if valid_mask[idx] <= 0:
                phase[idx] = 0
                continue
            if idx < done_bin:
                progress = idx / max(done_bin, 1)
                if progress < 0.20:
                    phase[idx] = 1
                elif progress < 0.45:
                    phase[idx] = 2
                elif progress < 0.75:
                    phase[idx] = 3
                else:
                    phase[idx] = 4
            elif idx == done_bin:
                phase[idx] = 5
                goal_reached[idx] = 1.0
            elif idx == done_bin + 1:
                phase[idx] = 6
                goal_reached[idx] = 1.0
                release[idx] = 1.0
                terminal_mask[idx] = 1.0
            else:
                phase[idx] = 7
                goal_reached[idx] = 1.0
                release[idx] = 1.0
                terminal_mask[idx] = 1.0

    valid_fraction = float(sum(valid_mask) / max(num_bins, 1))
    final_area = float(goal_bbox[2] * goal_bbox[3])
    quality = valid_fraction
    if total_disp < 0.04:
        quality *= 0.65
    if final_area < 0.002:
        quality *= 0.75
    if sum(terminal_mask) <= 0:
        quality *= 0.65
    quality = float(max(0.0, min(1.0, quality)))
    seq_mask = [float(valid) * quality for valid in valid_mask]
    terminal_mask = [float(value) * quality for value in terminal_mask]

    return {
        'phys_phase': phase,
        'phys_done_bin': int(done_bin),
        'phys_done_mask': float(quality > 0.0 and done_bin >= 0),
        'phys_terminal_mask': terminal_mask,
        'phys_goal_reached': goal_reached,
        'phys_release': release,
        'phys_object_motion': object_motion,
        'phys_label_quality': quality,
        'phys_phase_mask': seq_mask,
        'phys_goal_reached_mask': seq_mask,
        'phys_release_mask': seq_mask,
        'phys_object_motion_mask': seq_mask,
    }


def label_one_video(
    video_path: Path,
    prompt: str,
    num_frames: int,
    height: int,
    width: int,
    num_traj_points: int,
    min_area: int,
    min_done_bin: int,
) -> dict:
    frames = sample_frames(read_video_rgb(video_path), num_frames, height, width)
    parsed = parse_pick_place_prompt(prompt)
    object_text = parsed.object_name or prompt
    words = color_words(object_text)
    key_indexes = np.linspace(0, num_frames - 1, num_traj_points, dtype=int)

    boxes = []
    previous_center = None
    for key_pos, frame_idx in enumerate(key_indexes):
        frame = frames[int(frame_idx)]
        color = color_mask(frame, words)
        motion = motion_mask(frame, frames[0], frames[int(key_indexes[key_pos - 1])] if key_pos > 0 else None)
        if words and int((color > 0).sum()) >= min_area:
            dilated_motion = cv2.dilate(motion, KERNEL, iterations=2)
            candidate = color & dilated_motion
            if int((candidate > 0).sum()) < min_area:
                candidate = color
        else:
            candidate = motion
        box = choose_box(component_boxes(candidate, min_area), previous_center)
        if box is None:
            boxes.append(None)
            continue
        boxes.append(box)
        previous_center = (box[0] + box[2] / 2, box[1] + box[3] / 2)

    trajectory = []
    valid_mask = []
    bboxes = []
    for box in boxes:
        if box is None:
            trajectory.append([0.0, 0.0])
            bboxes.append([0.0, 0.0, 0.0, 0.0])
            valid_mask.append(0.0)
            continue
        bbox = bbox_to_norm(box, height, width)
        bboxes.append(bbox)
        trajectory.append(bbox[:2])
        valid_mask.append(1.0)

    first_valid = next((bbox for bbox, valid in zip(bboxes, valid_mask) if valid > 0), [0.0, 0.0, 0.0, 0.0])
    last_valid = next((bbox for bbox, valid in zip(reversed(bboxes), reversed(valid_mask)) if valid > 0), [0.0, 0.0, 0.0, 0.0])
    valid_fraction = float(sum(valid_mask) / max(len(valid_mask), 1))

    record = {
        'prompt': prompt,
        'video_path': str(video_path),
        'object_name': parsed.object_name,
        'source': parsed.source,
        'target': parsed.target,
        'color_words': words,
        'phys_state': first_valid,
        'phys_goal': last_valid,
        'phys_trajectory': trajectory,
        'phys_contact': contact_stages(valid_mask),
        'phys_state_mask': 1.0 if any(valid_mask) else 0.0,
        'phys_goal_mask': 1.0 if any(valid_mask) else 0.0,
        'phys_trajectory_mask': valid_mask,
        'phys_contact_mask': valid_mask,
        'phys_valid_mask': valid_mask,
        'valid_fraction': valid_fraction,
    }
    record.update(infer_phase_done_labels(bboxes, valid_mask, min_done_bin=min_done_bin))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate weak GR1 PhysLatent pseudo labels.')
    parser.add_argument('--packed-data-dir', type=Path, default=Path('/data/datasets/gagi/gr1_finetune_data/packed_data'))
    parser.add_argument('--raw-data-dir', type=Path, default=Path('/data/datasets/gagi/gr1_finetune_data/raw_data'))
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('/data/datasets/gagi/gr1_finetune_data/physlatent_pseudo_labels/gr1_physlabels_v3_phase_done.json'),
    )
    parser.add_argument('--num-frames', type=int, default=93)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--num-traj-points', type=int, default=8)
    parser.add_argument('--min-area', type=int, default=180)
    parser.add_argument('--min-done-bin', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    labels = load_labels(args.packed_data_dir)
    prompt_map = prompt_to_video_map(args.raw_data_dir)
    records = {}
    failures = []
    selected = labels[: args.limit] if args.limit > 0 else labels
    for item in selected:
        data_index = int(item['data_index'])
        prompt = item['prompt']
        video_path = prompt_map.get(prompt)
        if video_path is None:
            failures.append({'data_index': data_index, 'prompt': prompt, 'error': 'video_not_found'})
            continue
        try:
            records[str(data_index)] = label_one_video(
                video_path=video_path,
                prompt=prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_traj_points=args.num_traj_points,
                min_area=args.min_area,
                min_done_bin=args.min_done_bin,
            )
        except Exception as exc:
            failures.append({'data_index': data_index, 'prompt': prompt, 'error': repr(exc)})

    valid_fractions = [record['valid_fraction'] for record in records.values()]
    qualities = [record.get('phys_label_quality', 0.0) for record in records.values()]
    done_bins = [record.get('phys_done_bin', -1) for record in records.values()]
    done_hist = {str(idx): int(sum(done == idx for done in done_bins)) for idx in range(args.num_traj_points)}
    done_hist['unknown'] = int(sum(done < 0 for done in done_bins))
    payload = {
        'version': 3,
        'description': 'Weak image-space PhysLatent labels with phase/done/settle supervision generated from prompt color cues and motion foreground.',
        'num_frames': args.num_frames,
        'height': args.height,
        'width': args.width,
        'num_traj_points': args.num_traj_points,
        'min_done_bin': args.min_done_bin,
        'records': records,
        'summary': {
            'requested': len(selected),
            'labeled': len(records),
            'failed': len(failures),
            'mean_valid_fraction': float(np.mean(valid_fractions)) if valid_fractions else 0.0,
            'min_valid_fraction': float(np.min(valid_fractions)) if valid_fractions else 0.0,
            'mean_label_quality': float(np.mean(qualities)) if qualities else 0.0,
            'min_label_quality': float(np.min(qualities)) if qualities else 0.0,
            'done_bin_histogram': done_hist,
            'terminal_labeled': int(sum(sum(record.get('phys_terminal_mask', [])) > 0 for record in records.values())),
        },
        'failures': failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload['summary'], indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
