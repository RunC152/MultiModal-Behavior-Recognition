"""
YOLO person detection + IoU tracking for NTU videos.

Detects independently on RGB and IR directories, producing two track files.
IR videos are single-channel — automatically converted to 3-channel for YOLO.

Usage:
    python scripts/detect_person.py \
        --rgb-dir datasets/NTU/original \
        --ir-dir  datasets/NTU/original_ir \
        --output-rgb tracks_rgb.json \
        --output-ir  tracks_ir.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO
from tqdm import tqdm


def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def interpolate_bbox(track_hist, total_frames):
    """Fill missing frames by linear interpolation + edge extension."""
    if not track_hist:
        return []
    known = {t[0]: t[1:] for t in track_hist}
    frames = sorted(known.keys())
    for f in range(frames[0]):
        known[f] = known[frames[0]]
    for f in range(frames[-1] + 1, total_frames):
        known[f] = known[frames[-1]]
    frames = sorted(known.keys())
    result = []
    for f in range(total_frames):
        if f in known:
            result.append([f] + list(known[f]))
        else:
            lo = max(k for k in frames if k < f)
            hi = min(k for k in frames if k > f)
            alpha = (f - lo) / (hi - lo)
            interp = [known[lo][i] + alpha * (known[hi][i] - known[lo][i])
                      for i in range(5)]
            interp[-1] = -1.0
            result.append([f] + interp)
    return result


def detect_dir(model, video_dir, output_path, device, conf, iou_thresh):
    """
    Detect person tracks in all .avi files under video_dir.
    Auto-handles 1-channel (IR) and 3-channel (RGB) videos.
    """
    video_dir = Path(video_dir)
    video_files = sorted(video_dir.glob("*.avi"))

    if not video_files:
        print(f"No .avi files found in {video_dir}")
        return {}

    all_results = {}
    for vp in tqdm(video_files, desc=f"Detecting {video_dir.name}"):
        stem = vp.stem.replace("_rgb", "").replace("_ir", "")

        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── per-frame detection ──
        frame_dets = []
        for _ in range(total):
            ok, frame = cap.read()
            if not ok:
                frame_dets.append([])
                continue

            # 1ch → 3ch for YOLO
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            results = model(frame, conf=conf, device=device,
                            classes=[0], verbose=False)
            persons = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        c = float(box.conf[0])
                        persons.append([x1, y1, x2, y2, c])
            frame_dets.append(persons)
        cap.release()

        # ── IoU tracking ──
        active = {}
        track_hist = defaultdict(list)
        next_id = 0

        for fi, dets in enumerate(frame_dets):
            if not active:
                for d in dets:
                    tid = next_id; next_id += 1
                    active[tid] = d[:4]
                    cx, cy = (d[0]+d[2])/2, (d[1]+d[3])/2
                    track_hist[tid].append([fi, cx, cy, d[2]-d[0], d[3]-d[1], d[4]])
                continue

            prev_ids = list(active.keys())
            prev_boxes = [active[tid] for tid in prev_ids]
            matched = set()

            for pi, pb in enumerate(prev_boxes):
                best_iou, best_ci = 0, -1
                for ci, cb in enumerate(dets):
                    if ci in matched:
                        continue
                    iv = iou(pb, cb[:4])
                    if iv > best_iou:
                        best_iou, best_ci = iv, ci
                if best_iou > iou_thresh and best_ci >= 0:
                    tid = prev_ids[pi]
                    x1, y1, x2, y2 = dets[best_ci][:4]
                    active[tid] = dets[best_ci][:4]
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    track_hist[tid].append([fi, cx, cy, x2-x1, y2-y1,
                                            dets[best_ci][4]])
                    matched.add(best_ci)

            for ci in range(len(dets)):
                if ci not in matched:
                    tid = next_id; next_id += 1
                    x1, y1, x2, y2 = dets[ci][:4]
                    active[tid] = dets[ci][:4]
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    track_hist[tid].append([fi, cx, cy, x2-x1, y2-y1,
                                            dets[ci][4]])

        # ── filter & interpolate ──
        min_len = max(1, total * 0.25)
        tracks = {}
        for tid, hist in track_hist.items():
            if len(hist) >= min_len:
                tracks[str(tid)] = interpolate_bbox(hist, total)

        all_results[stem] = {
            "fps": fps, "width": w, "height": h,
            "total_frames": total, "tracks": tracks,
        }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--output-rgb", default="tracks_rgb.json")
    parser.add_argument("--output-ir", default="tracks_ir.json")
    parser.add_argument("--model", default="yolo11n")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.to(args.device)

    # ── RGB ──
    print("=" * 50)
    res_rgb = detect_dir(model, args.rgb_dir, args.output_rgb,
                         args.device, args.conf, iou_thresh=0.3)
    n_tracks = sum(len(v["tracks"]) for v in res_rgb.values())
    print(f"RGB: {len(res_rgb)} videos, {n_tracks} tracks → {args.output_rgb}")

    # ── IR ──
    print("=" * 50)
    res_ir = detect_dir(model, args.ir_dir, args.output_ir,
                        args.device, args.conf, iou_thresh=0.3)
    n_tracks = sum(len(v["tracks"]) for v in res_ir.values())
    print(f"IR:  {len(res_ir)} videos, {n_tracks} tracks → {args.output_ir}")


if __name__ == "__main__":
    main()
