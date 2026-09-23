import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda


class NTUDataset(Dataset):
    """
    Multimodal video dataset: RGB + IR.

    Modes:
      - Full frame (default): no tracks provided
      - Person crop: provide tracks_rgb_path and/or tracks_ir_path

    Each person-track becomes an independent training sample.
    RGB and IR use their own independent YOLO detections for cropping.
    """

    def __init__(
        self,
        rgb_dir,
        ir_dir,
        slow_num_frames=8,
        fast_num_frames=32,
        side_size=256,
        tracks_rgb_path=None,
        tracks_ir_path=None,
        crop_size=224,
    ):
        self.rgb_dir = Path(rgb_dir)
        self.ir_dir = Path(ir_dir)
        self.slow_num_frames = slow_num_frames
        self.fast_num_frames = fast_num_frames
        self.side_size = side_size
        self.crop_size = crop_size

        # Match RGB-IR pairs
        self.pairs = self._match_pairs()

        # Labels
        self.label_map = self._build_label_map()

        # Tracks (optional, independent per modality)
        self.tracks_rgb = None
        self.tracks_ir  = None
        self.samples = None   # list of (pair_idx, track_id) when cropping

        if tracks_rgb_path is not None:
            with open(tracks_rgb_path) as f:
                self.tracks_rgb = json.load(f)
        if tracks_ir_path is not None:
            with open(tracks_ir_path) as f:
                self.tracks_ir = json.load(f)

        if self.tracks_rgb is not None or self.tracks_ir is not None:
            self._build_crop_samples()

    # ── helpers ──────────────────────────────────────

    def _match_pairs(self):
        ir_dict = {f.stem.replace("_ir", ""): f
                   for f in self.ir_dir.glob("*_ir.avi")}
        pairs = []
        for rgb in sorted(self.rgb_dir.glob("*_rgb.avi")):
            pre = rgb.stem.replace("_rgb", "")
            if pre in ir_dict:
                pairs.append((rgb, ir_dict[pre]))
        return pairs

    def _build_label_map(self):
        label_map = {}
        for i, (rgb_path, _) in enumerate(self.pairs):
            name = rgb_path.stem
            a = int(name.split("A")[-1].split("_")[0])
            label_map[i] = a - 1
        return label_map

    def _build_crop_samples(self):
        """One entry per person-track. Use RGB tracks as primary source,
        IR tracks as fallback for track enumeration."""
        self.samples = []
        for i, (rgb_path, _) in enumerate(self.pairs):
            stem = rgb_path.stem.replace("_rgb", "")
            # Collect track IDs from both sources
            track_ids = set()
            if self.tracks_rgb and stem in self.tracks_rgb:
                track_ids.update(self.tracks_rgb[stem]["tracks"].keys())
            if self.tracks_ir and stem in self.tracks_ir:
                track_ids.update(self.tracks_ir[stem]["tracks"].keys())
            if track_ids:
                for tid in sorted(track_ids, key=int):
                    self.samples.append((i, tid))
            else:
                # No tracks → fall back to full frame with None track
                self.samples.append((i, None))

    def _stem(self, pair_idx):
        return self.pairs[pair_idx][0].stem.replace("_rgb", "")

    @staticmethod
    def _uniform_indices(total, count):
        """Always return exactly `count` indices, repeating frames if needed."""
        return [int(round(i * (total - 1) / max(count - 1, 1)))
                for i in range(count)]

    def _get_track(self, tracks_dict, pair_idx, track_id):
        """Get track data for a specific person, or None if unavailable."""
        if tracks_dict is None:
            return None
        stem = self._stem(pair_idx)
        if stem not in tracks_dict:
            return None
        if track_id is None or track_id not in tracks_dict[stem]["tracks"]:
            return None
        return tracks_dict[stem]["tracks"][track_id]

    def _bbox_lookup(self, track):
        """Build {frame: (cx,cy,w,h)} from track list."""
        if track is None:
            return {}
        return {t[0]: t[1:5] for t in track}

    # ── transforms ───────────────────────────────────

    def _get_transform(self, num_frames, is_rgb=True):
        if is_rgb:
            mean, std = [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
        else:
            mean, std = [0.5], [0.5]
        return ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                ShortSideScale(size=self.side_size),
            ]))

    def _norm_only(self, is_rgb=True):
        if is_rgb:
            mean, std = [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
        else:
            mean, std = [0.5], [0.5]
        return ApplyTransformToKey(
            key="video",
            transform=Compose([
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
            ]))

    # ── full-frame ───────────────────────────────────

    def _get_full_item(self, idx):
        rgb_path, ir_path = self.pairs[idx]
        label = self.label_map[idx]
        try:
            rgb_video = EncodedVideo.from_path(str(rgb_path))
            ir_video  = EncodedVideo.from_path(str(ir_path))
            dur = min(int(rgb_video.duration), int(ir_video.duration))
            rgb_clip = rgb_video.get_clip(0, dur)
            ir_clip  = ir_video.get_clip(0, dur)
            del rgb_video, ir_video

            item = {
                "rgb_slow": self._get_transform(self.slow_num_frames, True)(rgb_clip)["video"],
                "rgb_fast": self._get_transform(self.fast_num_frames, True)(rgb_clip)["video"],
                "ir_slow":  self._get_transform(self.slow_num_frames, False)(ir_clip)["video"],
                "ir_fast":  self._get_transform(self.fast_num_frames, False)(ir_clip)["video"],
                "label":    torch.tensor(label, dtype=torch.long),
            }
            # If in crop mode, resize full-frame tensors to crop_size
            if self.samples is not None:
                for key in ["rgb_slow", "rgb_fast", "ir_slow", "ir_fast"]:
                    item[key] = F.interpolate(
                        item[key].unsqueeze(0),
                        size=(item[key].shape[1], self.crop_size, self.crop_size),
                        mode='trilinear', align_corners=False,
                    ).squeeze(0)
            return item
        except Exception as e:
            print(f"Error idx {idx}: {e}")
            return self._zero_sample()

    # ── crop ─────────────────────────────────────────

    def _get_cropped_item(self, pair_idx, track_id):
        rgb_path, ir_path = self.pairs[pair_idx]
        label = self.label_map[pair_idx]

        # Get independent tracks per modality
        track_rgb = self._get_track(self.tracks_rgb, pair_idx, track_id)
        track_ir  = self._get_track(self.tracks_ir,  pair_idx, track_id)

        # RGB track missing but IR track exists → reuse IR bbox for RGB
        # (covers blurry/black RGB frames where YOLO can't detect)
        if track_rgb is None and track_ir is not None:
            track_rgb = track_ir

        bbox_rgb = self._bbox_lookup(track_rgb)
        bbox_ir  = self._bbox_lookup(track_ir)

        try:
            rgb_video = EncodedVideo.from_path(str(rgb_path))
            ir_video  = EncodedVideo.from_path(str(ir_path))
            dur = min(int(rgb_video.duration), int(ir_video.duration))

            rgb_clip = rgb_video.get_clip(0, dur)["video"]
            ir_clip  = ir_video.get_clip(0, dur)["video"]
            del rgb_video, ir_video  # release video handles

            _, T_rgb, H_rgb, W_rgb = rgb_clip.shape
            _, T_ir,  H_ir,  W_ir  = ir_clip.shape

            slow_idx_rgb = self._uniform_indices(T_rgb, self.slow_num_frames)
            fast_idx_rgb = self._uniform_indices(T_rgb, self.fast_num_frames)
            slow_idx_ir  = self._uniform_indices(T_ir,  self.slow_num_frames)
            fast_idx_ir  = self._uniform_indices(T_ir,  self.fast_num_frames)

            rgb_slow = self._crop_frames(rgb_clip, slow_idx_rgb, bbox_rgb, T_rgb, H_rgb, W_rgb)
            rgb_fast = self._crop_frames(rgb_clip, fast_idx_rgb, bbox_rgb, T_rgb, H_rgb, W_rgb)
            ir_slow  = self._crop_frames(ir_clip,  slow_idx_ir,  bbox_ir,  T_ir,  H_ir,  W_ir)
            ir_fast  = self._crop_frames(ir_clip,  fast_idx_ir,  bbox_ir,  T_ir,  H_ir,  W_ir)
            del rgb_clip, ir_clip

            norm_rgb = self._norm_only(is_rgb=True)
            norm_ir  = self._norm_only(is_rgb=False)

            return {
                "rgb_slow": norm_rgb({"video": rgb_slow})["video"],
                "rgb_fast": norm_rgb({"video": rgb_fast})["video"],
                "ir_slow":  norm_ir({"video": ir_slow})["video"],
                "ir_fast":  norm_ir({"video": ir_fast})["video"],
                "label":    torch.tensor(label, dtype=torch.long),
            }
        except Exception as e:
            print(f"Error crop {pair_idx}/{track_id}: {e}")
            return self._zero_sample()

    def _crop_frames(self, clip, indices, bbox_lookup, T_total, H, W):
        """
        Crop + resize selected frames.

        clip:        [C, T_total, H, W]
        indices:     frame indices to extract
        bbox_lookup: {frame: (cx,cy,w,h)} or empty dict for full-frame fallback
        Returns:     [C, len(indices), crop_size, crop_size]
        """
        has_bbox = len(bbox_lookup) > 0
        frames = []
        for fi in indices:
            frame = clip[:, fi, :, :]          # [C, H, W]

            if has_bbox and fi in bbox_lookup:
                cx, cy, bw, bh = bbox_lookup[fi]
            elif has_bbox:
                keys = sorted(bbox_lookup.keys())
                nearest = min(keys, key=lambda k: abs(k - fi))
                cx, cy, bw, bh = bbox_lookup[nearest]
            else:
                # No bbox → use full frame
                frame = F.interpolate(
                    frame.unsqueeze(0),
                    size=(self.crop_size, self.crop_size),
                    mode='bilinear', align_corners=False,
                ).squeeze(0)
                frames.append(frame)
                continue

            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(W, int(cx + bw / 2))
            y2 = min(H, int(cy + bh / 2))

            if x2 <= x1 or y2 <= y1:
                # Degenerate → full frame
                crop = frame
            else:
                crop = frame[:, y1:y2, x1:x2]

            crop = F.interpolate(
                crop.unsqueeze(0),
                size=(self.crop_size, self.crop_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            frames.append(crop)

        return torch.stack(frames, dim=1)

    # ── fallback ─────────────────────────────────────

    def _zero_sample(self):
        return {
            "rgb_slow": torch.zeros(3, self.slow_num_frames, self.side_size, self.side_size),
            "rgb_fast": torch.zeros(3, self.fast_num_frames, self.side_size, self.side_size),
            "ir_slow":  torch.zeros(1, self.slow_num_frames, self.side_size, self.side_size),
            "ir_fast":  torch.zeros(1, self.fast_num_frames, self.side_size, self.side_size),
            "label":    torch.tensor(-1),
        }

    # ── protocol ─────────────────────────────────────

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.samples is not None:
            pair_idx, track_id = self.samples[idx]
            if track_id is None:
                return self._get_full_item(pair_idx)
            return self._get_cropped_item(pair_idx, track_id)
        return self._get_full_item(idx)
