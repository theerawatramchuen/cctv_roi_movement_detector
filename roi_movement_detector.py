"""
ROI Movement Detector with Heatmap Overlay  v2
===============================================
Uses frame-differencing (frame[N] vs frame[N - diff_gap]) instead of
MOG2 background subtraction.  This is far more reliable on busy scenes
(factory floors, cleanrooms) where workers are always present and MOG2
would eventually absorb them into the background model.

Algorithm per frame
-------------------
1. Convert to grayscale → Gaussian blur
2. Absolute diff against a frame `--diff-gap` frames earlier
3. Threshold → morphological clean-up → contour detection
4. Keep only contours that fall inside each ROI polygon mask
5. If cumulative motion area ≥ threshold % of image area → save frame
   with JET heatmap overlay + ROI polygon overlays

Dependencies:
    pip install opencv-python numpy

Usage:
    python roi_movement_detector.py \\
        --video     input.mp4   \\
        --config    127.json    \\
        --output    output_frames/ \\
        --threshold 5.0         \\   # % of image area (default 5)
        --diff-gap  5           \\   # frames between compared pair (default 5)
        --diff-thresh 25        \\   # pixel intensity diff threshold 0-255 (default 25)
        --min-area  500         \\   # min contour px² to count (default 500)
        --skip      1           \\   # analyse every Nth frame (default 1)
        --warmup    30          \\   # skip first N frames while buffer fills (default 30)
        --cooldown  10          \\   # min frames between saves for same ROI (default 10)
        --decay     0.90            # heatmap temporal decay (default 0.90)

Reference: https://claude.ai/share/45a8411b-5640-476d-94a4-1c9649f49905
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────── helpers ────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_roi_masks(polygons: list, h: int, w: int):
    """Return list of (roi_id, pts_array, mask) tuples."""
    result = []
    for poly in polygons:
        pts  = np.array(poly["points"], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        result.append((poly["id"], pts, mask))
    return result


def draw_roi_borders(frame: np.ndarray, roi_data: list, colours: list) -> np.ndarray:
    out = frame.copy()
    for idx, (roi_id, pts, _) in enumerate(roi_data):
        c = colours[idx % len(colours)]
        cv2.polylines(out, [pts], isClosed=True, color=c, thickness=2)
        cv2.putText(out, f"ROI {roi_id}", tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return out


def overlay_heatmap(frame: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.55) -> np.ndarray:
    """Blend JET heatmap onto frame only where heatmap > 0."""
    if heatmap.max() == 0:
        return frame.copy()
    norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    jet  = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    mask = (norm > 0)[:, :, np.newaxis]
    blended = cv2.addWeighted(frame, 1 - alpha, jet, alpha, 0)
    return np.where(mask, blended, frame)


# ──────────────────────────── detector ───────────────────────────────────────

class ROIMovementDetector:

    COLOURS = [
        (0, 255, 0),    # green   – ROI 0
        (0, 165, 255),  # orange  – ROI 1
        (255, 0, 255),  # magenta – ROI 2
        (0, 255, 255),  # cyan    – ROI 3
        (255, 255, 0),  # yellow  – ROI 4
    ]

    def __init__(self, args):
        self.video_path   = args.video
        self.config       = load_config(args.config)
        self.out_dir      = Path(args.output)
        self.threshold    = args.threshold
        self.diff_gap     = args.diff_gap
        self.diff_thresh  = args.diff_thresh
        self.min_area     = args.min_area
        self.frame_skip   = args.skip
        self.warmup       = args.warmup
        self.cooldown     = args.cooldown
        self.decay        = args.decay
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── motion mask via frame differencing ───────────────────────────────────
    @staticmethod
    def _motion_mask(gray_now: np.ndarray,
                     gray_ref: np.ndarray,
                     diff_thresh: int) -> np.ndarray:
        diff = cv2.absdiff(gray_now, gray_ref)
        _, thresh = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
        return thresh

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open: {self.video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image_area = W * H

        print(f"[INFO] Video      : {self.video_path}")
        print(f"[INFO] Resolution : {W}×{H}  FPS={fps:.1f}  Frames={total}")
        print(f"[INFO] Threshold  : {self.threshold}% → "
              f"{image_area * self.threshold / 100:.0f} px²")
        print(f"[INFO] Diff gap   : {self.diff_gap} frames  |  "
              f"Pixel thresh: {self.diff_thresh}  |  "
              f"Min contour: {self.min_area} px²")

        roi_data  = build_roi_masks(self.config["polygons"], H, W)
        n_rois    = len(roi_data)

        # Rolling buffer of grayscale frames for diff_gap comparison
        buf_size  = self.diff_gap + 1
        gray_buf  = deque(maxlen=buf_size)

        # Per-ROI heatmaps + cooldown counters
        heatmaps  = {rid: np.zeros((H, W), dtype=np.float32) for rid, _, _ in roi_data}
        cooldowns = {rid: 0 for rid, _, _ in roi_data}

        trigger_log = []
        saved       = 0
        processed   = 0
        frame_no    = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            # Skip frames
            if frame_no % self.frame_skip != 0:
                continue
            processed += 1

            gray = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0
            )
            gray_buf.append(gray)

            # Tick down cooldowns
            for rid in cooldowns:
                if cooldowns[rid] > 0:
                    cooldowns[rid] -= 1

            # Need buffer full before we can diff
            if len(gray_buf) < buf_size:
                continue

            # Warm-up: let buffer fill and scene stabilise
            if processed <= self.warmup:
                if processed == self.warmup:
                    print(f"[INFO] Warm-up complete at frame {frame_no}")
                continue

            gray_ref   = gray_buf[0]          # oldest frame in buffer
            motion_all = self._motion_mask(gray, gray_ref, self.diff_thresh)

            # Decay all heatmaps
            for rid in heatmaps:
                heatmaps[rid] *= self.decay

            triggered = []

            for roi_id, pts, roi_mask in roi_data:
                # Motion pixels inside this ROI only
                roi_motion = cv2.bitwise_and(motion_all, roi_mask)

                contours, _ = cv2.findContours(
                    roi_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                motion_area = 0
                blob_mask   = np.zeros((H, W), dtype=np.uint8)
                for cnt in contours:
                    a = cv2.contourArea(cnt)
                    if a >= self.min_area:
                        motion_area += a
                        cv2.drawContours(blob_mask, [cnt], -1, 255, cv2.FILLED)

                # Accumulate heatmap
                heatmaps[roi_id] += blob_mask.astype(np.float32)

                motion_pct = motion_area / image_area * 100

                if motion_pct >= self.threshold and cooldowns[roi_id] == 0:
                    triggered.append((roi_id, motion_pct))
                    cooldowns[roi_id] = self.cooldown

            if triggered:
                # Build combined heatmap across all ROIs
                combined = np.zeros((H, W), dtype=np.float32)
                for _, hm in heatmaps.items():
                    combined = np.maximum(combined, hm)

                vis = overlay_heatmap(frame, combined, alpha=0.55)
                vis = draw_roi_borders(vis, roi_data, self.COLOURS)

                ts = frame_no / fps
                cv2.putText(vis, f"Frame {frame_no} | {ts:.2f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)

                for i, (rid, pct) in enumerate(triggered):
                    lbl = f"ROI {rid}: {pct:.1f}% > {self.threshold}% [TRIGGER]"
                    cv2.putText(vis, lbl, (10, 65 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2, cv2.LINE_AA)

                out_path = self.out_dir / f"frame_{frame_no:06d}.jpg"
                cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1

                entry = {
                    "frame": frame_no,
                    "timestamp_sec": round(ts, 3),
                    "triggered_rois": [
                        {"roi_id": r, "motion_pct": round(p, 3)}
                        for r, p in triggered
                    ],
                    "saved_to": str(out_path),
                }
                trigger_log.append(entry)
                print(f"[SAVE] {out_path.name}  –  "
                      + ", ".join(f"ROI {r}: {p:.1f}%" for r, p in triggered))

        cap.release()

        log_path = self.out_dir / "trigger_log.json"
        with open(log_path, "w") as f:
            json.dump(trigger_log, f, indent=2)

        print(f"\n[DONE] Analysed {processed} frames  |  Saved {saved} trigger frames")
        print(f"[DONE] Log → {log_path}")


# ──────────────────────────── CLI ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ROI movement detector – frame diff edition")
    p.add_argument("--video",       required=True)
    p.add_argument("--config",      required=True)
    p.add_argument("--output",      default="output_frames")
    p.add_argument("--threshold",   type=float, default=5.0,
                   help="Motion area trigger as %% of image (default 5.0)")
    p.add_argument("--diff-gap",    type=int,   default=5,
                   help="Gap in frames between compared pair (default 5)")
    p.add_argument("--diff-thresh", type=int,   default=25,
                   help="Pixel intensity diff threshold 0-255 (default 25)")
    p.add_argument("--min-area",    type=int,   default=500,
                   help="Min contour area px² (default 500)")
    p.add_argument("--skip",        type=int,   default=1,
                   help="Analyse every Nth frame (default 1)")
    p.add_argument("--warmup",      type=int,   default=30,
                   help="Skip first N processed frames (default 30)")
    p.add_argument("--cooldown",    type=int,   default=10,
                   help="Min frames between saves per ROI (default 10)")
    p.add_argument("--decay",       type=float, default=0.90,
                   help="Heatmap decay factor per frame (default 0.90)")
    return p.parse_args()


if __name__ == "__main__":
    detector = ROIMovementDetector(parse_args())
    detector.run()
