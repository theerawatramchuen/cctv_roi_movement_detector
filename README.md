# ROI Movement Detector with Heatmap Overlay

A Python computer-vision tool that analyses CCTV / IP-camera `.mp4` footage and detects object movement inside user-defined polygonal **Regions of Interest (ROI)**. When cumulative moving-object area inside any ROI exceeds a configurable percentage of the total image area, the triggering frame is saved as a JPEG with a colour heatmap overlay showing where motion occurred.

Designed for busy scenes such as **factory floors and cleanrooms** where workers are always present — environments where standard background-subtraction methods (e.g. MOG2) fail because they gradually absorb moving objects into the background model.

---

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/3ae59c54-085d-49b9-8398-246effd87e1d" />


## Table of Contents

1. [Features](#features)
2. [How It Works](#how-it-works)
3. [Requirements](#requirements)
4. [Project Structure](#project-structure)
5. [Configuration File](#configuration-file)
6. [Usage](#usage)
7. [CLI Arguments](#cli-arguments)
8. [Output](#output)
9. [Tuning Guide](#tuning-guide)
10. [Function & Class Reference](#function--class-reference)
11. [Troubleshooting](#troubleshooting)

---

## Features

- **Frame-differencing motion detection** — compares frame N against frame N − `diff_gap`, reliably catching motion regardless of how long workers have been present
- **Arbitrary polygon ROIs** — any number of polygons loaded from a JSON config; each ROI is evaluated independently
- **Per-ROI cooldown** — prevents hundreds of near-identical frames being saved during a sustained motion event
- **Accumulated JET heatmap overlay** — intensity reflects how much motion has occurred and where; decays over time so stale motion fades out
- **Warm-up period** — skips the first N frames to let the rolling buffer fill and avoid false triggers at video start
- **Structured trigger log** — every saved frame is recorded in `trigger_log.json` with frame number, timestamp, ROI ID, and motion percentage
- **Frame-skip support** — process every Nth frame to speed up analysis of long recordings

---

## How It Works

```
For each processed frame
│
├─ 1. Grayscale + Gaussian blur (noise reduction)
│
├─ 2. Absolute pixel diff vs frame from `diff_gap` frames ago
│       → produces a raw change map
│
├─ 3. Binary threshold + morphological open/close
│       → clean binary motion mask
│
├─ 4. For each ROI polygon
│   ├─ AND-mask motion pixels to keep only those inside the polygon
│   ├─ Find contours; discard any smaller than `min_area`
│   ├─ Sum surviving contour areas → motion_area
│   ├─ Accumulate blob pixels into per-ROI float32 heatmap (with decay)
│   └─ If motion_area / image_area × 100 ≥ threshold AND cooldown == 0
│           → mark ROI as triggered; reset cooldown counter
│
└─ 5. If any ROI triggered
    ├─ Merge per-ROI heatmaps (element-wise max)
    ├─ Blend JET heatmap onto frame (alpha composite, motion pixels only)
    ├─ Draw ROI polygon borders + trigger labels
    ├─ Save annotated JPEG
    └─ Append entry to trigger_log.json
```

### Why frame differencing instead of MOG2?

| Method | Behaviour on busy scenes |
|---|---|
| **MOG2** (background subtraction) | Learns the scene over time. In a cleanroom with workers always present, it eventually models workers as "background" — motion is no longer detected. Also produces a massive false trigger on the very first frames before any background is established. |
| **Frame differencing** (this tool) | Compares two frames separated by a fixed gap. A worker walking is always different between frame N and frame N−5. Motion is detected continuously regardless of scene history. |

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy

```bash
pip install opencv-python numpy
```

---

## Project Structure

```
project/
├── roi_movement_detector.py   # Main script
├── 127.json                   # ROI configuration file
├── 127.mp4                    # Input video
└── output_frames/             # Created automatically
    ├── frame_000045.jpg
    ├── frame_000312.jpg
    └── trigger_log.json
```

---

## Configuration File

ROIs are defined in a JSON file. Each polygon is an object with an `id` and a list of `[x, y]` pixel coordinates defining the polygon vertices (in image-pixel space).

```json
{
  "image": "127.png",
  "image_width": 1280,
  "image_height": 720,
  "polygons": [
    {
      "id": 0,
      "points": [
        [552, 70],
        [7,   566],
        [5,   706],
        [312, 709],
        [636, 78]
      ]
    },
    {
      "id": 1,
      "points": [
        [746,  78],
        [987,  552],
        [984,  711],
        [1272, 709],
        [1272, 285],
        [872,  65]
      ]
    }
  ]
}
```

- `image_width` / `image_height` are for reference only; the script reads actual dimensions from the video stream.
- Polygons can have any number of vertices (≥ 3).
- Up to 5 ROIs are colour-coded automatically; add more colours to `ROIMovementDetector.COLOURS` for larger sets.

---

## Usage

### Minimal

```bash
python roi_movement_detector.py --video 127.mp4 --config 127.json
```

### Full example

```bash
python roi_movement_detector.py \
    --video       127.mp4          \
    --config      127.json         \
    --output      output_frames/   \
    --threshold   5.0              \
    --diff-gap    5                \
    --diff-thresh 25               \
    --min-area    500              \
    --skip        1                \
    --warmup      30               \
    --cooldown    10               \
    --decay       0.90
```

### Conservative preset (large crowd events only)

```bash
python roi_movement_detector.py --video 127.mp4 --config 127.json \
    --threshold 7.0 --diff-gap 5 --diff-thresh 30 --min-area 800 --cooldown 20
```

### Sensitive preset (catch individual workers)

```bash
python roi_movement_detector.py --video 127.mp4 --config 127.json \
    --threshold 2.0 --diff-gap 3 --diff-thresh 20 --min-area 300 --cooldown 5
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video` | str | *(required)* | Path to the input `.mp4` video file |
| `--config` | str | *(required)* | Path to the ROI `.json` configuration file |
| `--output` | str | `output_frames` | Directory where saved frames and the log are written |
| `--threshold` | float | `5.0` | Motion area trigger level as **% of total image area**. Lower = more sensitive. |
| `--diff-gap` | int | `5` | Number of frames between the two frames being compared. Larger values catch slower motion; smaller values catch fast motion only. |
| `--diff-thresh` | int | `25` | Pixel intensity difference (0–255) required to classify a pixel as changed. Raise if camera flicker causes false positives. |
| `--min-area` | int | `500` | Minimum contour area in pixels² to count as a moving object. Filters out small noise blobs. |
| `--skip` | int | `1` | Process every Nth frame. `--skip 2` halves processing time at the cost of temporal resolution. |
| `--warmup` | int | `30` | Number of processed frames to skip at the start while the rolling buffer fills. |
| `--cooldown` | int | `10` | Minimum number of processed frames between successive saves **per ROI**. Prevents duplicate saves during sustained motion. |
| `--decay` | float | `0.90` | Multiplicative decay applied to the heatmap each frame (`0–1`). Lower = heatmap fades faster; higher = longer motion trails. |

---

## Output

### Saved JPEG frames

Each trigger frame is saved as `output_frames/frame_XXXXXX.jpg` and contains:

- **JET heatmap overlay** — blue (low activity) → green → yellow → red (high activity), blended only over pixels where motion was detected
- **ROI polygon borders** — colour-coded outlines with ROI ID labels
- **Frame number and timestamp** — top-left corner
- **Trigger labels** — one line per triggered ROI showing motion percentage and threshold

### trigger_log.json

```json
[
  {
    "frame": 45,
    "timestamp_sec": 4.5,
    "triggered_rois": [
      { "roi_id": 0, "motion_pct": 7.234 },
      { "roi_id": 1, "motion_pct": 5.801 }
    ],
    "saved_to": "output_frames\\frame_000045.jpg"
  }
]
```

---

## Tuning Guide

### Threshold (`--threshold`)

Controls how much of the **total image area** must be covered by moving objects inside an ROI before a frame is saved.

| Value | Typical scenario |
|---|---|
| `2–3 %` | Detect a single worker walking through the ROI |
| `5 %` *(default)* | Detect a small group of workers moving simultaneously |
| `10 %+` | Detect large crowd movements or equipment being moved |

For a 1280×720 image (921,600 px²), 5 % = ~46,080 px² ≈ a region roughly 215×215 pixels filled with motion.

### Diff gap (`--diff-gap`) and FPS

The effective "motion window" in seconds = `diff_gap / FPS`.

| FPS | `--diff-gap` | Motion window |
|---|---|---|
| 10 | 5 *(default)* | 0.5 s |
| 25 | 5 | 0.2 s |
| 25 | 12 | ~0.5 s |
| 30 | 15 | 0.5 s |

For slow-moving objects or low frame rates, increase `--diff-gap`. For fast action, decrease it.

### Cooldown (`--cooldown`)

At 10 FPS with `--cooldown 10`, the same ROI can trigger at most once per second. Adjust proportionally to your frame rate:

```
desired_gap_seconds × FPS / frame_skip = --cooldown value
```

Example: 3-second minimum gap at 25 FPS with `--skip 1` → `--cooldown 75`

### Heatmap decay (`--decay`)

| Value | Effect |
|---|---|
| `0.99` | Very slow fade — motion from many seconds ago still visible |
| `0.90` *(default)* | Moderate fade — recent motion dominates |
| `0.70` | Fast fade — only very recent motion shown |

---

## Function & Class Reference

### Module-level functions

#### `load_config(path: str) → dict`
Loads and returns the parsed JSON ROI configuration file.

#### `build_roi_masks(polygons: list, h: int, w: int) → list`
Converts polygon point lists into binary uint8 masks (255 inside, 0 outside).  
Returns a list of `(roi_id, pts_array, mask)` tuples.

#### `draw_roi_borders(frame, roi_data, colours) → np.ndarray`
Draws polygon outlines and ROI ID labels onto a copy of `frame`. Returns the annotated copy.

#### `overlay_heatmap(frame, heatmap, alpha=0.55) → np.ndarray`
Normalises `heatmap` (float32), applies `COLORMAP_JET`, and alpha-blends it onto `frame` only at pixels where the heatmap is non-zero. Returns the composited image.

---

### `ROIMovementDetector`

Main class encapsulating all detection logic.

#### `__init__(self, args)`
Initialises the detector from parsed CLI arguments. Creates the output directory.

| Attribute | Description |
|---|---|
| `self.video_path` | Path to input video |
| `self.config` | Parsed JSON dict |
| `self.out_dir` | `pathlib.Path` to output directory |
| `self.threshold` | Motion % trigger level |
| `self.diff_gap` | Frame comparison gap |
| `self.diff_thresh` | Pixel diff threshold |
| `self.min_area` | Minimum contour area |
| `self.frame_skip` | Process every Nth frame |
| `self.warmup` | Warm-up frame count |
| `self.cooldown` | Per-ROI save cooldown |
| `self.decay` | Heatmap temporal decay |

#### `_motion_mask(gray_now, gray_ref, diff_thresh) → np.ndarray` *(static)*
Computes a binary motion mask from two grayscale frames:
1. `cv2.absdiff` — absolute pixel difference
2. Binary threshold at `diff_thresh`
3. Morphological open (2 iterations) — removes noise
4. Morphological close (2 iterations) — fills gaps in blobs

#### `run(self)`
Main processing loop:
1. Opens the video with `cv2.VideoCapture`
2. Builds ROI masks and initialises per-ROI heatmaps and cooldown counters
3. Iterates frames; populates a rolling `deque` of grayscale frames
4. After warm-up, diffs the newest frame against the oldest in the buffer
5. For each ROI, measures motion area and checks the trigger condition
6. On trigger: composites heatmap, saves JPEG, appends to trigger log
7. On completion, writes `trigger_log.json` and prints a summary

---

## Troubleshooting

**Too many saves / constant triggering**
- Raise `--threshold` (e.g. `8.0`)
- Raise `--diff-thresh` (e.g. `35`) to ignore minor lighting flicker
- Raise `--cooldown` to enforce a longer gap between saves
- Raise `--min-area` to ignore small blobs

**No saves at all**
- Lower `--threshold` (e.g. `2.0`)
- Lower `--diff-thresh` (e.g. `15`) if movement is subtle
- Lower `--min-area` (e.g. `200`)
- Increase `--diff-gap` if objects move slowly between frames

**False triggers at the start of the video**
- Increase `--warmup` (e.g. `60`) to allow more frames before analysis begins

**Heatmap not visible / too faint**
- Decrease `--decay` (e.g. `0.80`) so accumulated motion stays brighter
- The heatmap alpha is fixed at `0.55` in `overlay_heatmap()`; edit the source to adjust

**Processing too slow**
- Use `--skip 2` or `--skip 3` to process every 2nd or 3rd frame
- Reduce input resolution upstream with ffmpeg before running the detector
