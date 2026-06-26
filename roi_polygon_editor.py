"""
ROI Polygon Editor
==================
Draw one or more ROI polygons on any image and save to JSON with the same name.

Controls:
  Left-click          — add point to current polygon
  Right-click         — close & finish current polygon (minimum 3 points)
  Ctrl+Z              — undo last point
  N                   — start a new polygon (after closing one)
  S                   — save all polygons to JSON
  C                   — clear all polygons and start over
  Q / Esc             — quit (prompts save if unsaved changes)

Usage:
  python roi_polygon_editor.py <image_path>
  python roi_polygon_editor.py                  # opens file dialog
"""

import cv2
import json
import numpy as np
import os
import sys
import copy


# ──────────────────────────────────────────────
#  Colour palette
# ──────────────────────────────────────────────
PALETTE = [
    (0, 255, 120),    # green
    (0, 180, 255),    # sky-blue
    (255, 100, 0),    # orange
    (200, 0, 255),    # purple
    (0, 255, 255),    # cyan
    (255, 255, 0),    # yellow
]
ACTIVE_COLOR   = (255, 255, 255)
POINT_RADIUS   = 5
LINE_THICKNESS = 2
FILL_ALPHA     = 0.18


class ROIEditor:
    def __init__(self, image_path: str):
        self.image_path   = os.path.abspath(image_path)
        self.json_path    = os.path.splitext(self.image_path)[0] + ".json"
        self.original     = cv2.imread(self.image_path)
        if self.original is None:
            raise FileNotFoundError(f"Cannot open image: {self.image_path}")

        self.canvas       = self.original.copy()
        self.polygons     = []          # list of completed polygons (list of [x,y])
        self.current      = []          # points of polygon being drawn
        self.mouse_pos    = (0, 0)
        self.unsaved      = False
        self.window_name  = f"ROI Editor — {os.path.basename(self.image_path)}"

        # Load existing JSON if present
        self._load_existing()

    # ── persistence ───────────────────────────

    def _load_existing(self):
        if os.path.exists(self.json_path):
            with open(self.json_path) as f:
                data = json.load(f)
            self.polygons = [p["points"] for p in data.get("polygons", [])]
            print(f"[info] Loaded {len(self.polygons)} polygon(s) from {self.json_path}")

    def save(self):
        all_polys = copy.deepcopy(self.polygons)
        # include the current open polygon if it has ≥3 points
        if len(self.current) >= 3:
            all_polys.append(self.current[:])

        payload = {
            "image": os.path.basename(self.image_path),
            "image_width":  self.original.shape[1],
            "image_height": self.original.shape[0],
            "polygons": [
                {"id": i, "points": pts}
                for i, pts in enumerate(all_polys)
            ]
        }
        with open(self.json_path, "w") as f:
            json.dump(payload, f, indent=2)
        self.unsaved = False
        print(f"[saved] {len(all_polys)} polygon(s) → {self.json_path}")

    # ── drawing helpers ────────────────────────

    def _poly_color(self, idx):
        return PALETTE[idx % len(PALETTE)]

    def _draw(self):
        frame = self.original.copy()
        overlay = frame.copy()

        # ── completed polygons
        for i, pts in enumerate(self.polygons):
            if len(pts) < 2:
                continue
            color = self._poly_color(i)
            np_pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [np_pts], color)
            cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0, frame)
            overlay = frame.copy()
            cv2.polylines(frame, [np_pts], isClosed=True,
                          color=color, thickness=LINE_THICKNESS)
            for pt in pts:
                cv2.circle(frame, tuple(pt), POINT_RADIUS, color, -1)
                cv2.circle(frame, tuple(pt), POINT_RADIUS + 1, (0, 0, 0), 1)
            # label
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.putText(frame, f"ROI {i}", (cx - 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(frame, f"ROI {i}", (cx - 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # ── current (open) polygon
        if self.current:
            for pt in self.current:
                cv2.circle(frame, tuple(pt), POINT_RADIUS, ACTIVE_COLOR, -1)
                cv2.circle(frame, tuple(pt), POINT_RADIUS + 1, (80, 80, 80), 1)
            for a, b in zip(self.current, self.current[1:]):
                cv2.line(frame, tuple(a), tuple(b), ACTIVE_COLOR, LINE_THICKNESS)
            # rubber-band line to mouse
            if self.mouse_pos:
                cv2.line(frame, tuple(self.current[-1]), self.mouse_pos,
                         ACTIVE_COLOR, 1, cv2.LINE_AA)
            # close-preview if ≥3 pts
            if len(self.current) >= 3:
                cv2.line(frame, tuple(self.current[-1]), tuple(self.current[0]),
                         ACTIVE_COLOR, 1, cv2.LINE_AA)

        # ── HUD
        self._draw_hud(frame)
        self.canvas = frame

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]
        lines = [
            "Left-click: add point",
            "Right-click: close polygon",
            "Ctrl+Z: undo  |  N: new polygon",
            "S: save  |  C: clear  |  Q: quit",
        ]
        pad = 8
        lh  = 20
        bh  = pad * 2 + lh * len(lines)
        bw  = 270
        sub = frame[h - bh - 10: h - 10, 10: 10 + bw]
        black = np.zeros_like(sub)
        cv2.addWeighted(black, 0.55, sub, 0.45, 0, sub)
        frame[h - bh - 10: h - 10, 10: 10 + bw] = sub
        for i, txt in enumerate(lines):
            y = h - bh - 10 + pad + (i + 1) * lh - 4
            cv2.putText(frame, txt, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # status bar top
        n_done   = len(self.polygons)
        n_cur    = len(self.current)
        status   = (f"Polygons: {n_done} saved  |  Current: {n_cur} pts"
                    + ("  [unsaved]" if self.unsaved else ""))
        cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
        cv2.putText(frame, status, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # ── mouse callback ─────────────────────────

    def _mouse(self, event, x, y, flags, _):
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append([x, y])
            self.unsaved = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current) >= 3:
                self.polygons.append(self.current[:])
                self.current = []
                self.unsaved = True
                print(f"[polygon {len(self.polygons)-1}] closed "
                      f"({len(self.polygons[-1])} points)")
            else:
                print("[warn] Need at least 3 points to close a polygon.")

        self._draw()

    # ── main loop ─────────────────────────────

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse)
        self._draw()

        print("\n=== ROI Polygon Editor ===")
        print(f"Image : {self.image_path}")
        print(f"Output: {self.json_path}")
        print("Left-click to add points, right-click to close polygon.\n")

        while True:
            cv2.imshow(self.window_name, self.canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s'):
                self.save()
                self._draw()

            elif key == ord('n'):
                if self.current:
                    print("[info] Discarded open polygon. Starting new one.")
                    self.current = []
                    self._draw()

            elif key == 26 or (key == ord('z') and
                               (cv2.getWindowProperty(self.window_name,
                                cv2.WND_PROP_VISIBLE) > 0)):
                # Ctrl+Z — undo last point
                if self.current:
                    self.current.pop()
                    self._draw()

            elif key == ord('c'):
                self.polygons = []
                self.current  = []
                self.unsaved  = True
                self._draw()
                print("[info] Cleared all polygons.")

            elif key in (ord('q'), 27):  # Q or Esc
                if self.unsaved:
                    print("[prompt] Unsaved changes. Press S to save or Q again to quit.")
                    cv2.imshow(self.window_name, self.canvas)
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 == ord('s'):
                        self.save()
                    elif k2 in (ord('q'), 27):
                        break
                else:
                    break

            # Ctrl+Z via flag
            if key == 26:
                if self.current:
                    self.current.pop()
                    self._draw()

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        print("Editor closed.")


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

def pick_file():
    """Simple CLI fallback — tries tkinter, else asks user to type path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files", "*.*")]
        )
        root.destroy()
        return path
    except Exception:
        return input("Enter image path: ").strip()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        img_path = pick_file()

    if not img_path:
        print("No image selected. Exiting.")
        sys.exit(0)

    try:
        editor = ROIEditor(img_path)
        editor.run()
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)
