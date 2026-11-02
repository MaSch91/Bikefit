"""Zeichnet Skelett, Winkel und Bodenkontakte ins Video."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .landmarks import DRAWN_POINTS, LEFT_IDS, RIGHT_IDS, SIDES, SKELETON

COL_LEFT = (214, 120, 42)    # BGR von #2a78d6 (blau)  = linke Körperseite
COL_RIGHT = (52, 104, 235)   # BGR von #eb6834 (orange) = rechte Körperseite
COL_MID = (220, 220, 220)
COL_CONTACT = (60, 200, 60)  # grün = Fuß am Boden


def _color(a: int, b: int):
    if a in LEFT_IDS and b in LEFT_IDS:
        return COL_LEFT
    if a in RIGHT_IDS and b in RIGHT_IDS:
        return COL_RIGHT
    return COL_MID


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def _label(img, text, org, scale, color=(255, 255, 255)):
    th = max(1, int(round(2 * scale)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, th, cv2.LINE_AA)


def draw_frame(img: np.ndarray, kp: np.ndarray, i: int, result=None, hud: Optional[list[str]] = None) -> np.ndarray:
    """Zeichnet Sequenzindex `i` der Keypoints `kp` (T,33,2) auf `img` (Originalauflösung)."""
    h, w = img.shape[:2]
    s = max(h, w) / 1000.0
    pts = kp[i]
    for a, b in SKELETON:
        if not np.isnan(pts[[a, b]]).any():
            cv2.line(img, _pt(pts[a]), _pt(pts[b]), _color(a, b), max(2, int(4 * s)), cv2.LINE_AA)
    for j in DRAWN_POINTS:
        if np.isnan(pts[j]).any():
            continue
        col = COL_LEFT if j in LEFT_IDS else COL_RIGHT if j in RIGHT_IDS else COL_MID
        cv2.circle(img, _pt(pts[j]), max(3, int(6 * s)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, _pt(pts[j]), max(2, int(4 * s)), col, -1, cv2.LINE_AA)

    if result is not None:
        ts = result.timeseries
        sagittal = result.settings.view == "seite"
        for side, ids in SIDES.items():
            col = COL_LEFT if side == "links" else COL_RIGHT
            knee = pts[ids["knee"]]
            key = f"{side}_knie" if sagittal else f"{side}_knie_fppa"
            if key in ts and not np.isnan(knee).any() and not np.isnan(ts[key].iat[i]):
                off = int(14 * s) if side == "rechts" else -int(85 * s)
                _label(img, f"{ts[key].iat[i]:.0f} deg", (int(knee[0]) + off, int(knee[1])), s, col)
            if ts[f"kontakt_{side}"].iat[i]:
                for k in ("heel", "foot"):
                    p = pts[ids[k]]
                    if not np.isnan(p).any():
                        cv2.circle(img, _pt(p), max(6, int(12 * s)), COL_CONTACT, max(2, int(3 * s)), cv2.LINE_AA)
    if hud:
        y = int(35 * s)
        for line in hud:
            _label(img, line, (int(15 * s), y), s * 1.1)
            y += int(32 * s)
    return img


def read_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def render_frame(video_path: str, frame_idx_map: np.ndarray, kp: np.ndarray, i: int, result=None,
                 title: str = "") -> Optional[np.ndarray]:
    """Liefert ein einzelnes annotiertes Bild (RGB) für Sequenzindex i."""
    img = read_frame(video_path, frame_idx_map[i])
    if img is None:
        return None
    draw_frame(img, kp, i, result, hud=[title] if title else None)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_overlay(video_path: str, frame_idx_map: np.ndarray, kp: np.ndarray, out_path: str | Path,
                   fps: float, result=None, max_dim: int = 1280,
                   progress: Optional[Callable[[float, str], None]] = None) -> Path:
    """Schreibt ein H.264-Video (browsertauglich) mit eingezeichnetem Skelett."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx_map[0]))
    ok, img = cap.read()
    if not ok:
        raise IOError("Video konnte nicht gelesen werden")
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2

    hud_base = []
    if result is not None:
        cad = result.get("kadenz")
        if cad and cad.value:
            hud_base.append(f"Kadenz {cad.value:.0f} spm")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{ow}x{oh}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = len(frame_idx_map)
    try:
        for i in range(n):
            if i > 0:
                ok, img = cap.read()
                if not ok:
                    break
            draw_frame(img, kp, i, result, hud_base + [f"t = {i / fps:5.2f} s"])
            if (img.shape[1], img.shape[0]) != (ow, oh):
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
            proc.stdin.write(img.tobytes())
            if progress and i % 15 == 0:
                progress(i / n, f"Overlay-Video: Frame {i}/{n}")
    finally:
        proc.stdin.close()
        proc.wait()
        cap.release()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg-Encoding fehlgeschlagen")
    return out_path
