"""Gesamtablauf: Video -> Pose -> Aufbereitung -> Kennzahlen -> Overlay/Exporte."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .metrics import AnalysisResult, Settings, analyze
from .overlay import render_overlay
from .pose import PoseSequence, extract_pose
from .signals import clean_keypoints

ProgressFn = Callable[[float, str], None]


@dataclass
class RunOutput:
    pose: PoseSequence
    kp: np.ndarray           # geglättete Keypoints (T,33,2)
    result: AnalysisResult
    overlay_path: Optional[Path]
    out_dir: Path


def analyze_pose(pose: PoseSequence, settings: Settings) -> tuple[np.ndarray, AnalysisResult]:
    """Aufbereitung + Kennzahlen für eine bereits erkannte Posensequenz."""
    kp, vis, info = clean_keypoints(pose.kp, pose.vis, pose.fps, settings.view)
    result = analyze(kp, vis, pose.fps, settings)
    result.extra.update(info)
    result.extra["erkennungsrate"] = pose.detection_rate()
    if pose.fps < 50:
        result.warnings.append(f"Bildrate nur {pose.fps:.0f} fps – Kontaktzeiten und Aufsatzwinkel sind grob. "
                               "Empfehlung: 60–240 fps (Zeitlupe).")
    return kp, result


def run(video_path: str | Path, settings: Settings, out_dir: str | Path, quality: str = "full",
        start_s: float = 0.0, end_s: Optional[float] = None, make_overlay: bool = True,
        progress: Optional[ProgressFn] = None) -> RunOutput:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def sub(lo, hi):
        return None if progress is None else (lambda f, msg: progress(lo + (hi - lo) * f, msg))

    pose = extract_pose(video_path, quality=quality, start_s=start_s, end_s=end_s, progress=sub(0.0, 0.7))
    if pose.detection_rate() < 0.3:
        raise RuntimeError(f"Person nur in {pose.detection_rate():.0%} der Frames erkannt – "
                           "bitte Video prüfen (ganze Person im Bild, ausreichend Licht).")
    return run_pose(pose, video_path, settings, out_dir, make_overlay, sub(0.7, 1.0))


def run_pose(pose: PoseSequence, video_path: str | Path, settings: Settings, out_dir: str | Path,
             make_overlay: bool = True, progress: Optional[ProgressFn] = None) -> RunOutput:
    """Analyse + Exporte für eine vorhandene Posensequenz (z. B. gespeichert oder Demo)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose.save(out_dir / "pose.npz")
    if progress:
        progress(0.05, "Glätten & Analysieren …")
    kp, result = analyze_pose(pose, settings)
    result.metrics_table().to_csv(out_dir / "kennzahlen.csv", index=False)
    result.timeseries.to_csv(out_dir / "zeitreihen.csv", index=False)

    overlay = None
    if make_overlay:
        sub = None if progress is None else (lambda f, msg: progress(0.1 + 0.9 * f, msg))
        overlay = render_overlay(str(video_path), pose.frame_idx, kp, out_dir / "overlay.mp4",
                                 pose.fps, result, progress=sub)
    if progress:
        progress(1.0, "Fertig")
    return RunOutput(pose, kp, result, overlay, out_dir)
