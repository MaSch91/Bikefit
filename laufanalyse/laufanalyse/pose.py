"""Posenerkennung: Video -> Keypoint-Zeitreihen mit MediaPipe Pose Landmarker."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .landmarks import NUM_LANDMARKS

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

ProgressFn = Callable[[float, str], None]


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class PoseSequence:
    """Keypoints über die Zeit.

    kp:    (T, 33, 2) Bildkoordinaten in Pixeln (x nach rechts, y nach unten), NaN = nicht erkannt
    vis:   (T, 33)    Sichtbarkeit/Konfidenz 0..1
    world: (T, 33, 3) metrische 3D-Koordinaten (Hüftmitte = Ursprung), NaN = nicht erkannt
    """

    info: VideoInfo
    frame_idx: np.ndarray
    times: np.ndarray
    kp: np.ndarray
    vis: np.ndarray
    world: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.info.fps

    def detection_rate(self) -> float:
        return float(np.mean(~np.isnan(self.kp[:, 0, 0]))) if len(self.kp) else 0.0

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, frame_idx=self.frame_idx, times=self.times, kp=self.kp, vis=self.vis,
            world=self.world,
            info=np.array([self.info.fps, self.info.width, self.info.height, self.info.frame_count]),
            path=np.array(self.info.path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        d = np.load(path, allow_pickle=False)
        fps, w, h, n = d["info"]
        info = VideoInfo(str(d["path"]), float(fps), int(w), int(h), int(n))
        return cls(info, d["frame_idx"], d["times"], d["kp"], d["vis"], d["world"])


def ensure_model(quality: str = "full") -> Path:
    """Lädt das Posemodell bei Bedarf herunter und gibt den lokalen Pfad zurück."""
    if quality not in MODEL_URLS:
        raise ValueError(f"Unbekannte Modellqualität: {quality}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"pose_landmarker_{quality}.task"
    if not path.exists() or path.stat().st_size < 1_000_000:
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[quality], tmp)
        tmp.rename(path)
    return path


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Video kann nicht geöffnet werden: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()  # Größe nach automatischer Rotation (Hochkant-Videos)
    cap.release()
    if not ok:
        raise IOError(f"Video enthält keine lesbaren Frames: {path}")
    h, w = frame.shape[:2]
    return VideoInfo(str(path), float(fps), w, h, n)


def extract_pose(
    video_path: str | Path,
    quality: str = "full",
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    max_dim: int = 1280,
    min_confidence: float = 0.5,
    progress: Optional[ProgressFn] = None,
) -> PoseSequence:
    """Führt die Posenerkennung Frame für Frame aus."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    info = probe_video(video_path)
    model_path = ensure_model(quality)
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    first = int(round(start_s * info.fps))
    last = info.frame_count if end_s is None else min(info.frame_count, int(round(end_s * info.fps)))
    if first > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    total = max(1, last - first)

    scale = min(1.0, max_dim / max(info.width, info.height))
    frames, kps, viss, worlds = [], [], [], []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = first
        while idx < last:
            ok, frame = cap.read()
            if not ok:
                break
            small = (cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                     if scale < 1.0 else frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            ts_ms = int(round(idx * 1000.0 / info.fps))
            res = landmarker.detect_for_video(image, ts_ms)

            kp = np.full((NUM_LANDMARKS, 2), np.nan)
            vis = np.zeros(NUM_LANDMARKS)
            wl = np.full((NUM_LANDMARKS, 3), np.nan)
            if res.pose_landmarks:
                for j, lm in enumerate(res.pose_landmarks[0]):
                    kp[j] = (lm.x * info.width, lm.y * info.height)
                    v = lm.visibility if lm.visibility is not None else 1.0
                    p = lm.presence if lm.presence is not None else 1.0
                    vis[j] = min(v, p)
                if res.pose_world_landmarks:
                    for j, lm in enumerate(res.pose_world_landmarks[0]):
                        wl[j] = (lm.x, lm.y, lm.z)
            frames.append(idx)
            kps.append(kp)
            viss.append(vis)
            worlds.append(wl)
            idx += 1
            if progress and (idx - first) % 10 == 0:
                progress((idx - first) / total, f"Posenerkennung: Frame {idx - first}/{total}")
    cap.release()

    frame_idx = np.asarray(frames, dtype=int)
    return PoseSequence(
        info=info,
        frame_idx=frame_idx,
        times=frame_idx / info.fps,
        kp=np.asarray(kps, dtype=float).reshape(-1, NUM_LANDMARKS, 2),
        vis=np.asarray(viss, dtype=float).reshape(-1, NUM_LANDMARKS),
        world=np.asarray(worlds, dtype=float).reshape(-1, NUM_LANDMARKS, 3),
        meta={"quality": quality},
    )
