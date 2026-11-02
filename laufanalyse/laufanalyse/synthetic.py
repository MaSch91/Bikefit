"""Synthetischer Läufer (Strichfigur) mit bekannter Kadenz und bekannten Winkeln – für Tests."""

from __future__ import annotations

import numpy as np

from laufanalyse.landmarks import (L_ANKLE, L_EAR, L_ELBOW, L_FOOT, L_HEEL, L_HIP, L_KNEE,
                                   L_SHOULDER, L_WRIST, NOSE, NUM_LANDMARKS, R_ANKLE, R_EAR,
                                   R_ELBOW, R_FOOT, R_HEEL, R_HIP, R_KNEE, R_SHOULDER, R_WRIST)

THIGH = SHANK = 120.0


def leg_angles(phi):
    """Oberschenkelwinkel (vorwärts positiv) und Kniebeugung als Funktion der Zyklusphase."""
    thigh = 5.0 + 20.0 * np.cos(phi)
    knee = 20.0 + 80.0 * np.clip(-np.sin(phi), 0, None) ** 1.5
    return thigh, knee


def side_runner(fps=60.0, seconds=8.0, cadence=170.0, d=1, trunk_deg=6.0, hip=(500.0, 400.0)):
    """Erzeugt Keypoints (T,33,2) einer Seitenansicht und die exakten Sollwinkel."""
    n = int(fps * seconds)
    t = np.arange(n) / fps
    f = cadence / 120.0  # Doppelschritte pro Sekunde
    kp = np.full((n, NUM_LANDMARKS, 2), np.nan)
    truth = {}
    # Hüfte am tiefsten in der Mitte jeder Stützphase (Einfedern), ca. 6 cm Oszillation
    hip_y = hip[1] - 15.0 * np.cos(2 * 2 * np.pi * f * t)
    hip_pt = np.stack([np.full(n, hip[0]), hip_y], axis=1)
    tr = np.radians(trunk_deg)
    shoulder = hip_pt + np.array([d * 160 * np.sin(tr), -160 * np.cos(tr)])

    legs = (("links", 0.0, (L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT, L_SHOULDER, L_ELBOW, L_WRIST)),
            ("rechts", np.pi, (R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT, R_SHOULDER, R_ELBOW, R_WRIST)))
    for side, phase, ids in legs:
        phi = 2 * np.pi * f * t + phase
        th, kn = leg_angles(phi)
        sh = th - kn
        h_i, k_i, a_i, he_i, fo_i, s_i, e_i, w_i = ids
        knee = hip_pt + THIGH * np.stack([d * np.sin(np.radians(th)), np.cos(np.radians(th))], axis=1)
        ankle = knee + SHANK * np.stack([d * np.sin(np.radians(sh)), np.cos(np.radians(sh))], axis=1)
        kp[:, h_i] = hip_pt
        kp[:, k_i] = knee
        kp[:, a_i] = ankle
        kp[:, he_i] = ankle + np.array([-d * 12.0, 10.0])
        kp[:, fo_i] = ankle + np.array([d * 35.0, 10.0])
        kp[:, s_i] = shoulder
        arm = -20.0 * np.cos(phi)
        elbow = shoulder + 70 * np.stack([d * np.sin(np.radians(arm)), np.cos(np.radians(arm))], axis=1)
        fore = arm + 90.0
        kp[:, e_i] = elbow
        kp[:, w_i] = elbow + 65 * np.stack([d * np.sin(np.radians(fore)), np.cos(np.radians(fore))], axis=1)
        truth[side] = {"oberschenkel": th, "knie": kn, "unterschenkel": sh}
    head = shoulder + np.array([d * 10.0, -50.0])
    kp[:, L_EAR] = head
    kp[:, R_EAR] = head
    kp[:, NOSE] = head + np.array([d * 15.0, 0.0])
    truth["rumpf"] = np.full(n, trunk_deg)
    return kp, np.ones((n, NUM_LANDMARKS)), truth


def frontal_runner(fps=60.0, seconds=8.0, cadence=170.0, view="hinten", drop_deg=6.0):
    """Front-/Rückansicht mit Beckenabkippen der Gegenseite während der Stützphase."""
    n = int(fps * seconds)
    t = np.arange(n) / fps
    f = cadence / 120.0
    kp = np.full((n, NUM_LANDMARKS, 2), np.nan)
    cx, half = 500.0, 50.0
    sgn = -1 if view == "hinten" else 1  # Rückansicht: linke Körperseite links im Bild
    legs = (("links", 0.0, (L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT, L_SHOULDER, L_WRIST), 1),
            ("rechts", np.pi, (R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT, R_SHOULDER, R_WRIST), -1))
    for side, phase, ids, lat in legs:
        phi = 2 * np.pi * f * t + phase
        h_i, k_i, a_i, he_i, fo_i, s_i, w_i = ids
        x = cx + sgn * lat * half
        lift = 60.0 * np.clip(-np.sin(phi), 0, None) ** 1.5
        kp[:, h_i] = np.stack([np.full(n, x), np.full(n, 400.0)], axis=1)
        kp[:, k_i] = np.stack([np.full(n, x), 520.0 - lift / 2], axis=1)
        kp[:, a_i] = np.stack([np.full(n, x), 640.0 - lift], axis=1)
        kp[:, he_i] = kp[:, a_i] + np.array([0, 10.0])
        kp[:, fo_i] = kp[:, a_i] + np.array([0, 12.0])
        kp[:, s_i] = np.stack([np.full(n, cx + sgn * lat * 70), np.full(n, 240.0)], axis=1)
        kp[:, w_i] = np.stack([np.full(n, cx + sgn * lat * 60), np.full(n, 380.0)], axis=1)
    # Stütz links (sin > 0) -> rechte Hüfte sinkt um drop_deg
    tilt = np.tan(np.radians(drop_deg)) * half * np.clip(np.sin(2 * np.pi * f * t), None, None)
    kp[:, R_HIP, 1] += tilt
    kp[:, L_HIP, 1] -= tilt
    return kp, np.ones((n, NUM_LANDMARKS))


def make_demo(out_dir, seconds: float = 6.0, fps: float = 60.0, cadence: float = 172.0):
    """Erzeugt ein Demo-Video (Strichfigur) samt PoseSequence – zum Ausprobieren ohne echtes Video."""
    import subprocess
    from pathlib import Path

    import cv2

    from .landmarks import SKELETON
    from .pose import PoseSequence, VideoInfo

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kp, vis, _ = side_runner(fps=fps, seconds=seconds, cadence=cadence)
    w, h = 1000, 800
    path = out_dir / "demo_laeufer.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(len(kp)):
        img = np.full((h, w, 3), 235, np.uint8)
        cv2.rectangle(img, (0, 665), (w, h), (180, 180, 180), -1)  # "Laufband"
        for a, b in SKELETON:
            p, q = kp[i, a], kp[i, b]
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (90, 90, 90), 14, cv2.LINE_AA)
        hd = kp[i, NOSE]
        cv2.circle(img, (int(hd[0] - 10), int(hd[1])), 28, (90, 90, 90), -1, cv2.LINE_AA)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    n = len(kp)
    pose = PoseSequence(VideoInfo(str(path), fps, w, h, n), np.arange(n), np.arange(n) / fps,
                        kp, vis, np.full((n, NUM_LANDMARKS, 3), np.nan), meta={"demo": True})
    return path, pose
