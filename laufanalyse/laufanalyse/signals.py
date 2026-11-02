"""Aufbereitung der Keypoint-Zeitreihen: Lücken füllen, Seitenverwechslungen korrigieren, glätten."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from .landmarks import BILATERAL_PAIRS, L_HIP, LEG_PAIRS, R_HIP


def interpolate_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Füllt NaN-Lücken bis zu `max_gap` Samples linear (entlang Achse 0)."""
    x = np.array(x, dtype=float, copy=True)
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for c in range(flat.shape[1]):
        col = flat[:, c]
        bad = np.isnan(col)
        if not bad.any() or bad.all():
            continue
        filled = np.interp(t, t[~bad], col[~bad])
        edges = np.flatnonzero(np.diff(np.concatenate([[0], bad.astype(int), [0]])))
        for s, e in zip(edges[::2], edges[1::2]):
            inner = s > 0 and e < len(col)
            if inner and (e - s) <= max_gap:
                col[s:e] = filled[s:e]
        flat[:, c] = col
    return flat.reshape(x.shape)


def lowpass(x: np.ndarray, fps: float, cutoff_hz: float = 6.0, order: int = 4) -> np.ndarray:
    """Phasenfreier Butterworth-Tiefpass (entlang Achse 0). NaN-Abschnitte bleiben NaN."""
    x = np.asarray(x, dtype=float)
    wn = min(cutoff_hz / (fps / 2.0), 0.95)
    b, a = butter(order, wn)
    out = np.full_like(x, np.nan)
    flat_in = x.reshape(len(x), -1)
    flat_out = out.reshape(len(x), -1)
    padlen = 3 * max(len(a), len(b))
    for c in range(flat_in.shape[1]):
        col = flat_in[:, c]
        valid = ~np.isnan(col)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(int), [0]])))
        for s, e in zip(edges[::2], edges[1::2]):
            seg = col[s:e]
            flat_out[s:e, c] = filtfilt(b, a, seg) if len(seg) > padlen else seg
    return out


def _swap(arr: np.ndarray, t: int, pairs) -> None:
    for a, b in pairs:
        arr[t, [a, b]] = arr[t, [b, a]]


def fix_frontal_sides(kp: np.ndarray, vis: np.ndarray, view: str) -> int:
    """Erzwingt in Front-/Rückansicht die anatomisch korrekte Links/Rechts-Zuordnung.

    Von vorne liegt die linke Hüfte der Person rechts im Bild (größeres x),
    von hinten links im Bild. Gibt die Zahl korrigierter Frames zurück.
    """
    n = 0
    for t in range(len(kp)):
        lx, rx = kp[t, L_HIP, 0], kp[t, R_HIP, 0]
        if np.isnan(lx) or np.isnan(rx):
            continue
        wrong = (lx < rx) if view == "vorne" else (lx > rx)
        if wrong:
            _swap(kp, t, BILATERAL_PAIRS)
            _swap(vis, t, BILATERAL_PAIRS)
            n += 1
    return n


def fix_leg_swaps(kp: np.ndarray, vis: np.ndarray, ratio: float = 0.5) -> int:
    """Korrigiert sprunghafte Links/Rechts-Vertauschungen der Beine (typisch in der Seitenansicht).

    Vergleicht jeden Frame mit einer linearen Vorhersage aus den zwei Vorframes; ist die
    vertauschte Zuordnung deutlich näher, werden die Beinpunkte getauscht.
    """
    n = 0
    legs_l = [a for a, _ in LEG_PAIRS]
    legs_r = [b for _, b in LEG_PAIRS]
    both = legs_l + legs_r
    for t in range(2, len(kp)):
        prev1, prev2 = kp[t - 1], kp[t - 2]
        pred = np.where(np.isnan(prev2), prev1, 2 * prev1 - prev2)
        cur = kp[t]
        if np.isnan(cur[both]).any() or np.isnan(pred[both]).any():
            continue
        keep = (np.linalg.norm(cur[legs_l] - pred[legs_l], axis=1).sum()
                + np.linalg.norm(cur[legs_r] - pred[legs_r], axis=1).sum())
        swap = (np.linalg.norm(cur[legs_r] - pred[legs_l], axis=1).sum()
                + np.linalg.norm(cur[legs_l] - pred[legs_r], axis=1).sum())
        thigh = np.nanmean([np.linalg.norm(cur[25] - cur[23]), np.linalg.norm(cur[26] - cur[24])])
        # nur bei klarem Sprung tauschen, damit das normale Überkreuzen der Beine nicht kippt
        if swap < ratio * keep and keep > 0.5 * thigh:
            _swap(kp, t, LEG_PAIRS)
            _swap(vis, t, LEG_PAIRS)
            n += 1
    return n


def clean_keypoints(kp: np.ndarray, vis: np.ndarray, fps: float, view: str,
                    min_vis: float = 0.3, cutoff_hz: float = 6.0):
    """Komplette Aufbereitung. Gibt (geglättete kp, vis, Info-Dict) zurück."""
    kp = np.array(kp, dtype=float, copy=True)
    vis = np.array(vis, dtype=float, copy=True)

    info = {}
    if view in ("vorne", "hinten"):
        info["seitenkorrekturen"] = fix_frontal_sides(kp, vis, view)
    else:
        info["seitenkorrekturen"] = fix_leg_swaps(kp, vis)

    kp[vis < min_vis] = np.nan
    kp = interpolate_gaps(kp, max_gap=max(2, int(0.25 * fps)))
    cutoff = min(cutoff_hz, 0.4 * fps / 2)
    kp = lowpass(kp, fps, cutoff_hz=cutoff)
    info["filter_hz"] = round(cutoff, 2)
    return kp, vis, info
