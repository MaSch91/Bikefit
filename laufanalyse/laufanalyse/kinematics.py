"""Gelenkwinkel aus 2D-Keypoints.

Konventionen (Bildkoordinaten: x nach rechts, y nach unten):
- Seitenansicht: `d` = Laufrichtung im Bild (+1 = nach rechts, -1 = nach links).
- Kniebeugung 0° = gestreckt, positiv = gebeugt.
- Hüfte positiv = Beugung (Oberschenkel vor dem Rumpf), negativ = Streckung.
- Unterschenkelwinkel positiv = Sprunggelenk vor dem Knie (z. B. beim Overstriding).
- Fußwinkel positiv = Fußspitze oben (Fersenaufsatz).
- Rumpfneigung positiv = Vorlage.
"""

from __future__ import annotations

import numpy as np

from .landmarks import (L_EAR, L_FOOT, L_HEEL, L_HIP, L_SHOULDER, NOSE, R_EAR, R_FOOT,
                        R_HEEL, R_HIP, R_SHOULDER, SIDES)


def included_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Innenwinkel ABC in Grad (180 = gestreckt). Arrays der Form (T, 2)."""
    v1 = a - b
    v2 = c - b
    dot = np.sum(v1 * v2, axis=-1)
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / n, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def segment_angle_from_down(prox: np.ndarray, dist: np.ndarray, d: int) -> np.ndarray:
    """Winkel des Segments prox->dist zur Senkrechten (nach unten), positiv in Laufrichtung."""
    v = dist - prox
    return np.degrees(np.arctan2(d * v[..., 0], v[..., 1]))


def facing_direction(kp: np.ndarray) -> int:
    """Schätzt die Laufrichtung in der Seitenansicht (+1 rechts, -1 links)."""
    votes = []
    for heel, foot in ((L_HEEL, L_FOOT), (R_HEEL, R_FOOT)):
        votes.append(np.nanmedian(kp[:, foot, 0] - kp[:, heel, 0]))
    nose_dx = kp[:, NOSE, 0] - np.nanmean(kp[:, [L_EAR, R_EAR], 0], axis=1)
    votes.append(np.nanmedian(nose_dx))
    s = np.nansum(np.sign(votes))
    return 1 if s >= 0 else -1


def sagittal_angles(kp: np.ndarray, d: int) -> dict:
    """Winkelverläufe der Seitenansicht. Rückgabe: {'rumpf': arr, 'links': {...}, 'rechts': {...}}."""
    sh_mid = np.nanmean(kp[:, [L_SHOULDER, R_SHOULDER]], axis=1)
    hip_mid = np.nanmean(kp[:, [L_HIP, R_HIP]], axis=1)
    v = sh_mid - hip_mid
    trunk = np.degrees(np.arctan2(d * v[:, 0], -v[:, 1]))

    out = {"rumpf": trunk}
    for side, ids in SIDES.items():
        P = {k: kp[:, i] for k, i in ids.items()}
        thigh = segment_angle_from_down(P["hip"], P["knee"], d)
        shank = segment_angle_from_down(P["knee"], P["ankle"], d)
        fv = P["foot"] - P["heel"]
        foot = np.degrees(np.arctan2(-fv[:, 1], d * fv[:, 0]))
        out[side] = {
            "knie": thigh - shank,
            "huefte": thigh + trunk,
            "oberschenkel": thigh,
            "unterschenkel": shank,
            "sprunggelenk": foot - shank,
            "fuss": foot,
            "ellbogen": 180.0 - included_angle(P["shoulder"], P["elbow"], P["wrist"]),
            "oberarm": segment_angle_from_down(P["shoulder"], P["elbow"], d),
        }
    return out


def frontal_angles(kp: np.ndarray) -> dict:
    """Winkel der Front-/Rückansicht.

    - becken: Beckenschiefstand, positiv = rechte Hüfte tiefer (relativ zum Clip-Median)
    - rumpf_seitlich: seitliche Rumpfneigung relativ zum Clip-Median, positiv = Richtung Bild-rechts
    - knie_fppa je Seite: Frontalebenen-Projektionswinkel, positiv = Valgus (Knie nach innen)
    """
    lh, rh = kp[:, L_HIP], kp[:, R_HIP]
    width = np.abs(lh[:, 0] - rh[:, 0])
    obliq = np.degrees(np.arctan2(rh[:, 1] - lh[:, 1], width))

    sh_mid = np.nanmean(kp[:, [L_SHOULDER, R_SHOULDER]], axis=1)
    hip_mid = np.nanmean(kp[:, [L_HIP, R_HIP]], axis=1)
    v = sh_mid - hip_mid
    lateral = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))

    out = {
        "becken": obliq - np.nanmedian(obliq),
        "rumpf_seitlich": lateral - np.nanmedian(lateral),
        "hueftbreite": width,
    }
    for side, ids in SIDES.items():
        hip, knee, ankle = kp[:, ids["hip"]], kp[:, ids["knee"]], kp[:, ids["ankle"]]
        dev = 180.0 - included_angle(hip, knee, ankle)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (knee[:, 1] - hip[:, 1]) / (ankle[:, 1] - hip[:, 1])
        x_line = hip[:, 0] + frac * (ankle[:, 0] - hip[:, 0])
        medial_dir = np.sign(hip_mid[:, 0] - hip[:, 0])
        sign = np.sign((knee[:, 0] - x_line) * medial_dir)
        sign[sign == 0] = 1
        out[side] = {"knie_fppa": dev * sign}
    return out
