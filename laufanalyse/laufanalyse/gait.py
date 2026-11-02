"""Schritterkennung: Bodenkontakte, Kadenz, Zykluszerlegung."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import welch


@dataclass
class Contact:
    side: str
    start: int  # Fußaufsatz (Index innerhalb der Sequenz)
    end: int    # Abdruck / Zehenablösung

    def duration(self, fps: float) -> float:
        return (self.end - self.start) / fps


def foot_low_y(kp: np.ndarray, heel: int, foot: int) -> np.ndarray:
    """Tiefster Punkt des Fußes im Bild (größtes y)."""
    return np.fmax(kp[:, heel, 1], kp[:, foot, 1])


def _runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = mask.copy()
    if max_gap <= 0:
        return mask
    for s, e in _runs(~mask):
        if 0 < s and e < len(mask) and (e - s) <= max_gap:
            mask[s:e] = True
    return mask


def detect_contacts(low_y: np.ndarray, fps: float, ref_len: float, side: str,
                    thresh_frac: float = 0.06, ground_window_s: float = 2.0,
                    min_contact_s: float = 0.06, min_gap_s: float = 0.04) -> list[Contact]:
    """Findet Bodenkontaktphasen eines Fußes.

    Der Boden wird als gleitendes 95. Perzentil der tiefsten Fußposition geschätzt; der Fuß gilt
    als am Boden, solange er weniger als `thresh_frac * ref_len` (Beinlänge in px) darüber liegt.
    Kontakte, die am Clip-Anfang oder -Ende abgeschnitten sind, werden verworfen.
    """
    y = np.asarray(low_y, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < fps:
        return []
    y_f = np.where(valid, y, np.nanmedian(y))
    win = max(3, int(ground_window_s * fps) | 1)
    ground = percentile_filter(y_f, 95, size=win, mode="nearest")
    on = (y_f >= ground - thresh_frac * ref_len) & valid
    on = _close_gaps(on, int(min_gap_s * fps))

    contacts = []
    for s, e in _runs(on):
        if (e - s) >= max(2, int(min_contact_s * fps)) and s > 0 and e < len(y):
            contacts.append(Contact(side, int(s), int(e - 1)))
    return contacts


def cadence_from_contacts(contacts: list[Contact], fps: float) -> float | None:
    """Kadenz (Schritte/min) aus den Zeitabständen gleichseitiger Fußaufsätze."""
    stride_times = []
    for side in ("links", "rechts"):
        starts = sorted(c.start for c in contacts if c.side == side)
        stride_times += list(np.diff(starts) / fps)
    stride_times = [s for s in stride_times if 0.4 < s < 2.0]
    if len(stride_times) < 2:
        return None
    return 120.0 / float(np.median(stride_times))


def cadence_from_spectrum(signals: list[np.ndarray], fps: float) -> float | None:
    """Kadenz aus der dominanten Frequenz der Links-Rechts-Differenzsignale (Gegencheck).

    Ein Doppelschritt ist eine volle Periode des Differenzsignals; Kadenz = 2 * f * 60.
    """
    psd_sum, freqs = None, None
    for s in signals:
        s = np.asarray(s, dtype=float)
        if np.isnan(s).all() or len(s) < 2 * fps:
            continue
        s = np.where(np.isnan(s), np.nanmean(s), s)
        s = s - s.mean()
        f, p = welch(s, fs=fps, nperseg=min(len(s), int(8 * fps)), nfft=max(4096, len(s)))
        p = p / (p.max() + 1e-12)
        psd_sum = p if psd_sum is None else psd_sum + p
        freqs = f
    if psd_sum is None:
        return None
    band = (freqs >= 0.5) & (freqs <= 2.0)  # 60–240 Schritte/min
    if not band.any():
        return None
    return float(2 * freqs[band][np.argmax(psd_sum[band])] * 60)


def strides(contacts: list[Contact], side: str, fps: float) -> list[tuple[int, int, int]]:
    """Doppelschritte einer Seite als (Aufsatz, Abdruck, nächster Aufsatz)."""
    cs = sorted((c for c in contacts if c.side == side), key=lambda c: c.start)
    out = []
    for a, b in zip(cs[:-1], cs[1:]):
        if 0.4 < (b.start - a.start) / fps < 2.0:
            out.append((a.start, a.end, b.start))
    return out


def normalize_cycles(series: np.ndarray, cycles: list[tuple[int, int, int]], n: int = 101):
    """Normiert ein Signal auf 0–100 % Laufzyklus. Rückgabe: (Matrix [k, n], Abdruck-% je Zyklus)."""
    rows, toe_off = [], []
    x_new = np.linspace(0, 1, n)
    for s, to, e in cycles:
        seg = series[s:e + 1]
        if len(seg) < 5 or np.isnan(seg).mean() > 0.2:
            continue
        x = np.linspace(0, 1, len(seg))
        ok = ~np.isnan(seg)
        rows.append(np.interp(x_new, x[ok], seg[ok]))
        toe_off.append(100.0 * (to - s) / (e - s))
    if not rows:
        return np.empty((0, n)), []
    return np.vstack(rows), toe_off
