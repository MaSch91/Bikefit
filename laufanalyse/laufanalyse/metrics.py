"""Berechnung der Laufkennzahlen aus aufbereiteten Keypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import gait
from .kinematics import facing_direction, frontal_angles, sagittal_angles
from .landmarks import L_ANKLE, L_HIP, L_WRIST, R_ANKLE, R_HIP, R_WRIST, SIDES

# Anthropometrie nach Winter (2009): Oberschenkel 0.245·H, Unterschenkel 0.246·H
THIGH_FRAC, SHANK_FRAC = 0.245, 0.246

SIDE_NAMES = ("links", "rechts")
INF = float("inf")


@dataclass
class Settings:
    view: str = "seite"            # "seite" | "vorne" | "hinten"
    setting: str = "laufband"      # "laufband" | "draussen"
    speed_kmh: Optional[float] = None
    height_cm: Optional[float] = None


@dataclass
class Ref:
    """Orientierungsbereich: ok = [ok_lo, ok_hi], Hinweis bis [warn_lo, warn_hi], sonst auffällig."""
    ok_lo: float
    ok_hi: float
    warn_lo: float
    warn_hi: float
    text: str

    def status(self, v: float) -> str:
        if v is None or np.isnan(v):
            return "info"
        if self.ok_lo <= v <= self.ok_hi:
            return "ok"
        if self.warn_lo <= v <= self.warn_hi:
            return "hinweis"
        return "auffällig"


# Orientierungswerte für Freizeitläufer (stark tempo- und personenabhängig – keine Diagnose).
REFS = {
    "kadenz": Ref(165, 190, 155, 200, "ca. 165–190 Schritte/min (tempoabhängig)"),
    "kontaktzeit": Ref(180, 300, 150, 340, "ca. 180–300 ms bei Freizeittempo"),
    "duty_factor": Ref(0.25, 0.40, 0.20, 0.45, "Anteil Bodenkontakt am Doppelschritt, ca. 0.25–0.40"),
    "knie_aufsatz": Ref(10, 25, 5, 30, "ca. 10–25° Beugung beim Aufsetzen"),
    "knie_stuetz_max": Ref(35, 50, 30, 55, "ca. 35–50° max. Beugung in der Stützphase"),
    "knie_schwung_max": Ref(80, 130, 65, 140, "ca. 80–130° (steigt mit dem Tempo)"),
    "unterschenkel_aufsatz": Ref(-2, 8, -6, 12, "Schienbein beim Aufsetzen nahe senkrecht (ca. 0–8°)"),
    "overstride": Ref(-INF, 20, -INF, 30, "Ferse möglichst nah unter dem Körperschwerpunkt (< ca. 20 cm)"),
    "huefte_streckung": Ref(5, 20, 0, 25, "ca. 5–20° Hüftstreckung beim Abdruck"),
    "rumpf": Ref(2, 10, 0, 14, "leichte Vorlage, ca. 2–10°"),
    "vert_osz": Ref(5, 10, 4, 12, "ca. 5–10 cm"),
    "ellbogen": Ref(70, 110, 55, 125, "ca. 70–110° Beugung"),
    "beckenabkippen": Ref(-INF, 5, -INF, 8, "unter ca. 5° unauffällig, über ca. 8° deutlich"),
    "knie_valgus": Ref(-INF, 8, -INF, 12, "Knie-Einwärtsknick in der Stützphase unter ca. 8–10°"),
    "spurbreite": Ref(0.0, INF, -0.15, INF, "Fuß landet nicht über der Körpermitte (Wert ≥ 0)"),
    "symmetrie": Ref(-INF, 5, -INF, 10, "Seitenunterschied unter ca. 5 %"),
}


@dataclass
class Metric:
    key: str
    label: str
    value: Optional[float]
    unit: str
    side: Optional[str] = None
    status: str = "info"
    ref_text: str = ""
    info: str = ""

    def as_row(self) -> dict:
        return dict(Kennzahl=self.label, Seite=self.side or "–",
                    Wert=None if self.value is None else round(float(self.value), 2 if abs(self.value) < 1 else 1),
                    Einheit=self.unit, Bewertung=self.status, Orientierung=self.ref_text,
                    Hinweis=self.info)


@dataclass
class AnalysisResult:
    settings: Settings
    fps: float
    metrics: list[Metric]
    timeseries: pd.DataFrame
    contacts: list[gait.Contact]
    cycles: dict                      # {winkel: {seite: (matrix, toe_off_list)}}
    keyframes: list[tuple[str, int]]  # (Beschriftung, Sequenz-Index)
    angles: dict
    warnings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def metrics_table(self) -> pd.DataFrame:
        return pd.DataFrame([m.as_row() for m in self.metrics])

    def get(self, key: str, side: Optional[str] = None) -> Optional[Metric]:
        for m in self.metrics:
            if m.key == key and m.side == side:
                return m
        return None


# --------------------------------------------------------------------------- Hilfsfunktionen

def _nanmedian(values) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return float(np.nanmedian(arr)) if arr.size and not np.isnan(arr).all() else float("nan")


def _seg_len(kp, a, b) -> np.ndarray:
    return np.linalg.norm(kp[:, a] - kp[:, b], axis=1)


def _metric(key, label, value, unit, side=None, ref_key=None, info="") -> Metric:
    v = None if value is None or np.isnan(value) else float(value)
    ref = REFS.get(ref_key or key)
    status = ref.status(v) if (ref and v is not None) else "info"
    return Metric(key, label, v, unit, side, status, ref.text if ref else "", info)


def _leg_len_px(kp) -> float:
    vals = [np.nanpercentile(_seg_len(kp, ids["hip"], ids["knee"]), 90)
            + np.nanpercentile(_seg_len(kp, ids["knee"], ids["ankle"]), 90) for ids in SIDES.values()]
    return float(np.nanmax(vals))


def _scale_m_per_px(leg_px: float, height_cm: Optional[float]) -> Optional[float]:
    """Maßstab aus Körpergröße und Beinsegmentlängen (90. Perzentil gegen perspektivische Verkürzung)."""
    if not height_cm or not leg_px or np.isnan(leg_px):
        return None
    return (THIGH_FRAC + SHANK_FRAC) * height_cm / 100.0 / leg_px


def _common_gait(kp, fps, s: Settings, warnings: list[str]) -> dict:
    """Kontakte, Kadenz, Tempo – für alle Ansichten gleich."""
    leg_px = _leg_len_px(kp)
    contacts = []
    for side, ids in SIDES.items():
        low = gait.foot_low_y(kp, ids["heel"], ids["foot"])
        contacts += gait.detect_contacts(low, fps, leg_px, side)
    contacts.sort(key=lambda c: c.start)

    cad_ev = gait.cadence_from_contacts(contacts, fps)
    diff_signals = [kp[:, L_ANKLE, 0] - kp[:, R_ANKLE, 0], kp[:, L_ANKLE, 1] - kp[:, R_ANKLE, 1]]
    cad_fft = gait.cadence_from_spectrum(diff_signals, fps)
    cadence, cad_src = cad_ev, "Bodenkontakte"
    if cad_ev is None and cad_fft is not None:
        cadence, cad_src = cad_fft, "Frequenzanalyse"
        warnings.append("Zu wenige saubere Bodenkontakte – Kadenz per Frequenzanalyse geschätzt.")
    elif cad_ev is not None and cad_fft is not None and abs(cad_ev - cad_fft) / cad_fft > 0.1:
        warnings.append(f"Kadenz aus Bodenkontakten ({cad_ev:.0f}) und Frequenzanalyse ({cad_fft:.0f}) "
                        "weichen ab – bitte Overlay-Video prüfen.")

    scale = _scale_m_per_px(leg_px, s.height_cm)
    speed, speed_src = None, None
    if s.speed_kmh:
        speed, speed_src = s.speed_kmh / 3.6, "Eingabe"
    elif s.setting == "draussen" and scale and s.view == "seite":
        hip_x = np.nanmean(kp[:, [L_HIP, R_HIP], 0], axis=1)
        ok = ~np.isnan(hip_x)
        if ok.sum() > fps and abs(hip_x[ok][-1] - hip_x[ok][0]) > 1.5 * leg_px:
            t = np.flatnonzero(ok) / fps
            speed = abs(np.polyfit(t, hip_x[ok], 1)[0]) * scale
            speed_src = "Videobild (feste Kamera)"
        else:
            warnings.append("Läufer bewegt sich kaum durchs Bild (mitschwenkende Kamera?) – "
                            "Schrittlänge nur mit Tempo-Angabe berechenbar.")
    return dict(contacts=contacts, cadence=cadence, cad_src=cad_src, cad_ev=cad_ev, cad_fft=cad_fft,
                scale=scale, speed=speed, speed_src=speed_src, leg_px=leg_px)


def _gait_metrics(g) -> list[Metric]:
    cad = g["cadence"]
    ms = [_metric("kadenz", "Schrittfrequenz (Kadenz)", cad, "Schritte/min", info=f"Quelle: {g['cad_src']}")]
    if g["speed"] and cad:
        step = g["speed"] / (cad / 60.0)
        ms += [
            _metric("schrittlaenge", "Schrittlänge", step * 100, "cm", info=f"Tempo-Quelle: {g['speed_src']}"),
            _metric("doppelschritt", "Doppelschrittlänge", 2 * step * 100, "cm"),
            _metric("tempo", "Geschwindigkeit", g["speed"] * 3.6, "km/h", info=f"Quelle: {g['speed_src']}"),
        ]
    else:
        ms.append(_metric("schrittlaenge", "Schrittlänge", None, "cm",
                          info="Benötigt Bandgeschwindigkeit/Tempo (draußen alternativ Körpergröße + feste Kamera)."))
    return ms


def _symmetry(a: float, b: float) -> float:
    if a is None or b is None or np.isnan(a) or np.isnan(b) or (abs(a) + abs(b)) == 0:
        return float("nan")
    return 200.0 * abs(a - b) / (abs(a) + abs(b))


def _strike_pattern(foot_angle: float) -> str:
    """Fußaufsatz nach Altman & Davis (2012): >8° Rückfuß, -1.6..8° Mittelfuß, < -1.6° Vorfuß."""
    if foot_angle is None or np.isnan(foot_angle):
        return ""
    if foot_angle > 8:
        return "→ Rückfuß-/Fersenaufsatz"
    if foot_angle >= -1.6:
        return "→ Mittelfußaufsatz"
    return "→ Vorfußaufsatz"


# --------------------------------------------------------------------------- Seitenansicht

def analyze_side(kp, vis, fps, s: Settings) -> AnalysisResult:
    warnings: list[str] = []
    d = facing_direction(kp)
    ang = sagittal_angles(kp, d)
    g = _common_gait(kp, fps, s, warnings)
    contacts, scale, leg_px = g["contacts"], g["scale"], g["leg_px"]

    vis_side = {side: float(np.nanmean(vis[:, [ids["knee"], ids["ankle"]]])) for side, ids in SIDES.items()}
    near = max(vis_side, key=vis_side.get)

    metrics = _gait_metrics(g)
    hip_mid = np.nanmean(kp[:, [L_HIP, R_HIP]], axis=1)
    per_side, cycles = {}, {}
    for side, ids in SIDES.items():
        A = ang[side]
        st = gait.strides(contacts, side, fps)
        cs = [c for c in contacts if c.side == side]
        over = [d * (kp[c.start, ids["heel"], 0] - hip_mid[c.start, 0]) for c in cs]
        if scale:
            over_v, over_u = [o * scale * 100 for o in over], "cm"
        else:
            over_v, over_u = [100 * o / leg_px for o in over], "% Beinlänge"
        per_side[side] = dict(
            n=len(cs), over_unit=over_u, overstride=_nanmedian(over_v),
            kontaktzeit=_nanmedian([c.duration(fps) * 1000 for c in cs]),
            knie_aufsatz=_nanmedian([A["knie"][c.start] for c in cs]),
            unterschenkel_aufsatz=_nanmedian([A["unterschenkel"][c.start] for c in cs]),
            fuss_aufsatz=_nanmedian([A["fuss"][c.start] for c in cs]),
            knie_stuetz_max=_nanmedian([np.nanmax(A["knie"][c.start:c.end + 1]) for c in cs]),
            huefte_streckung=-_nanmedian([np.nanmin(A["huefte"][max(0, c.end - 2):c.end + 3]) for c in cs]),
            knie_schwung_max=_nanmedian([np.nanmax(A["knie"][to:e + 1]) for _, to, e in st]),
            duty_factor=_nanmedian([(to - a) / (e - a) for a, to, e in st]),
        )
        for name in ("knie", "huefte", "sprunggelenk", "ellbogen"):
            cycles.setdefault(name, {})[side] = gait.normalize_cycles(A[name], st)

    for side in SIDE_NAMES:
        p = per_side[side]
        tag = "kameranah" if side == near else "kamerafern – ungenauer"
        metrics += [
            _metric("kontaktzeit", "Bodenkontaktzeit", p["kontaktzeit"], "ms", side, info=f"{p['n']} Kontakte, {tag}"),
            _metric("duty_factor", "Duty Factor", p["duty_factor"], "", side),
            _metric("knie_aufsatz", "Kniebeugung beim Aufsetzen", p["knie_aufsatz"], "°", side),
            _metric("knie_stuetz_max", "Max. Kniebeugung Stützphase", p["knie_stuetz_max"], "°", side),
            _metric("knie_schwung_max", "Max. Kniebeugung Schwungphase", p["knie_schwung_max"], "°", side),
            _metric("unterschenkel_aufsatz", "Schienbeinwinkel beim Aufsetzen", p["unterschenkel_aufsatz"], "°",
                    side, info="positiv = Fuß landet vor dem Knie"),
            _metric("overstride", "Fußaufsatz vor der Hüfte", p["overstride"], p["over_unit"], side,
                    ref_key="overstride" if p["over_unit"] == "cm" else "_none"),
            _metric("huefte_streckung", "Hüftstreckung beim Abdruck", p["huefte_streckung"], "°", side),
            _metric("fuss_aufsatz", "Fußwinkel beim Aufsetzen", p["fuss_aufsatz"], "°", side,
                    info=_strike_pattern(p["fuss_aufsatz"])),
        ]

    metrics.append(_metric("rumpf", "Rumpfvorlage", float(np.nanmedian(ang["rumpf"])), "°"))

    starts = [c.start for c in contacts]
    osc = [np.nanmax(hip_mid[a:b, 1]) - np.nanmin(hip_mid[a:b, 1])
           for a, b in zip(starts[:-1], starts[1:]) if 0.2 < (b - a) / fps < 1.0]
    if osc:
        if scale:
            metrics.append(_metric("vert_osz", "Vertikale Oszillation", _nanmedian(osc) * scale * 100, "cm"))
        else:
            metrics.append(_metric("vert_osz", "Vertikale Oszillation", 100 * _nanmedian(osc) / leg_px,
                                   "% Beinlänge", ref_key="_none", info="Für cm bitte Körpergröße angeben."))

    for side in SIDE_NAMES:
        arm = ang[side]["oberarm"]
        metrics += [
            _metric("ellbogen", "Ellbogenbeugung (Median)", float(np.nanmedian(ang[side]["ellbogen"])), "°", side),
            _metric("armschwung", "Armschwung (Spannweite Oberarm)",
                    float(np.nanpercentile(arm, 95) - np.nanpercentile(arm, 5)), "°", side),
        ]
    metrics.append(_metric("symmetrie", "Seitenunterschied Kontaktzeit",
                           _symmetry(per_side["links"]["kontaktzeit"], per_side["rechts"]["kontaktzeit"]), "%"))

    if not contacts:
        warnings.append("Keine Bodenkontakte erkannt – ist die ganze Person inkl. Füße im Bild?")
    extra = dict(richtung="rechts" if d > 0 else "links", kameraseite=near, scale_m_per_px=scale,
                 kadenz_kontakte=g["cad_ev"], kadenz_frequenz=g["cad_fft"])
    return AnalysisResult(s, fps, metrics, _timeseries(fps, ang, contacts, len(kp)), contacts, cycles,
                          _side_keyframes(ang, contacts, near, fps), ang, warnings, extra)


def _side_keyframes(ang, contacts, side, fps):
    st = gait.strides(contacts, side, fps)
    if not st:
        return []
    a, to, e = st[len(st) // 2]
    knee = ang[side]["knie"]
    mid = a + int(np.nanargmax(knee[a:to + 1]))
    swing = to + int(np.nanargmax(knee[to:e + 1]))
    return [(f"Aufsetzen ({side})", a), (f"Mittlere Stützphase ({side})", mid),
            (f"Abdruck ({side})", to), (f"Schwungphase ({side})", swing)]


# --------------------------------------------------------------------------- Front-/Rückansicht

def analyze_frontal(kp, vis, fps, s: Settings) -> AnalysisResult:
    warnings: list[str] = []
    fa = frontal_angles(kp)
    g = _common_gait(kp, fps, s, warnings)
    contacts = g["contacts"]
    metrics = _gait_metrics(g)

    hip_mid_x = np.nanmean(kp[:, [L_HIP, R_HIP], 0], axis=1)
    width = fa["hueftbreite"]
    ids_of = {"links": (L_HIP, L_ANKLE, L_WRIST), "rechts": (R_HIP, R_ANKLE, R_WRIST)}
    cycles = {"knie_fppa": {}, "becken": {}}
    per_side = {}
    for side in SIDE_NAMES:
        hip_i, ankle_i, wrist_i = ids_of[side]
        cs = [c for c in contacts if c.side == side]
        sign = 1 if side == "links" else -1  # Stütz links -> rechte Hüfte sinkt -> becken > 0
        lat_dir = np.sign(kp[:, hip_i, 0] - hip_mid_x)
        with np.errstate(invalid="ignore", divide="ignore"):
            width_rel = [(kp[m, ankle_i, 0] - hip_mid_x[m]) * lat_dir[m] / width[m]
                         for m in ((c.start + c.end) // 2 for c in cs)]
            wrist_off = (kp[:, wrist_i, 0] - hip_mid_x) * lat_dir / width
        per_side[side] = dict(
            n=len(cs),
            drop=_nanmedian([np.nanmax(sign * fa["becken"][c.start:c.end + 1]) for c in cs]),
            valgus=_nanmedian([np.nanmax(fa[side]["knie_fppa"][c.start:c.end + 1]) for c in cs]),
            width=_nanmedian(width_rel),
            arm_cross=float(np.nanmean(wrist_off < 0) * 100),
        )
        st = gait.strides(contacts, side, fps)
        cycles["knie_fppa"][side] = gait.normalize_cycles(fa[side]["knie_fppa"], st)
        cycles["becken"][side] = gait.normalize_cycles(sign * fa["becken"], st)

    for side in SIDE_NAMES:
        p = per_side[side]
        other = "rechte" if side == "links" else "linke"
        metrics += [
            _metric("beckenabkippen", f"Beckenabkippen ({other} Seite sinkt)", p["drop"], "°", side,
                    info=f"Stützphase {side}, {p['n']} Kontakte"),
            _metric("knie_valgus", "Knie-Einwärtsknick (Valgus) Stützphase", p["valgus"], "°", side),
            _metric("spurbreite", "Fußaufsatz seitlich der Körpermitte", p["width"], "× Hüftbreite", side,
                    info="< 0 = Überkreuzen (Crossover)"),
            _metric("arm_cross", "Hand über Körpermitte", p["arm_cross"], "% der Zeit", side),
        ]
    lat = fa["rumpf_seitlich"]
    metrics.append(_metric("rumpf_seitlich", "Seitliche Rumpfbewegung (Spannweite)",
                           float(np.nanpercentile(lat, 95) - np.nanpercentile(lat, 5)), "°"))

    keyframes = []
    for side in SIDE_NAMES:
        cs = [c for c in contacts if c.side == side]
        if cs:
            c = cs[len(cs) // 2]
            m = c.start + int(np.nanargmax(fa[side]["knie_fppa"][c.start:c.end + 1]))
            keyframes += [(f"Aufsetzen ({side})", c.start), (f"Stützphase ({side})", m)]
    if not contacts:
        warnings.append("Keine Bodenkontakte erkannt – sind die Füße vollständig im Bild?")
    extra = dict(scale_m_per_px=g["scale"], kadenz_kontakte=g["cad_ev"], kadenz_frequenz=g["cad_fft"])
    return AnalysisResult(s, fps, metrics, _timeseries(fps, fa, contacts, len(kp)), contacts, cycles,
                          keyframes, fa, warnings, extra)


# --------------------------------------------------------------------------- gemeinsam

def _timeseries(fps, ang, contacts, n) -> pd.DataFrame:
    cols = {"t": np.arange(n) / fps}
    for side in SIDE_NAMES:
        for k, v in ang[side].items():
            cols[f"{side}_{k}"] = v
        flag = np.zeros(n, dtype=bool)
        for c in contacts:
            if c.side == side:
                flag[c.start:c.end + 1] = True
        cols[f"kontakt_{side}"] = flag
    for k, v in ang.items():
        if k not in SIDE_NAMES:
            cols[k] = v
    return pd.DataFrame(cols)


def analyze(kp, vis, fps, settings: Settings) -> AnalysisResult:
    if settings.view == "seite":
        return analyze_side(kp, vis, fps, settings)
    return analyze_frontal(kp, vis, fps, settings)
