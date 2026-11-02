"""Weboberfläche der Laufanalyse.  Start:  streamlit run app.py"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from laufanalyse.metrics import Settings
from laufanalyse.overlay import render_frame
from laufanalyse.pipeline import run, run_pose

APP_DIR = Path(__file__).resolve().parent
OUT_ROOT = APP_DIR / "output"

# Farben: Slot 1/2 der validierten Kategorie-Palette (links = blau, rechts = orange) – wie im Overlay
SIDE_COLORS = {"links": "#2a78d6", "rechts": "#eb6834"}
STATUS_LABEL = {"ok": "✅ im Bereich", "hinweis": "⚠️ Grenzbereich", "auffällig": "🔴 auffällig", "info": "ℹ️ Info"}
VIEW_LABEL = {"seite": "Seitenansicht", "vorne": "Ansicht von vorne", "hinten": "Ansicht von hinten"}
SHORT_UNITS = {"Schritte/min": "/min", "% Beinlänge": "% BL", "× Hüftbreite": "× HB"}
QUALITY = {"Schnell": "lite", "Standard": "full", "Genau (langsam)": "heavy"}

CYCLE_TITLES = {
    "knie": "Kniebeugung (0° = gestreckt)",
    "huefte": "Hüftwinkel (+ Beugung / − Streckung)",
    "sprunggelenk": "Sprunggelenk (+ Dorsalextension)",
    "ellbogen": "Ellbogenbeugung",
    "knie_fppa": "Knie-Frontalwinkel (+ Valgus / X-Bein-Richtung)",
    "becken": "Beckenabkippen der Gegenseite",
}

st.set_page_config(page_title="Laufanalyse", page_icon="🏃", layout="wide")


# --------------------------------------------------------------------------- Hilfen

def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:60]


def _fmt(v, unit=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    s = f"{v:.2f}" if abs(v) < 1 else f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"
    return f"{s} {unit}".strip()


def cycle_chart(cycles: dict, name: str, unit: str = "°") -> go.Figure | None:
    fig = go.Figure()
    any_data = False
    x = np.linspace(0, 100, 101)
    for side, (mat, toe) in cycles.items():
        if mat.shape[0] == 0:
            continue
        any_data = True
        col = SIDE_COLORS[side]
        mean, sd = mat.mean(axis=0), mat.std(axis=0)
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([mean + sd, (mean - sd)[::-1]]),
                                 fill="toself", fillcolor=col, opacity=0.15, line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", line=dict(color=col, width=2),
                                 name=f"{side} (n={mat.shape[0]})",
                                 hovertemplate=f"{side}: %{{y:.1f}}{unit} bei %{{x:.0f}} %<extra></extra>"))
        if toe:
            fig.add_vline(x=float(np.mean(toe)), line=dict(color=col, width=1, dash="dot"))
    if not any_data:
        return None
    fig.update_layout(
        title=dict(text=CYCLE_TITLES.get(name, name), font=dict(size=15)),
        height=300, margin=dict(l=10, r=10, t=45, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="% Laufzyklus (0 = Aufsetzen, gepunktet = Abdruck)", gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(title=f"Winkel [{unit}]", tickformat=".0f", gridcolor="rgba(128,128,128,0.15)", zeroline=False),
    )
    return fig


def time_chart(ts: pd.DataFrame, key: str, title: str) -> go.Figure:
    fig = go.Figure()
    for side in ("links", "rechts"):
        col = f"{side}_{key}"
        if col in ts:
            fig.add_trace(go.Scatter(x=ts["t"], y=ts[col], mode="lines", name=side,
                                     line=dict(color=SIDE_COLORS[side], width=2),
                                     hovertemplate=f"{side}: %{{y:.1f}}° bei %{{x:.2f}} s<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=280, hovermode="x unified",
                      margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", y=-0.35),
                      plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Zeit [s]", gridcolor="rgba(128,128,128,0.15)"),
                      yaxis=dict(title="Winkel [°]", tickformat=".0f", gridcolor="rgba(128,128,128,0.15)"))
    return fig


def metric_table(result) -> pd.DataFrame:
    df = result.metrics_table()
    df["Bewertung"] = df["Bewertung"].map(STATUS_LABEL)
    df["Wert"] = [_fmt(m.value) for m in result.metrics]
    return df[["Kennzahl", "Seite", "Wert", "Einheit", "Bewertung", "Orientierung", "Hinweis"]]


# --------------------------------------------------------------------------- Seitenleiste

if "analysen" not in st.session_state:
    st.session_state.analysen = []

with st.sidebar:
    st.header("1 · Video")
    upload = st.file_uploader("Laufvideo hochladen", type=["mp4", "mov", "m4v", "avi", "mkv", "webm"])
    view = st.radio("Kameraperspektive", list(VIEW_LABEL), format_func=VIEW_LABEL.get)
    st.header("2 · Rahmenbedingungen")
    setting = st.radio("Umgebung", ["laufband", "draussen"],
                       format_func={"laufband": "Laufband", "draussen": "Draußen"}.get, horizontal=True)
    speed = st.number_input("Geschwindigkeit [km/h] (0 = unbekannt)", 0.0, 40.0, 0.0, 0.5,
                            help="Laufband: Bandgeschwindigkeit. Draußen: z. B. Tempo von der Uhr. "
                                 "Wird für die Schrittlänge benötigt.")
    height = st.number_input("Körpergröße [cm] (0 = unbekannt)", 0, 230, 0, 1,
                             help="Dient als Maßstab für cm-Werte (Oszillation, Fußaufsatz).")
    st.header("3 · Optionen")
    quality = st.select_slider("Erkennungsgenauigkeit", list(QUALITY), value="Standard")
    trim = (0.0, None)
    if upload is not None:
        st.caption("Zeitbereich (Ende 0 = bis zum Schluss)")
        c1, c2 = st.columns(2)
        t0 = c1.number_input("Start [s]", 0.0, 3600.0, 0.0, 0.5)
        t1 = c2.number_input("Ende [s]", 0.0, 3600.0, 0.0, 0.5)
        trim = (t0, t1 if t1 > t0 else None)
    go_btn = st.button("▶ Analyse starten", type="primary", disabled=upload is None, use_container_width=True)
    demo_btn = st.button("Demo mit synthetischem Läufer", use_container_width=True)

settings = Settings(view=view, setting=setting, speed_kmh=speed or None, height_cm=height or None)


# --------------------------------------------------------------------------- Analyse ausführen

def _progress_ui():
    bar = st.progress(0.0, text="Starte …")
    return lambda f, msg: bar.progress(min(1.0, max(0.0, f)), text=msg)


if go_btn and upload is not None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    up_dir = OUT_ROOT / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    vid_path = up_dir / f"{stamp}_{_slug(upload.name)}"
    vid_path.write_bytes(upload.getbuffer())
    try:
        out = run(vid_path, settings, OUT_ROOT / f"{stamp}_{_slug(Path(upload.name).stem)}",
                  quality=QUALITY[quality], start_s=trim[0], end_s=trim[1], progress=_progress_ui())
        st.session_state.analysen.append(dict(name=upload.name, video=str(vid_path), out=out))
    except Exception as exc:  # Fehler verständlich anzeigen statt Stacktrace
        st.error(f"Analyse fehlgeschlagen: {exc}")

if demo_btn:
    from laufanalyse.synthetic import make_demo
    stamp = time.strftime("%Y%m%d-%H%M%S")
    demo_dir = OUT_ROOT / f"{stamp}_demo"
    path, pose = make_demo(demo_dir)
    demo_settings = Settings(view="seite", setting="laufband", speed_kmh=12.0, height_cm=180)
    out = run_pose(pose, path, demo_settings, demo_dir, progress=_progress_ui())
    st.session_state.analysen.append(dict(name="Demo (synthetisch, 12 km/h, 180 cm)", video=str(path), out=out))


# --------------------------------------------------------------------------- Ergebnisse

def show_analysis(a: dict, key: str):
    out = a["out"]
    res, pose = out.result, out.pose
    s = res.settings
    info = [VIEW_LABEL[s.view], f"{pose.fps:.0f} fps", f"{len(pose.kp) / pose.fps:.1f} s analysiert",
            f"Person erkannt in {pose.detection_rate():.0%} der Frames"]
    if "richtung" in res.extra:
        info.append(f"läuft nach {res.extra['richtung']}, Kameraseite: {res.extra['kameraseite']}")
    st.caption(" · ".join(info))
    for w in res.warnings:
        st.warning(w)

    left, right = st.columns([3, 2])
    with left:
        if out.overlay_path and Path(out.overlay_path).exists():
            st.video(str(out.overlay_path))
            st.caption("Blau = linke Körperseite, Orange = rechte. Grüne Kreise = Fuß am Boden.")
    with right:
        st.subheader("Auf einen Blick")
        keys = (["kadenz", "schrittlaenge", "kontaktzeit", "knie_aufsatz", "unterschenkel_aufsatz", "rumpf", "vert_osz"]
                if s.view == "seite" else ["kadenz", "schrittlaenge", "beckenabkippen", "knie_valgus", "spurbreite"])
        tiles = [m for k in keys for m in res.metrics if m.key == k and m.value is not None]
        cols = st.columns(2)
        for i, m in enumerate(tiles[:10]):
            label = m.label + (f" ({m.side})" if m.side else "")
            unit = SHORT_UNITS.get(m.unit, m.unit)
            cols[i % 2].metric(label, _fmt(m.value, unit), help=f"{m.label}: {STATUS_LABEL[m.status]} – {m.ref_text}")

    st.subheader("Kennzahlen")
    st.dataframe(metric_table(res), hide_index=True, use_container_width=True)
    st.caption("Orientierungswerte gelten für Freizeitläufer und hängen stark vom Tempo ab – "
               "sie sind keine medizinische Bewertung.")

    st.subheader("Laufzyklus (Mittelwert ± Standardabweichung)")
    cols = st.columns(2)
    for i, name in enumerate(res.cycles):
        fig = cycle_chart(res.cycles[name], name)
        if fig is not None:
            cols[i % 2].plotly_chart(fig, use_container_width=True, key=f"cyc_{key}_{name}")

    if s.view == "seite":
        st.plotly_chart(time_chart(res.timeseries, "knie", "Kniebeugung über die Zeit"),
                        use_container_width=True, key=f"time_{key}")

    if res.keyframes:
        st.subheader("Schlüsselmomente")
        cols = st.columns(len(res.keyframes))
        for c, (label, idx) in zip(cols, res.keyframes):
            img = render_frame(a["video"], pose.frame_idx, out.kp, idx, res)
            if img is not None:
                c.image(img, caption=f"{label} · {idx / pose.fps:.2f} s", use_container_width=True)

    with st.expander("Einzelbild-Explorer"):
        i = st.slider("Frame", 0, len(pose.kp) - 1, len(pose.kp) // 2, key=f"slider_{key}")
        img = render_frame(a["video"], pose.frame_idx, out.kp, i, res, title=f"t = {i / pose.fps:.2f} s")
        if img is not None:
            st.image(img, use_container_width=True)

    st.subheader("Downloads")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Kennzahlen (CSV)", (out.out_dir / "kennzahlen.csv").read_bytes(), "kennzahlen.csv",
                       key=f"d1_{key}")
    c2.download_button("Zeitreihen (CSV)", (out.out_dir / "zeitreihen.csv").read_bytes(), "zeitreihen.csv",
                       key=f"d2_{key}")
    if out.overlay_path:
        c3.download_button("Overlay-Video (MP4)", Path(out.overlay_path).read_bytes(), "overlay.mp4",
                           key=f"d3_{key}")


st.title("🏃 Laufanalyse")
analysen = st.session_state.analysen
if not analysen:
    st.markdown(
        "Lade links ein Laufvideo hoch, wähle die Perspektive und starte die Analyse – "
        "oder probiere die **Demo** aus.\n\n"
        "**Tipps für gute Aufnahmen**\n"
        "- Kamera fest (Stativ), auf Hüfthöhe, **senkrecht** zur Laufrichtung (Seite) bzw. mittig dahinter/davor.\n"
        "- Ganze Person inkl. Füße im Bild, enganliegende Kleidung, gutes Licht.\n"
        "- Möglichst **60 fps oder Zeitlupe (120/240 fps)**, mindestens 10 Schritte.\n"
        "- Laufband: Geschwindigkeit notieren. Draußen: Körpergröße und Tempo angeben."
    )
else:
    labels = [f"{i + 1}. {VIEW_LABEL[a['out'].result.settings.view]} – {a['name']}" for i, a in enumerate(analysen)]
    multi = len(analysen) > 1
    tabs = st.tabs((["Übersicht"] if multi else []) + labels)
    if multi:
        with tabs[0]:
            frames = []
            for i, a in enumerate(analysen):
                df = metric_table(a["out"].result)
                df.insert(0, "Analyse", labels[i])
                frames.append(df)
            st.dataframe(pd.concat(frames), hide_index=True, use_container_width=True)
    for i, (tab, a) in enumerate(zip(tabs[1:] if multi else tabs, analysen)):
        with tab:
            show_analysis(a, key=str(i))
    if st.sidebar.button("Ergebnisse leeren", use_container_width=True):
        st.session_state.analysen = []
        st.rerun()
