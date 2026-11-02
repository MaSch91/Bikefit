"""Kommandozeile:  python -m laufanalyse.cli VIDEO --ansicht seite --tempo 12 --groesse 180"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .metrics import Settings
from .pipeline import run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Videobasierte Laufanalyse")
    ap.add_argument("video")
    ap.add_argument("--ansicht", choices=["seite", "vorne", "hinten"], default="seite")
    ap.add_argument("--umgebung", choices=["laufband", "draussen"], default="laufband")
    ap.add_argument("--tempo", type=float, default=None, help="Geschwindigkeit in km/h")
    ap.add_argument("--groesse", type=float, default=None, help="Körpergröße in cm")
    ap.add_argument("--modell", choices=["lite", "full", "heavy"], default="full")
    ap.add_argument("--start", type=float, default=0.0, help="Startzeit [s]")
    ap.add_argument("--ende", type=float, default=None, help="Endzeit [s]")
    ap.add_argument("--ausgabe", default=None, help="Ausgabeordner (Standard: output/<videoname>)")
    ap.add_argument("--ohne-video", action="store_true", help="kein Overlay-Video erzeugen")
    a = ap.parse_args(argv)

    out_dir = Path(a.ausgabe) if a.ausgabe else Path("output") / Path(a.video).stem
    settings = Settings(view=a.ansicht, setting=a.umgebung, speed_kmh=a.tempo, height_cm=a.groesse)

    def progress(f, msg):
        print(f"\r[{f:5.0%}] {msg:<50}", end="", file=sys.stderr, flush=True)

    out = run(a.video, settings, out_dir, quality=a.modell, start_s=a.start, end_s=a.ende,
              make_overlay=not a.ohne_video, progress=progress)
    print(file=sys.stderr)
    for w in out.result.warnings:
        print(f"Hinweis: {w}")
    print(out.result.metrics_table()[["Kennzahl", "Seite", "Wert", "Einheit", "Bewertung"]].to_string(index=False))
    print(f"\nErgebnisse in: {out.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
