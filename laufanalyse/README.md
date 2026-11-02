# Laufanalyse

Videobasierte Bewegungsanalyse für Läufer: Ein Video (Laufband oder draußen, von der Seite,
von vorne oder von hinten) wird Frame für Frame ausgewertet. Ein Skelett mit Gelenkpunkten
wird eingezeichnet, und daraus werden Laufkennzahlen berechnet.

## Schnellstart

```bash
cd laufanalyse
pip install -r requirements.txt
streamlit run app.py
```

Danach öffnet sich die Weboberfläche im Browser (http://localhost:8501). Dort kannst du ein
Video hochladen, Perspektive, Laufband/draußen, Tempo und Körpergröße angeben und die Analyse
starten. Mit **„Demo“** kannst du die Oberfläche ohne eigenes Video ausprobieren.

Beim ersten Start wird das Posemodell (MediaPipe, ca. 6–30 MB) automatisch nach `models/`
geladen. Unter Linux ohne Desktop wird zusätzlich `libegl1` benötigt (`apt install libegl1 libgles2`).

Kommandozeile (ohne Oberfläche):

```bash
python -m laufanalyse.cli mein_video.mp4 --ansicht seite --tempo 12 --groesse 180
```

## Was wird gemessen?

| Perspektive | Kennzahlen |
|---|---|
| alle | Schrittfrequenz (Kadenz), Schrittlänge & Doppelschrittlänge (mit Tempo-Angabe), Bodenkontakte |
| Seite | Kniebeugung beim Aufsetzen / max. in Stütz- und Schwungphase, Schienbeinwinkel beim Aufsetzen, Fußaufsatz vor der Hüfte (Overstriding), Fußaufsatzmuster (Ferse/Mittelfuß/Vorfuß), Hüftstreckung beim Abdruck, Rumpfvorlage, vertikale Oszillation, Bodenkontaktzeit, Duty Factor, Ellbogenwinkel, Armschwung, Seitensymmetrie |
| Vorne/Hinten | Beckenabkippen (Hip Drop), Knie-Einwärtsknick (Valgus, FPPA), Spurbreite/Überkreuzen, Hand über Körpermitte, seitliche Rumpfbewegung |

Dazu Laufzyklus-Diagramme (Mittelwert ± Streuung über alle Schritte, links/rechts),
Schlüsselmomente (Aufsetzen, Stützphase, Abdruck, Schwungphase), ein Einzelbild-Explorer,
ein Overlay-Video und CSV-Exporte.

## Wie es funktioniert

1. **Posenerkennung** (`laufanalyse/pose.py`): MediaPipe Pose Landmarker, 33 Körperpunkte pro Frame inkl. Ferse und Fußspitze.
2. **Aufbereitung** (`signals.py`): Links/Rechts-Verwechslungen korrigieren, kurze Lücken interpolieren, phasenfreier Butterworth-Tiefpass (6 Hz).
3. **Winkel** (`kinematics.py`): 2D-Gelenkwinkel in der Bildebene; Laufrichtung wird automatisch erkannt.
4. **Schritterkennung** (`gait.py`): Bodenkontakt = Fuß nahe der (gleitend geschätzten) Bodenlinie; Kadenz zusätzlich per Frequenzanalyse gegengeprüft.
5. **Kennzahlen** (`metrics.py`): inkl. Orientierungsbereichen für Freizeitläufer.
6. **Overlay** (`overlay.py`): Skelett (blau = links, orange = rechts), Kniewinkel, grüne Kreise bei Bodenkontakt.

## Tipps für gute Aufnahmen

- Kamera **fest** (Stativ), auf Hüfthöhe, **senkrecht zur Laufrichtung** (Seite) bzw. mittig dahinter/davor.
- Ganze Person inkl. Füße im Bild, enganliegende Kleidung, gutes Licht, ruhiger Hintergrund.
- **60 fps oder Zeitlupe (120/240 fps)** – bei 30 fps sind Kontaktzeiten und Aufsatzwinkel nur grob.
- Mindestens 10 Schritte. Laufband: Geschwindigkeit notieren; draußen: Körpergröße + Tempo angeben.

## Grenzen

- **2D-Analyse** mit einer Kamera: Winkel stimmen nur bei passender Kameraposition; schräge Aufnahmen verfälschen sie.
- In der Seitenansicht ist das **kameraferne Bein** oft verdeckt und ungenauer.
- **cm-Werte** basieren auf der Körpergröße (Segmentlängen nach Winter) – Genauigkeit ca. ±10 %.
- Die Orientierungswerte sind grobe Richtwerte und **keine medizinische Diagnose**.

## Tests

```bash
python -m pytest
```

Die Tests nutzen einen synthetischen Läufer mit bekannter Kadenz und bekannten Winkeln (`laufanalyse/synthetic.py`).
