import subprocess

import numpy as np
import pytest

from laufanalyse import gait
from laufanalyse.kinematics import facing_direction, included_angle, sagittal_angles
from laufanalyse.landmarks import L_ANKLE, L_HIP, L_KNEE, R_ANKLE, R_HIP, R_KNEE
from laufanalyse.metrics import Settings, analyze
from laufanalyse.signals import (clean_keypoints, fix_frontal_sides, fix_leg_swaps,
                                 interpolate_gaps, lowpass)

from laufanalyse.synthetic import frontal_runner, side_runner


# ----------------------------------------------------------------------------- Kinematik

def test_included_angle_basic():
    a = np.array([[0.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert included_angle(a, b, np.array([[1.0, 1.0]]))[0] == pytest.approx(90)
    assert included_angle(a, b, np.array([[0.0, 2.0]]))[0] == pytest.approx(180)


@pytest.mark.parametrize("d", [1, -1])
def test_sagittal_angles_recover_truth(d):
    kp, _, truth = side_runner(d=d)
    assert facing_direction(kp) == d
    ang = sagittal_angles(kp, d)
    for side in ("links", "rechts"):
        np.testing.assert_allclose(ang[side]["knie"], truth[side]["knie"], atol=1e-6)
        np.testing.assert_allclose(ang[side]["oberschenkel"], truth[side]["oberschenkel"], atol=1e-6)
    np.testing.assert_allclose(ang["rumpf"], truth["rumpf"], atol=1e-6)


# ----------------------------------------------------------------------------- Signale

def test_interpolate_short_gap_only():
    x = np.arange(20, dtype=float)
    x[5:7] = np.nan     # kurze Lücke
    x[10:18] = np.nan   # lange Lücke
    y = interpolate_gaps(x, max_gap=3)
    assert y[5] == pytest.approx(5) and y[6] == pytest.approx(6)
    assert np.isnan(y[12])


def test_lowpass_keeps_slow_signal_removes_noise():
    fps = 60
    t = np.arange(600) / fps
    slow = np.sin(2 * np.pi * 1.4 * t)
    noisy = slow + 0.3 * np.random.default_rng(0).standard_normal(len(t))
    out = lowpass(noisy, fps, 6)
    err = np.sqrt(np.mean((out - slow) ** 2))
    assert err < 0.15  # Rohrauschen 0.3 -> mindestens halbiert


def test_fix_leg_swaps_corrects_label_swap():
    kp, vis, _ = side_runner()
    bad = kp.copy()
    t = 100
    for a, b in ((L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE), (29, 30), (31, 32)):
        bad[t:, [a, b]] = bad[t:, [b, a]]  # ab Frame 100 dauerhaft vertauscht
    assert fix_leg_swaps(bad, vis.copy()) >= 1
    np.testing.assert_allclose(bad[t + 5, L_ANKLE], kp[t + 5, L_ANKLE])


def test_fix_frontal_sides():
    kp, vis = frontal_runner(view="hinten")
    wrong = kp.copy()
    wrong[:, [L_HIP, R_HIP]] = wrong[:, [R_HIP, L_HIP]]
    assert fix_frontal_sides(wrong, vis.copy(), "hinten") == len(kp)
    np.testing.assert_allclose(wrong[:, L_HIP], kp[:, L_HIP])


# ----------------------------------------------------------------------------- Schritterkennung

@pytest.mark.parametrize("cad", [150.0, 170.0, 185.0])
def test_cadence_detection(cad):
    kp, vis, _ = side_runner(cadence=cad)
    res = analyze(kp, vis, 60.0, Settings(view="seite"))
    assert res.get("kadenz").value == pytest.approx(cad, rel=0.03)
    assert res.extra["kadenz_frequenz"] == pytest.approx(cad, rel=0.03)
    assert {c.side for c in res.contacts} == {"links", "rechts"}


def test_normalize_cycles_shape():
    sig = np.sin(np.linspace(0, 10 * np.pi, 500))
    m, to = gait.normalize_cycles(sig, [(0, 30, 100), (100, 130, 200)])
    assert m.shape == (2, 101) and len(to) == 2


# ----------------------------------------------------------------------------- Kennzahlen

def test_step_length_from_treadmill_speed():
    kp, vis, _ = side_runner(cadence=170)
    res = analyze(kp, vis, 60.0, Settings(view="seite", setting="laufband", speed_kmh=12.0))
    expected_cm = 12 / 3.6 / (170 / 60) * 100
    assert res.get("schrittlaenge").value == pytest.approx(expected_cm, rel=0.04)


def test_side_metrics_plausible():
    kp, vis, _ = side_runner(cadence=170, trunk_deg=6)
    res = analyze(kp, vis, 60.0, Settings(view="seite", height_cm=180))
    assert res.get("rumpf").value == pytest.approx(6, abs=0.5)
    assert res.get("knie_aufsatz", "links").value == pytest.approx(20, abs=8)
    assert res.get("knie_schwung_max", "links").value == pytest.approx(100, abs=5)
    assert res.get("vert_osz").unit == "cm"
    assert len(res.keyframes) == 4


def test_frontal_pelvic_drop():
    kp, vis = frontal_runner(drop_deg=6.0, view="hinten")
    kp_c, vis_c, _ = clean_keypoints(kp, vis, 60.0, "hinten")
    res = analyze(kp_c, vis_c, 60.0, Settings(view="hinten"))
    assert res.get("beckenabkippen", "links").value == pytest.approx(6, abs=1.5)
    assert res.get("kadenz").value == pytest.approx(170, rel=0.03)


# ----------------------------------------------------------------------------- Overlay

def test_overlay_render(tmp_path):
    from laufanalyse.overlay import render_overlay
    kp, vis, _ = side_runner(seconds=2)
    video = tmp_path / "blank.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=gray:s=1000x800:r=60:d=2", "-pix_fmt", "yuv420p", str(video)], check=True)
    res = analyze(kp, vis, 60.0, Settings(view="seite"))
    out = render_overlay(str(video), np.arange(len(kp)), kp, tmp_path / "o.mp4", 60.0, res)
    assert out.exists() and out.stat().st_size > 1000
