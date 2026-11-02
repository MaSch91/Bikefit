"""MediaPipe-Pose-Landmarken: Indizes, Namen und Skelettverbindungen."""

NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32  # Fußspitze ("foot index")

NUM_LANDMARKS = 33

# Paare (links, rechts) aller bilateralen Landmarken – für Seitentausch.
BILATERAL_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]
LEG_PAIRS = [(25, 26), (27, 28), (29, 30), (31, 32)]

SIDES = {
    "links": dict(shoulder=L_SHOULDER, elbow=L_ELBOW, wrist=L_WRIST, hip=L_HIP,
                  knee=L_KNEE, ankle=L_ANKLE, heel=L_HEEL, foot=L_FOOT),
    "rechts": dict(shoulder=R_SHOULDER, elbow=R_ELBOW, wrist=R_WRIST, hip=R_HIP,
                   knee=R_KNEE, ankle=R_ANKLE, heel=R_HEEL, foot=R_FOOT),
}

# Vereinfachtes Skelett (ohne Gesichts- und Fingerpunkte).
SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT), (L_ANKLE, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT), (R_ANKLE, R_FOOT),
]

DRAWN_POINTS = [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
                L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_FOOT, R_FOOT]

LEFT_IDS = {a for a, _ in BILATERAL_PAIRS}
RIGHT_IDS = {b for _, b in BILATERAL_PAIRS}
