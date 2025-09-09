BASE_COORDS = {
    "BP":    (1655, 118, 207, 47),
    "HR":    (1639,  70,  69, 48),
    "CVP":   (1639, 171,  61, 56),
    "SpO2":  (1639, 275,  58, 51),
    "Tskin": (1768, 292,  70, 42),
    "Trect": (1864, 296,  58, 38),
    "BSR1":  (1767, 337,  42, 34),
    "BSR2":  (1861, 341,  41, 34),
    "etCO2": (1647, 421,  44, 45),
    "aRR":   (1862, 431,  39, 35),
    # 左上（人工呼吸器画面）系
    "Ppeak": ( 710,  86,  72, 48),
    "Pmean": ( 856,  78,  49, 27),
    "aPEEP": ( 859, 127,  48, 26),
    "I_E":   ( 842, 178,  90, 34),
    "VTi":   ( 860, 279,  45, 35),
    "VTe":   ( 864, 329,  49, 31),
    "FIO2":  ( 162, 504,  45, 35),
    "sPEEP": ( 240, 506,  56, 32),
    "sRR":   ( 318, 508,  52, 32),
    "TV":    ( 397, 506,  50, 33),
    "Ti":    ( 630, 506,  61, 32),
}

BED_WIDTH = 1920
BED_HEIGHT = 1080

def _shift(coord, dx, dy):
    x, y, w, h = coord
    return (x + dx, y + dy, w, h)

def _build_bed(dx, dy):
    b = {k: _shift(v, dx, dy) for k, v in BASE_COORDS.items()}
    return {
        "BP_COMBINED_COORD": b["BP"],
        "CVP_COORDS": b["CVP"],
        "vital_crop": {
            "HR": b["HR"],
            "SpO2": b["SpO2"],
            "Tskin": b["Tskin"],
            "Trect": b["Trect"],
            "BSR1": b["BSR1"],
            "BSR2": b["BSR2"],
            "etCO2": b["etCO2"],
            "RR": b["aRR"],
            "Ppeak": b["Ppeak"],
            "Pmean": b["Pmean"],
            "PEEPact": b["aPEEP"],
            "I_E": b["I_E"],
            "VTi": b["VTi"],
            "VTe": b["VTe"],
            "FiO2": b["FIO2"],
            "PEEPset": b["sPEEP"],
            "RRact": b["sRR"],
            "VTset": b["TV"],
            "Ti": b["Ti"],
        },
    }

BED_COORDS_4 = {
    1: _build_bed(0, 0),
    2: _build_bed(BED_WIDTH, 0),
    3: _build_bed(0, BED_HEIGHT),
    4: _build_bed(BED_WIDTH, BED_HEIGHT),
}
