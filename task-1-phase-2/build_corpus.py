"""
build_corpus.py
Member 1 deliverable — Protocol & Data Architect

Reuses the Phase 1 loader/cleaner (load_and_condition) on multiple DroneDetect
.dat files to build an expanded corpus of *_conditioned.npz files, then
catalogs them all into corpus_catalog.csv for Member 3's batch pipeline.

Edit DRONES_TO_PROCESS below to point at your actual downloaded .dat files.
"""

import os
import csv
import numpy as np
from scipy.ndimage import uniform_filter1d

FS = 60_000_000        # 60 million samples per second
FC = 2.4375e9           # center frequency 2.4375 GHz

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_DIR = os.path.join(OUT_DIR, "corpus")
os.makedirs(NPZ_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# EDIT THIS: list every clean-condition .dat file you want in the corpus.
# (path_to_dat_file, drone_model_name)
# ---------------------------------------------------------------------------
DRONES_TO_PROCESS = [
    ("AH Sample/AIR_0001_00.dat", "DJI_Mavic_Air"),
    ("AH Sample/MA1_0001_00.dat", "DJI_Mavic_Pro"),
    # ("AH Sample/PHA_0001_00.dat", "DJI_Phantom_4"),
    # ("AH Sample/INS_0001_00.dat", "DJI_Inspire_2"),
    # ("AH Sample/M2P_0001_00.dat", "DJI_Mavic_Pro_2"),
    # add more clean-condition DroneDetect files here to hit the >=6-10 pair target
]


# ---------------------------------------------------------------------------
# Phase 1 loader/cleaner logic, reused unchanged
# ---------------------------------------------------------------------------
def read_iq(path):
    raw = np.fromfile(path, dtype="float32")
    iq = raw.view(np.complex64)
    return iq


def normalize(iq):
    iq = (iq - iq.mean()) / np.sqrt(iq.var())
    return iq.astype(np.complex64)


def activity_mask(iq, fs, win_ms=0.2, k=5.0):
    win = max(1, int(win_ms / 1000 * fs))
    power = (iq.real**2 + iq.imag**2).astype(np.float32)
    env = uniform_filter1d(power, size=win)
    thr = k * np.median(env)
    mask = env > thr
    return mask, env, thr, win


def clean_mask(mask, win):
    smooth = uniform_filter1d(mask.astype(np.float32), size=win)
    return smooth > 0.5


def load_and_condition(path, fs, fc, drone, condition="clean"):
    iq = read_iq(path)
    iq = normalize(iq)
    mask, env, thr, win = activity_mask(iq, fs)
    mask = clean_mask(mask, win)
    meta = {
        "fs_hz": fs, "fc_hz": fc, "drone": drone,
        "condition": condition, "source_file": path,
    }
    return iq, mask, meta


# ---------------------------------------------------------------------------
# Step 1: process every file in DRONES_TO_PROCESS, save conditioned .npz
# ---------------------------------------------------------------------------
def expand_corpus():
    saved = []
    for path, drone_name in DRONES_TO_PROCESS:
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found — check the path before re-running")
            continue

        print(f"Processing {drone_name} ({path}) ...")
        iq, mask, meta = load_and_condition(path, FS, FC, drone_name)

        out_name = f"drone{drone_name}_conditioned.npz"
        out_path = os.path.join(NPZ_DIR, out_name)
        np.savez(out_path, iq=iq, mask=mask, meta=np.array([meta], dtype=object))

        talking_pct = 100 * mask.mean()
        print(f"  saved {out_path} | talking {talking_pct:.1f}%")
        saved.append((out_path, meta, iq, mask))

    return saved


# ---------------------------------------------------------------------------
# Step 2: build corpus_catalog.csv from the saved files
# ---------------------------------------------------------------------------
def build_catalog(saved):
    catalog_path = os.path.join(OUT_DIR, "corpus_catalog.csv")
    rows = []
    for out_path, meta, iq, mask in saved:
        rows.append({
            "drone_model": meta["drone"],
            "condition": meta["condition"],
            "source_file": meta["source_file"],
            "conditioned_npz_path": out_path,
            "fs_hz": meta["fs_hz"],
            "fc_hz": meta["fc_hz"],
            "duration_s": len(iq) / meta["fs_hz"],
            "activity_fraction": float(mask.mean()),
        })

    if not rows:
        print("No files processed — corpus_catalog.csv not written.")
        return

    with open(catalog_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\ncorpus_catalog.csv written with {len(rows)} entries -> {catalog_path}")


if __name__ == "__main__":
    saved = expand_corpus()
    build_catalog(saved)

    n = len(saved)
    target = 6
    if n < target:
        print(f"\n[NOTE] Corpus has {n} drone recording(s); team's diversity "
              f"target is >= {target} independent pairs. Add more .dat files to "
              f"DRONES_TO_PROCESS and re-run before handing off to Member 3.")
    else:
        print(f"\nCorpus size ({n}) meets the diversity target.")
