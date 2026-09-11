#!/usr/bin/env python3
"""
Member C — Synthesize & Package
================================
Implements Stage C (steps 9-13) of the team plan, consuming Member B's
swarm_combined.npz (combined_iq, slot_map, gap_list, meta) and producing the
final labeled training dataset + data card + validation report.

Usage
-----
    python3 01_member_C_pipeline.py --input swarm_combined.npz --outdir dataset_out

    # "hard" variant (extra background noise floor, per plan item 11)
    python3 01_member_C_pipeline.py --input swarm_combined.npz --outdir dataset_out_hard --variant hard

Point --input at your REAL swarm_combined.npz once you have it. Nothing else
needs to change -- this script only assumes the Part-4 contract shapes:
    combined_iq : complex array
    slot_map    : list of {start_sample, end_sample, owner: 'D1'|'D2'}
    gap_list    : list of {start_sample, end_sample, duration_s}
    meta        : {fs_hz, fc_hz, tdma_slot_s, drones:[...]}
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 9 — fake inter-drone packets (QPSK, root-raised-cosine shaped)
# ---------------------------------------------------------------------------
def rrc_taps(beta, sps, span_symbols):
    """Root-raised-cosine filter taps. Deliberately narrowband + structured
    so it's spectrally distinguishable from the broadband drone bursts
    (plan item 9: 'make them distinguishable... so a model has something
    learnable')."""
    N = span_symbols * sps
    t = (np.arange(-N / 2, N / 2 + 1)) / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0.0:
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta != 0 and abs(ti) == 1 / (4 * beta):
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    return (taps / np.sqrt(np.sum(taps ** 2))).astype(np.float32)


def make_qpsk_packet(duration_s, fs, sps, beta, rng):
    """Random-bit QPSK burst, RRC pulse-shaped, unit average power."""
    num_symbols = max(4, int(duration_s * fs / sps))
    bits = rng.integers(0, 4, size=num_symbols)
    const = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    symbols = const[bits]

    upsampled = np.zeros(num_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    taps = rrc_taps(beta, sps, span_symbols=8)
    pulse = np.convolve(upsampled, taps, mode="same").astype(np.complex64)

    power = np.mean(np.abs(pulse) ** 2)
    if power > 0:
        pulse = pulse / np.sqrt(power)
    return pulse


# ---------------------------------------------------------------------------
# Steps 10-11 — insert into SOME gaps at a target SNR, optional noise floor
# ---------------------------------------------------------------------------
def local_noise_floor(iq, s, e):
    seg = iq[max(0, s - 2000):s] if s > 0 else iq[e:e + 2000]
    if len(seg) == 0:
        seg = iq[s:e]
    return float(np.mean(np.abs(seg) ** 2)) + 1e-12


def synthesize_and_insert(combined_iq, gap_list, fs, cfg, rng):
    iq = combined_iq.copy()
    insertions = []  # ground truth log (step 10)

    for gi, gap in enumerate(gap_list):
        s, e = int(gap["start_sample"]), int(gap["end_sample"])
        gap_len = e - s
        margin = int(0.1e-3 * fs)  # keep clear of gap edges
        usable = gap_len - 2 * margin

        will_insert = rng.random() < cfg["insert_prob"] and usable > int(0.2e-3 * fs)

        record = {"gap_id": gi, "gap_start": s, "gap_end": e,
                  "gap_duration_s": gap["duration_s"], "inserted": False}

        if will_insert:
            max_dur = min(cfg["packet_max_ms"] * 1e-3, usable / fs)
            min_dur = min(cfg["packet_min_ms"] * 1e-3, max_dur)
            dur = rng.uniform(min_dur, max_dur) if max_dur > min_dur else max_dur

            packet = make_qpsk_packet(dur, fs, cfg["sps"], cfg["rrc_beta"], rng)
            plen = min(len(packet), usable)
            packet = packet[:plen]

            offset = s + margin + rng.integers(0, max(1, usable - plen + 1))

            snr_db = rng.uniform(cfg["snr_min_db"], cfg["snr_max_db"])
            noise_p = local_noise_floor(iq, s, e)
            target_sig_p = noise_p * (10 ** (snr_db / 10))
            packet = packet * np.sqrt(target_sig_p)

            iq[offset:offset + plen] += packet.astype(np.complex64)

            record.update({"inserted": True, "pkt_start": int(offset),
                            "pkt_end": int(offset + plen),
                            "pkt_duration_s": plen / fs,
                            "snr_db": float(snr_db)})
        insertions.append(record)

    # Step 11 — optional realism: faint background noise floor (toggle)
    if cfg["variant"] == "hard":
        sig_p = float(np.mean(np.abs(iq) ** 2))
        floor_p = sig_p * cfg["hard_noise_frac"]
        noise = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))).astype(np.complex64)
        noise *= np.sqrt(floor_p / 2)
        iq = iq + noise

    return iq, insertions


# ---------------------------------------------------------------------------
# Step 12 — cut & label
# ---------------------------------------------------------------------------
def build_activity_tracks(N, slot_map, gap_list):
    """Rebuild per-drone activity purely from slot_map + gap_list (the only
    things Member C receives per the Part-4 contract -- no raw masks)."""
    owner_track = np.full(N, "", dtype=object)
    for slot in slot_map:
        s, e = int(slot["start_sample"]), int(slot["end_sample"])
        owner_track[s:e] = slot["owner"]

    silent = np.zeros(N, dtype=bool)
    for gap in gap_list:
        s, e = int(gap["start_sample"]), int(gap["end_sample"])
        silent[s:e] = True

    active = {}
    for d in ("D1", "D2"):
        active[d] = (owner_track == d) & (~silent)
    return active


def cut_and_label(iq, insertions, slot_map, gap_list, fs, drones, example_len_s, outdir):
    N = len(iq)
    ex_len = int(example_len_s * fs)
    active = build_activity_tracks(N, slot_map, gap_list)

    iq_dir = os.path.join(outdir, "iq")
    os.makedirs(iq_dir, exist_ok=True)

    # flatten insertion log to easily test window overlap
    packets = [r for r in insertions if r["inserted"]]

    rows = []
    idx = 0
    for start in range(0, N - ex_len + 1, ex_len):
        end = start + ex_len
        idx += 1
        ex_id = f"example_{idx:05d}"

        drones_active = [d for d, mask in active.items() if mask[start:end].any()]
        drone_names_active = []
        for d in drones_active:
            drone_names_active.append(drones[0] if d == "D1" else drones[1])

        present = False
        snr_db = np.nan
        rel_start = np.nan
        rel_end = np.nan
        for p in packets:
            ov_s, ov_e = max(start, p["pkt_start"]), min(end, p["pkt_end"])
            if ov_e > ov_s:
                present = True
                snr_db = p["snr_db"]
                rel_start = (max(start, p["pkt_start"]) - start) / fs
                rel_end = (min(end, p["pkt_end"]) - start) / fs
                break

        np.save(os.path.join(iq_dir, ex_id + ".npy"), iq[start:end].astype(np.complex64))

        rows.append({
            "id": ex_id,
            "start_sample": start,
            "end_sample": end,
            "duration_s": example_len_s,
            "drones_active": ";".join(drone_names_active) if drone_names_active else "none",
            "num_drones_active": len(drone_names_active),
            "interdrone_present": present,
            "interdrone_snr_db": snr_db,
            "interdrone_rel_start_s": rel_start,
            "interdrone_rel_end_s": rel_end,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 13 — validate: spectrogram checks, label balance, baseline classifier
# ---------------------------------------------------------------------------
def validate(labels_df, iq_dir, fs, outdir, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.signal as sig
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    report_lines = []
    report_lines.append(f"# Member C — Validation Report\n")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}Z\n")

    # --- label balance ---
    n_total = len(labels_df)
    n_present = int(labels_df["interdrone_present"].sum())
    n_absent = n_total - n_present
    balance = labels_df["drones_active"].value_counts()

    report_lines.append("## Label balance\n")
    report_lines.append(f"- Total examples: **{n_total}**")
    report_lines.append(f"- `interdrone_present = True`: **{n_present}** ({100*n_present/n_total:.1f}%)")
    report_lines.append(f"- `interdrone_present = False`: **{n_absent}** ({100*n_absent/n_total:.1f}%)")
    report_lines.append("\n`drones_active` distribution:\n")
    for k, v in balance.items():
        report_lines.append(f"- `{k}`: {v}")
    report_lines.append("")

    # --- spectrogram spot-checks: one present, one absent ---
    present_ids = labels_df[labels_df["interdrone_present"]]["id"].tolist()
    absent_ids = labels_df[~labels_df["interdrone_present"]]["id"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ids, title in [(axes[0], present_ids, "interdrone_present = True"),
                            (axes[1], absent_ids, "interdrone_present = False")]:
        if ids:
            pick = ids[len(ids) // 2]
            arr = np.load(os.path.join(iq_dir, pick + ".npy"))
            f, t, Sxx = sig.spectrogram(arr, fs=fs, nperseg=min(256, len(arr)), return_onesided=False)
            ax.pcolormesh(t * 1e3, np.fft.fftshift(f) / 1e6,
                          10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12), shading="auto")
            ax.set_title(f"{title}\n({pick})")
            ax.set_xlabel("ms"); ax.set_ylabel("MHz")
        else:
            ax.set_title(f"{title}\n(no examples)")
    plt.tight_layout()
    spec_path = os.path.join(outdir, "validation_spectrograms.png")
    plt.savefig(spec_path, dpi=110)
    plt.close()
    report_lines.append("## Spectrogram spot-check\n")
    report_lines.append(f"See `validation_spectrograms.png` — one `present` and one `absent` "
                         f"example, side by side. The `present` example should show a narrower, "
                         f"more structured band (the synthetic QPSK packet) against the broadband "
                         f"drone bursts.\n")

    # --- quick baseline classifier ---
    feats, labels = [], []
    for _, row in labels_df.iterrows():
        arr = np.load(os.path.join(iq_dir, row["id"] + ".npy"))
        p = np.abs(arr) ** 2
        f, Pxx = sig.welch(arr, fs=fs, nperseg=min(256, len(arr)), return_onesided=False)
        Pxx_n = Pxx / (Pxx.sum() + 1e-15)
        spectral_flatness = np.exp(np.mean(np.log(Pxx_n + 1e-15))) / (np.mean(Pxx_n) + 1e-15)
        occupancy = float(np.mean(p > 3 * np.median(p)))
        papr_db = 10 * np.log10((p.max() + 1e-15) / (p.mean() + 1e-15))
        feats.append([10 * np.log10(p.mean() + 1e-15), p.std(), spectral_flatness, occupancy, papr_db])
        labels.append(int(row["interdrone_present"]))

    X = np.array(feats)
    y = np.array(labels)
    report_lines.append("## Baseline classifier (sanity check, not a real model)\n")
    if len(set(y)) < 2 or n_total < 10:
        report_lines.append("_Not enough examples / not enough class variety in this demo run "
                             "to train a meaningful baseline classifier. Re-run on the full "
                             "real dataset for this check._\n")
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y if min(np.bincount(y)) > 1 else None)
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        cm = confusion_matrix(yte, clf.predict(Xte))
        report_lines.append(f"- Features: mean power (dB), power std, spectral flatness, "
                             f"occupancy fraction, peak-to-average power ratio (dB)")
        report_lines.append(f"- Test accuracy: **{acc*100:.1f}%** (n_test={len(yte)})")
        report_lines.append(f"- Confusion matrix: {cm.tolist()}")
        report_lines.append("")
        if acc > 0.98:
            report_lines.append("⚠️ Accuracy is suspiciously close to 100% — check for a label "
                                 "leak (e.g. inserted packets too clean / too easy to spot). "
                                 "Plan item 4 recommends adding AWGN + a wider SNR sweep.")
        elif acc < 0.55:
            report_lines.append("⚠️ Accuracy is near chance — check for a labeling bug before "
                                 "trusting this dataset.")
        else:
            report_lines.append("This is in a healthy range: learnable but not trivial.")

    report_path = os.path.join(outdir, "validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(l) for l in report_lines))
    return report_path, spec_path


# ---------------------------------------------------------------------------
# Data card (plan item 12)
# ---------------------------------------------------------------------------
def write_data_card(outdir, meta, cfg, labels_df, insertions_df, source_path):
    n_total = len(labels_df)
    n_present = int(labels_df["interdrone_present"].sum())
    card = f"""# DATA CARD — Synthetic Drone Swarm RF Dataset

## ⚠️ This is a SIMULATED swarm, not a real capture.
Two single-drone recordings (DroneDetect, clean condition) were combined with
strict time-division multiple access (TDMA) turn-taking, and short synthetic
QPSK "inter-drone message" bursts were inserted into a subset of the resulting
silent gaps. No real multi-drone RF capture, and no real inter-drone protocol,
was used or modeled with any fidelity. See Part 6, item 8 of the team plan:
"'Strict TDMA' is a simplification."

## Source
- Built from: `{source_path}`
- Base recordings: {meta.get('drones')}
- Sample rate: {meta.get('fs_hz'):,} Hz
- Center frequency: {meta.get('fc_hz')/1e9:.4f} GHz
- TDMA slot length: {meta.get('tdma_slot_s')*1000:.2f} ms

## Generation process (Stage C, plan Part 5)
1. Synthetic inter-drone packets: random-bit QPSK, root-raised-cosine pulse
   shaped (β={cfg['rrc_beta']}, {cfg['sps']} samples/symbol), duration drawn
   uniformly in [{cfg['packet_min_ms']}, {cfg['packet_max_ms']}] ms.
2. Insertion: each silent gap independently receives a packet with
   probability {cfg['insert_prob']}, at a random offset (with edge margin) and
   a target SNR drawn uniformly from [{cfg['snr_min_db']}, {cfg['snr_max_db']}] dB
   relative to the local noise floor. Every insertion attempt is logged in
   `insertion_log.csv`, whether or not it happened.
3. Background noise floor: variant = **{cfg['variant']}**{' (extra AWGN added across the whole stream, fraction=' + str(cfg['hard_noise_frac']) + ')' if cfg['variant']=='hard' else ' (none added beyond the base recordings)'}.
4. Cutting: sliced into fixed-length, non-overlapping examples of
   {cfg['example_ms']} ms each.
5. Labeling: `drones_active` is reconstructed from `slot_map` + `gap_list`
   only (the actual data Member C receives, per the team's Part-4 contract —
   no raw per-drone activity masks are used at this stage).
   `interdrone_present` / `interdrone_snr_db` come directly from the
   insertion log (ground truth).

## Reproducibility
- Random seed: {cfg['seed']}
- All generation parameters are also saved in `metadata.json`.

## Files
```
dataset/
  iq/example_00001.npy ...      # complex64 IQ, shape (example_len_samples,)
  labels.csv                    # id, drones_active, interdrone_present, snr, timing
  insertion_log.csv             # every gap + whether/how a packet was inserted (ground truth)
  metadata.json                 # fs, fc, tdma config, seed, packet config
  validation_report.md          # label balance + baseline classifier sanity check
  validation_spectrograms.png   # present vs. absent spectrogram spot-check
  DATA_CARD.md                  # this file
```

## Dataset stats (this run)
- Examples: {n_total}
- `interdrone_present = True`: {n_present} ({100*n_present/max(n_total,1):.1f}%)
- Gaps considered: {len(insertions_df)}
- Gaps with a packet inserted: {int(insertions_df['inserted'].sum())}

## Intended use
Training/evaluating a classifier to (a) detect drone RF activity in a shared
channel and (b) detect the presence of a third, non-drone signal (the
synthetic "inter-drone message") in the silent gaps between drone bursts.
This is a methodology / pipeline demonstration dataset, not a substitute for
real multi-emitter RF captures — do not use it to make claims about real
drone-swarm coordination signals.

## Known limitations
- Strict TDMA (no overlap, no collisions) is a simplification of real
  multi-drone channel access.
- The synthetic inter-drone packet is a generic QPSK burst, not modeled on
  any real drone-to-drone protocol.
- `drones_active` labels are derived from the slot schedule and gap list,
  not from re-checking the raw waveform at label time — they inherit
  whatever accuracy Member A's activity detector had.
"""
    path = os.path.join(outdir, "DATA_CARD.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(card)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Member C — synthesize & package")
    ap.add_argument("--input", required=True, help="path to swarm_combined.npz")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--insert-prob", type=float, default=0.5)
    ap.add_argument("--snr-min", type=float, default=-3.0)
    ap.add_argument("--snr-max", type=float, default=15.0)
    ap.add_argument("--packet-min-ms", type=float, default=0.2)
    ap.add_argument("--packet-max-ms", type=float, default=1.0)
    ap.add_argument("--sps", type=int, default=20, help="samples per QPSK symbol")
    ap.add_argument("--rrc-beta", type=float, default=0.35)
    ap.add_argument("--example-ms", type=float, default=5.0)
    ap.add_argument("--variant", choices=["easy", "hard"], default="easy")
    ap.add_argument("--hard-noise-frac", type=float, default=0.15)
    args = ap.parse_args()

    cfg = {
        "seed": args.seed, "insert_prob": args.insert_prob,
        "snr_min_db": args.snr_min, "snr_max_db": args.snr_max,
        "packet_min_ms": args.packet_min_ms, "packet_max_ms": args.packet_max_ms,
        "sps": args.sps, "rrc_beta": args.rrc_beta,
        "example_ms": args.example_ms, "variant": args.variant,
        "hard_noise_frac": args.hard_noise_frac,
    }
    rng = np.random.default_rng(args.seed)

    d = np.load(args.input, allow_pickle=True)
    combined_iq = d["combined_iq"]
    slot_map = list(d["slot_map"])
    gap_list = list(d["gap_list"])
    meta = dict(d["meta"][0])
    fs = meta["fs_hz"]
    drones = meta["drones"]

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[1/5] Loaded {args.input}: {len(combined_iq):,} samples, "
          f"{len(slot_map)} slots, {len(gap_list)} gaps")

    print("[2/5] Synthesizing + inserting inter-drone packets...")
    iq_out, insertions = synthesize_and_insert(combined_iq, gap_list, fs, cfg, rng)
    insertions_df = pd.DataFrame(insertions)
    insertions_df.to_csv(os.path.join(args.outdir, "insertion_log.csv"), index=False, encoding="utf-8")
    n_ins = int(insertions_df["inserted"].sum())
    print(f"       {n_ins}/{len(gap_list)} gaps received a synthetic packet")

    print("[3/5] Cutting & labeling examples...")
    labels_df = cut_and_label(iq_out, insertions, slot_map, gap_list, fs, drones,
                               args.example_ms / 1000.0, args.outdir)
    labels_df.to_csv(os.path.join(args.outdir, "labels.csv"), index=False, encoding="utf-8")
    print(f"       {len(labels_df)} examples written to {args.outdir}/iq/")

    print("[4/5] Writing metadata + data card...")
    metadata_out = {
        "fs_hz": int(fs), "fc_hz": float(meta["fc_hz"]),
        "tdma_slot_s": float(meta["tdma_slot_s"]), "drones": list(drones),
        "example_len_s": args.example_ms / 1000.0, "n_examples": len(labels_df),
        "generation_config": cfg,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_swarm_file": os.path.abspath(args.input),
    }
    with open(os.path.join(args.outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2)
    write_data_card(args.outdir, meta, cfg, labels_df, insertions_df, args.input)

    print("[5/5] Validating (label balance, spectrogram check, baseline classifier)...")
    validate(labels_df, os.path.join(args.outdir, "iq"), fs, args.outdir, rng)

    print(f"\nDone. Deliverables in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
