# Member C — How to run this on your real data

**Heads up:** `swarm_combined.npz` (Member B's real output) wasn't uploaded,
so everything in this folder except `01_member_C_pipeline.py` was generated
from a small **stand-in** dataset built to the exact same schema — just to
prove the pipeline works end to end. Swap in your real file and rerun.

## 1. Run it

```bash
python3 01_member_C_pipeline.py \
    --input swarm_combined.npz \
    --outdir dataset_final \
    --seed 7
```

That's it — defaults match the plan's Stage C spec (steps 9–13). It will print
progress and, on completion, `dataset_final/` will contain everything listed
in Part 5, step 12 of the team plan:

```
dataset_final/
  iq/example_00001.npy ...      # complex64 IQ slices
  labels.csv                    # id, drones_active, interdrone_present, snr, timing
  insertion_log.csv             # ground truth: every gap + insertion decision
  metadata.json                 # fs, fc, tdma config, seed, packet config
  validation_report.md          # label balance + baseline classifier sanity check
  validation_spectrograms.png   # present vs. absent spectrogram spot-check
  DATA_CARD.md                  # full data card (simulated-swarm disclosure etc.)
```

## 2. Also build the "hard" variant (item 11 in the plan — optional)

```bash
python3 01_member_C_pipeline.py \
    --input swarm_combined.npz \
    --outdir dataset_final_hard \
    --seed 7 \
    --variant hard
```

Adds a faint background noise floor across the whole stream so "easy" vs.
"hard" dataset flavors both exist, per the plan.

## 3. Useful knobs

| Flag | Default | What it does |
|---|---|---|
| `--insert-prob` | 0.5 | fraction of silent gaps that get a synthetic packet (keep <1.0 — you need "absent" examples too, per plan item 10) |
| `--snr-min` / `--snr-max` | -3 / 15 dB | SNR sweep for inserted packets (plan item 4: avoid making them "too clean") |
| `--packet-min-ms` / `--packet-max-ms` | 0.2 / 1.0 | inter-drone packet duration range |
| `--example-ms` | 5.0 | length of each cut-and-labeled training example |
| `--sps`, `--rrc-beta` | 20, 0.35 | QPSK pulse-shaping params (higher `sps` = narrower packet bandwidth = easier for a model to distinguish from the broadband drone bursts, and vice versa) |

## 4. Check the validation report before trusting the dataset

`validation_report.md` runs a quick baseline classifier on simple power/
spectral features. Per plan item 13:
- **~100% accuracy instantly** → the synthetic signal is too easy to spot
  (a label leak) — widen the SNR range or lower `--insert-prob` variety.
- **~random (≈50%)** → something is likely mislabeled — check
  `insertion_log.csv` against `labels.csv` for a bug before using the data.
- Anything comfortably in between is the healthy zone.

## What was actually validated in this delivery

Since the real `swarm_combined.npz` wasn't available, `00_make_demo_stand_in.py`
was used to fabricate two short (0.3 s) fake "drone" recordings in Member A's
exact output format, ran **Member B's real, unmodified final code** from
`task2.md` on them to get a real-schema `swarm_combined_DEMO.npz` (17 silent
gaps found), then ran the full Member C pipeline on that. The included
`dataset_demo_output.zip`, `DATA_CARD.md`, `validation_report.md`, and
`validation_spectrograms.png` at the top level are the results of that dry
run — they exist to prove the code works, not as your real dataset.
`00_make_demo_stand_in.py` is not included since you won't need it once you
have the real file — the only script that matters going forward is
`01_member_C_pipeline.py`.