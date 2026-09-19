# Phase 2 — fixes and cleanups by Member 3

Reviewed Member 1's and Member 2's Phase-2 code, fixed two real bugs, and tidied
the repo. All changes verified with the existing self-tests plus new stress checks.

## Bug 1 — decoder could not tell two message types apart (and crashed)
**Where:** `task-1-phase-2/message_schema.py` (`parse_on_air_bits` / `deserialize`)

BEACON and POSITION_REPORT produced the **identical on-air length** (1224-bit
body), so the decoder could not distinguish them by length. When it tried the
wrong interpretation, `deserialize` raised `IndexError` (message-type byte >= 5),
which `parse_on_air_bits` did not catch — so it crashed.

Measured on 5000 clean, error-free packets **before**: 2977 correct, 126
mis-typed, 1897 crashed/failed (~40% unusable). **After:** 5000/5000 correct.

**Fix:**
- POSITION_REPORT now uses **10 repeats (not 9)** so every message type's on-air
  length is unique (duration 0.232 ms, still inside the validated 0.2-1.0 ms band).
- `deserialize` raises a clean `ValueError` for an unknown type byte.
- `parse_on_air_bits` catches `(ValueError, IndexError)`.
- `PROTOCOL_SPEC.md` table updated to match.

## Bug 2 — SNR label did not reflect the real signal-to-noise ratio
**Where:** `task-2-phase-2/modulator_and_injector.py` and
`task-3/code+documentation/01_member_C_pipeline.py`

`local_noise_floor()` measured power in the 2000 samples **just before the gap**,
which is usually the drone's active burst — not noise. So the logged `snr_db` was
"SNR vs the neighbouring drone burst," not vs the noise floor. A packet logged at
14 dB came out at power ~51 while the true floor was ~0.056.

**Fix:** compute one consistent noise-floor power up front (a fixed fraction of the
active signal power) and scale every packet's SNR against that **same** floor that
gets added to the stream. Verified: reference floor now matches the measured
mid-gap power, so `snr_db` is meaningful.

## Cleanups
- **De-duplicated `message_schema.py`.** `task-2-phase-2/message_schema.py` is now a
  thin shim that re-exports the single canonical copy in `task-1-phase-2/`, so the
  two can no longer drift apart.
- **`insert_prob` aligned to 0.4** (matching `PROTOCOL_SPEC.md`) in both the injector
  default and the Member-C pipeline default + its README.
- **Double-noise guard.** The injector now has an `add_noise_floor` flag so it is not
  applied twice when Member 3's pipeline also adds a floor.
- **Repo tidy.** Team plans and diagrams moved into `docs/`; README expanded with the
  repo layout and roles.

## Still open (needs team decision)
From `PROTOCOL_SPEC.md` "Open Decisions": preamble length, whether to spread message
durations for graded difficulty, and CRC-8 sufficiency.
