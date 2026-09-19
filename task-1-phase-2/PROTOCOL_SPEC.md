# Inter-Drone Communication Protocol Spec (v2 — grounded in real pipeline data)

**Status:** Draft for team review — finalize together before freezing.
**Scope:** This protocol is entirely synthetic, designed for this project. It does not
represent any real, standardized, or manufacturer-specific drone-to-drone protocol.

## Why this version differs from the first draft

The original message sizes were sized against theoretical constraints ("does it fit in
the gap"). Once checked against the *actual, already-validated* Task 3 pipeline
parameters — 20 samples/symbol at 60 MS/s (3 Msym/s effective rate), QPSK, packet
durations drawn from [0.2, 1.0] ms — real message content turned out to be far too
short (0.01-0.03 ms) to match. Dropping messages in at that size would silently change
the detection difficulty the team already tuned and validated (AUC=0.68,
accuracy=0.748 on `interdrone_present`, per `readiness_report.md`).

This version adds a preamble and payload repetition so every message type lands back
inside the existing, validated 0.2-1.0 ms envelope — Member 2's job becomes swapping
the *content* of the packets for real messages, not re-tuning the whole experiment.

## Transport Layer

Unacknowledged datagram (UDP-like), connectionless. No handshake, no session state,
no retransmission — real drone links are fast-moving and packet loss is tolerable.

## On-Air Packet Structure

```
[ PREAMBLE (16 symbols, fixed sync sequence) ][ MESSAGE, repeated N times ]
```

- **Preamble:** a fixed 16-symbol sequence, identical across all messages. Gives
  Member 2's synthesis code a realistic sync/detection anchor (real RF bursts almost
  always start this way) and adds a small, constant duration floor.
- **Repetition (N):** the serialized message (header + payload + CRC) is repeated
  N times back-to-back, where N is chosen per message type so the total on-air
  duration lands inside the pipeline's existing [0.2, 1.0] ms target band. This is a
  simple redundancy/spreading scheme — realistic for a noisy, low-power link, and
  gives Member 3's validation something concrete to check (does recovered content
  match across repeats?).

## Message Format (per repeat)

| Field     | Type          | Notes                                                       |
|-----------|---------------|----------------------------------------------------------------|
| msg_type  | enum          | BEACON, ACK, POSITION_REPORT, COLLISION_WARNING, TASK_ASSIGN   |
| src_id    | int (1 byte)  | 1 = Drone 1, 2 = Drone 2                                        |
| dst_id    | int (1 byte)  | 0 = broadcast, or a specific drone id                           |
| seq_no    | int (1 byte)  | increments per message, wraps at 256                            |
| payload   | bytes         | size depends on msg_type                                        |
| checksum  | uint8 (CRC-8) | over header + payload                                            |

## Message Types, Sizes, and Repeat Counts

Computed against the pipeline's actual symbol rate (3 Msym/s = 20 samples/symbol
@ 60 MS/s), QPSK (2 bits/symbol), 16-symbol preamble:

| msg_type          | Payload | Wire bytes | Base symbols | Repeats (N) | Final duration |
|-------------------|---------|------------|---------------|--------------|------------------|
| BEACON            | 4 B     | 9 B        | 36            | 17           | ~0.209 ms        |
| ACK               | 2 B     | 7 B        | 28            | 21           | ~0.201 ms        |
| POSITION_REPORT   | 12 B    | 17 B       | 68            | 10           | ~0.232 ms        |
| COLLISION_WARNING | 8 B     | 13 B       | 52            | 12           | ~0.213 ms        |
| TASK_ASSIGN       | 20 B    | 25 B       | 100           | 6            | ~0.205 ms        |

All five types land in a tight band (~0.20-0.23 ms) — comfortably inside the
pipeline's validated [0.2, 1.0] ms envelope, near the short end, leaving headroom for
longer variants (e.g. TASK_ASSIGN with a larger payload) without exceeding 1.0 ms.

**Note:** POSITION_REPORT uses 10 repeats (not 9). With 9 repeats its on-air body was
1224 bits — identical to BEACON's — so a decoder could not tell the two types apart by
length. Ten repeats makes every message type's on-air length unique.

## Justification

- **Preamble + repetition, not padding with junk bits:** repeating the real message is
  both more realistic (matches how low-power spread links actually improve
  reliability) and directly useful for validation — Member 3 can check that the
  decoded repeats agree, which junk padding could not offer.
- **Duration matched to the existing pipeline, not re-derived from scratch:** keeps
  the whole experiment's calibration (SNR sweep, insert probability, gap-fit logic)
  valid without rerunning Member 3's validation from zero.
- **All 5 types cluster near 0.2 ms** rather than spreading across the full 0.2-1.0 ms
  range: intentional for v1 — keeps every message type's difficulty roughly
  comparable. A future iteration could deliberately vary repeat count further to make
  larger messages (e.g. TASK_ASSIGN) harder to detect than small ones (e.g. BEACON),
  if the team wants message-type detection to be a graded-difficulty task.

## Explicitly Out of Scope

- Any task-specific labels (e.g., friendly/foe, threat level) — reserved for a future
  phase, not built here.
- Acknowledgement/retry logic — modeled as unacknowledged/best-effort.
- Forward error correction beyond simple repetition — no interleaving, no Reed-Solomon,
  etc. Repetition alone is the redundancy mechanism for this version.

## Open Decisions For Team Review

1. Confirm 16-symbol preamble length is appropriate, or adjust.
2. Confirm whether all message types should cluster at ~0.2ms (simple, v1) or be
   deliberately spread across the full 0.2-1.0ms range for graded difficulty (v2 idea).
3. Confirm CRC-8 is sufficient, or decide on an alternative.
4. Confirm this preserves compatibility with Member 3's existing SNR sweep
   ([-3, 15] dB) and insert probability (0.4) without needing to re-validate — should
   be true since duration, not content, was the variable that mattered to the
   pipeline's noise/detection model.
