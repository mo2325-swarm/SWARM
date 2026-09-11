## Task 2 Summary — Mixing the Two Drones into a Fake Swarm

**Goal:** Take the two separate drone recordings from Task 1 and combine them into one fake "swarm" stream where they alternate turns (TDMA), then find the silent gaps for Task 3 to fill with fake inter-drone messages.

**Inputs used:**
- `droneAIR_conditioned.npz` (DJI Mavic Air)
- `droneMA1_conditioned.npz` (DJI Mavic Pro)

**What was done:**
1. Loaded both `.npz` files and confirmed they share the same sample rate (60 MS/s) and center frequency (2.4375 GHz)
2. Trimmed both signals to equal length
3. Set a fixed TDMA slot length of **2 ms** (an earlier auto-calculated version picked up noisy flickers instead of real bursts, producing slots that were far too short — fixed by hardcoding 2 ms)
4. Built an alternating slot map: Drone 1 owns slot 1, Drone 2 owns slot 2, repeating across the full 2-second recording
5. Gated each drone's signal to only play during its own slots (with a raised-cosine taper at slot edges to avoid clicks), then summed them into one clean combined stream — no overlap since slots never coincide
6. Detected silent gaps using each drone's **real activity mask** from Task 1 (not raw power) — found gaps where the slot's rightful owner wasn't actually transmitting
7. Verified visually with a spectrogram, a full-length power-vs-owner plot, and a zoomed-in 20ms plot that clearly showed correct alternation

**Result:**
- 249 usable silent gaps identified, 2–26 ms long, totaling 1.1 seconds of silence out of the 2-second recording

**Output saved:** `swarm_combined.npz`, containing:
- `combined_iq` — the merged fake-swarm signal
- `slot_map` — which drone owned each time slot
- `gap_list` — the 249 silent gaps, ready for Task 3
- `meta` — sample rate, center frequency, slot duration, drone names

**Status:** ✅ Task 2 complete, output ready to hand off to Member C for Task 3.

---

### Final working code

```python
import numpy as np

# ---- Load Task 1 outputs ----
d1 = np.load("droneAIR_conditioned.npz", allow_pickle=True)
d2 = np.load("droneMA1_conditioned.npz", allow_pickle=True)

iq1, mask1, meta1 = d1["iq"], d1["mask"], d1["meta"][0]
iq2, mask2, meta2 = d2["iq"], d2["mask"], d2["meta"][0]

# ---- Sanity check ----
assert meta1["fs_hz"] == meta2["fs_hz"], "Sample rates don't match!"
assert meta1["fc_hz"] == meta2["fc_hz"], "Center frequencies don't match!"
FS = meta1["fs_hz"]

# ---- Trim both to same length ----
N = min(len(iq1), len(iq2))
iq1, mask1 = iq1[:N], mask1[:N]
iq2, mask2 = iq2[:N], mask2[:N]
print(f"Using {N:,} samples ({N/FS:.2f} s) from each drone")

# ---- Set a realistic slot length (2 ms) ----
slot_len = int(0.002 * FS)
print(f"TDMA slot length: {slot_len} samples ({slot_len/FS*1000:.2f} ms)")

# ---- Build alternating slot map ----
slot_map = []
pos = 0
turn = 0  # 0 = D1, 1 = D2
while pos < N:
    end = min(pos + slot_len, N)
    owner = "D1" if turn == 0 else "D2"
    slot_map.append({"start_sample": pos, "end_sample": end, "owner": owner})
    pos = end
    turn = 1 - turn

# ---- Raised-cosine taper (avoids clicks at slot edges) ----
def taper(n, edge=50):
    edge = min(edge, n // 2)
    w = np.ones(n, dtype=np.float32)
    if edge > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
        w[:edge] = ramp
        w[-edge:] = ramp[::-1]
    return w

# ---- Gate each drone into its own slots only ----
drone1_gated = np.zeros(N, dtype=np.complex64)
drone2_gated = np.zeros(N, dtype=np.complex64)

for slot in slot_map:
    s, e = slot["start_sample"], slot["end_sample"]
    n = e - s
    w = taper(n)
    if slot["owner"] == "D1":
        drone1_gated[s:e] = iq1[s:e] * w
    else:
        drone2_gated[s:e] = iq2[s:e] * w

combined_iq = drone1_gated + drone2_gated
print("Combined stream built:", combined_iq.shape)

# ---- Find gaps using the REAL activity mask ----
gap_list = []
in_gap = False
gap_start = 0
min_gap_samples = int(0.0005 * FS)  # ignore gaps shorter than 0.5 ms

for slot in slot_map:
    s, e = slot["start_sample"], slot["end_sample"]
    real_mask = mask1 if slot["owner"] == "D1" else mask2
    slot_is_silent = not real_mask[s:e].any()

    if slot_is_silent and not in_gap:
        in_gap = True
        gap_start = s
    elif not slot_is_silent and in_gap:
        in_gap = False
        if s - gap_start >= min_gap_samples:
            gap_list.append({"start_sample": gap_start, "end_sample": s,
                              "duration_s": (s - gap_start)/FS})
if in_gap and N - gap_start >= min_gap_samples:
    gap_list.append({"start_sample": gap_start, "end_sample": N,
                      "duration_s": (N - gap_start)/FS})

print(f"Found {len(gap_list)} usable gaps")

# ---- Save output for Member C ----
meta_combined = {
    "fs_hz": FS,
    "fc_hz": meta1["fc_hz"],
    "tdma_slot_s": slot_len / FS,
    "drones": [meta1["drone"], meta2["drone"]],
}

np.savez("swarm_combined.npz",
    combined_iq=combined_iq,
    slot_map=np.array(slot_map, dtype=object),
    gap_list=np.array(gap_list, dtype=object),
    meta=np.array([meta_combined], dtype=object)
)
print("saved swarm_combined.npz")
```

### Verification / plotting code (run after the above)

```python
import scipy.signal as sig
import matplotlib.pyplot as plt

# ---- Spectrogram check ----
chunk = combined_iq[:4_000_000]
f, tt, Sxx = sig.spectrogram(chunk, fs=FS, nperseg=4096, return_onesided=False)

plt.figure(figsize=(12, 4))
plt.pcolormesh(tt*1e3, np.fft.fftshift(f)/1e6,
               10*np.log10(np.fft.fftshift(Sxx, axes=0)+1e-12), shading="auto")
plt.ylabel("Frequency (MHz)"); plt.xlabel("Time (ms)")
plt.title("Combined swarm stream — should show drones alternating, never stacked")
plt.colorbar(label="Power (dB)")
plt.show()

# ---- Power vs owner check (full length) ----
step = 2000
power = np.abs(combined_iq)**2
p_ds = power[::step]
t_ds = np.arange(len(p_ds)) * step / FS

owner_track = np.zeros(N)
for slot in slot_map:
    s, e = slot["start_sample"], slot["end_sample"]
    owner_track[s:e] = 0 if slot["owner"] == "D1" else 1
owner_ds = owner_track[::step]

fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(t_ds, 10*np.log10(p_ds + 1e-12), lw=0.7, color="steelblue", label="power (dB)")
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Power (dB)")

ax2 = ax1.twinx()
ax2.step(t_ds, owner_ds, where="post", color="orange", alpha=0.6, label="owner (0=D1, 1=D2)")
ax2.set_ylabel("Slot owner"); ax2.set_yticks([0, 1]); ax2.set_yticklabels(["D1", "D2"])

plt.title("Combined stream: power vs. assigned owner")
fig.legend(loc="upper right")
plt.show()

# ---- Gap stats ----
if len(gap_list) > 0:
    durations = [g["duration_s"] for g in gap_list]
    print(f"Gaps found: {len(gap_list)}")
    print(f"Total gap time: {sum(durations):.4f} s out of {N/FS:.2f} s total")
    print(f"Shortest gap: {min(durations)*1000:.2f} ms | Longest gap: {max(durations)*1000:.2f} ms")
else:
    print("⚠️ Still no gaps — try increasing slot_len (e.g. 0.005) and re-run")

# ---- Zoomed-in check (first 20 ms) ----
zoom_samples = int(0.020 * FS)
t_zoom = np.arange(zoom_samples) / FS
power_zoom = np.abs(combined_iq[:zoom_samples])**2

owner_zoom = np.zeros(zoom_samples)
for slot in slot_map:
    s, e = slot["start_sample"], slot["end_sample"]
    if s >= zoom_samples: break
    e = min(e, zoom_samples)
    owner_zoom[s:e] = 0 if slot["owner"] == "D1" else 1

fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(t_zoom*1000, 10*np.log10(power_zoom + 1e-12), lw=0.8, color="steelblue")
ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Power (dB)")
ax2 = ax1.twinx()
ax2.step(t_zoom*1000, owner_zoom, where="post", color="orange", alpha=0.6)
ax2.set_ylabel("Owner"); ax2.set_yticks([0,1]); ax2.set_yticklabels(["D1","D2"])
plt.title("Zoomed: first 20 ms — you should now see clear alternating blocks")
plt.show()
```