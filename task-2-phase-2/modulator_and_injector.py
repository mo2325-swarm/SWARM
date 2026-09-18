# import sys
# import os

# member_1_folder_path = r"G:\My Drive\02_University\Year_4\Swarm_Shield_Project\Code\SWARM\task-1-phase-2"
# sys.path.append(member_1_folder_path)

import numpy as np
import scipy.signal as sig
import random

# Import Member 1's work!

from message_schema import MsgType, make_random_message

def rrc_taps(beta, sps, span_symbols):
    """(Unchanged from Phase 1) Generates the Root-Raised-Cosine filter."""
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

def generate_realistic_protocol_packet(fs, sps, beta, py_rng, max_cfo_hz=500):

    """
    1. Asks Member 1's code for a message.
    2. Converts the bit string to QPSK.
    3. Shapes the pulse and adds real-world frequency drift (CFO).
    """
    import random
    
    # --- A. Get the bits from Member 1's protocol ---
    # We seed it using our numpy py_rng to keep the whole script reproducible!
    
    std_rng = random.Random(int(py_rng.integers(0, 999999)))
    
    # Use std_rng for Member 1's functions
    msg_type = std_rng.choice(list(MsgType))
    msg = make_random_message(msg_type, std_rng, src_id=1, dst_id=2)
    
    # Get the raw string of '1's and '0's (preamble + repeated payload)
    on_air_bits_str = msg.build_on_air_bits() 
    
    # Convert string "1011" to a numpy array of integers [1, 0, 1, 1]
    message_bits = np.array([int(b) for b in on_air_bits_str])

    # --- B. Map to QPSK ---
    # QPSK groups 2 bits per symbol. (e.g., [1, 0] becomes symbol 2)
    if len(message_bits) % 2 != 0:
        message_bits = np.append(message_bits, 0) # Pad if odd
        
    bit_pairs = message_bits.reshape(-1, 2)
    symbol_indices = bit_pairs[:, 0] * 2 + bit_pairs[:, 1]
    
    # QPSK Constellation coordinates
    const = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    symbols = const[symbol_indices]

    # --- C. Pulse Shaping ---
    num_symbols = len(symbols)
    upsampled = np.zeros(num_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    
    taps = rrc_taps(beta, sps, span_symbols=8)
    pulse = np.convolve(upsampled, taps, mode="same").astype(np.complex64)

    # --- D. INJECT REALISM: Carrier Frequency Offset (CFO) ---
    # We rotate the phase slightly over time to simulate hardware oscillator drift
    cfo_hz = py_rng.uniform(-max_cfo_hz, max_cfo_hz)
    t = np.arange(len(pulse)) / fs
    cfo_rotation = np.exp(1j * 2 * np.pi * cfo_hz * t)
    
    realistic_pulse = pulse * cfo_rotation

    # --- E. Normalize ---
    # Set power to 1.0 (The SNR scaling happens later when we put it in the gap)
    power = np.mean(np.abs(realistic_pulse) ** 2)
    if power > 0:
        realistic_pulse = realistic_pulse / np.sqrt(power)
        
    return realistic_pulse, msg_type.value

def local_noise_floor(iq, s, e):
    """
    Measures the background noise level right outside the gap.
    We need this to know how loud to make our synthetic packet.
    """
    seg = iq[max(0, s - 2000):s] if s > 0 else iq[e:e + 2000]
    if len(seg) == 0:
        seg = iq[s:e]
    return float(np.mean(np.abs(seg) ** 2)) + 1e-12

def synthesize_and_insert(combined_iq, gap_list, fs, sps, beta, py_rng, insert_prob=0.5, snr_min=-3.0, snr_max=15.0):
    """
    Loops through every silent gap, decides whether to inject a message,
    scales the volume (SNR), and adds ambient background noise.
    """
    iq = combined_iq.copy()
    insertions = []  # This is our Ground Truth log for Member 3

    for gi, gap in enumerate(gap_list):
        s, e = int(gap["start_sample"]), int(gap["end_sample"])
        gap_len = e - s
        margin = int(0.1e-3 * fs)  # Leave a 0.1ms buffer so we don't hit the real drone's signal
        usable = gap_len - 2 * margin

        # 1. Decide if we inject here (e.g., 50% chance)
        will_insert = py_rng.random() < insert_prob
        
        record = {"gap_id": gi, "gap_start": s, "gap_end": e, "inserted": False}

        if will_insert:
            # 2. Get the radio wave from our Step 1 Modulator
            packet, msg_type_name = generate_realistic_protocol_packet(fs, sps, beta, py_rng)
            plen = len(packet)

            # Check if our packet physically fits inside this specific gap
            if plen <= usable:
                # Pick a random starting position inside the gap
                offset = s + margin + py_rng.integers(0, usable - plen + 1)

                # 3. VARYING SIGNAL STRENGTHS (SNR)
                # Pick a random SNR between -3dB (very faint) and 15dB (loud)
                snr_db = py_rng.uniform(snr_min, snr_max)
                
                # Measure how loud the environment is right here, and scale our packet to match the SNR
                noise_p = local_noise_floor(iq, s, e)
                target_sig_p = noise_p * (10 ** (snr_db / 10))
                packet = packet * np.sqrt(target_sig_p)

                # 4. Inject the packet into the main radio stream!
                iq[offset:offset + plen] += packet.astype(np.complex64)

                # Update our log so Member 3 knows exactly what we did
                record.update({
                    "inserted": True,
                    "msg_type": msg_type_name,
                    "pkt_start": int(offset),
                    "pkt_end": int(offset + plen),
                    "snr_db": float(snr_db)
                })
                
        insertions.append(record)

    # 5. ADD BACKGROUND NOISE (Realism)
    # Add a faint, gritty layer of white noise across the entire recording so it isn't "mathematically perfect"
    sig_p = float(np.mean(np.abs(iq) ** 2))
    floor_p = sig_p * 0.05  # The noise floor is 5% of the average signal power
    
    noise = (py_rng.standard_normal(len(iq)) + 1j * py_rng.standard_normal(len(iq))).astype(np.complex64)
    noise *= np.sqrt(floor_p / 2)
    iq = iq + noise

    return iq, insertions

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal as sig
    
    print("Loading baseline data...")
    # 1. Load the merged drone data
    d = np.load("swarm_combined.npz", allow_pickle=True)
    combined_iq = d["combined_iq"]
    gap_list = list(d["gap_list"])
    meta = dict(d["meta"][0])
    
    fs = meta["fs_hz"]
    sps = 20
    beta = 0.35
    rng = np.random.default_rng(42)  # Fixed seed for reproducible testing

    print(f"Loaded {len(gap_list)} silent gaps. Running Injector...")
    
    # 2. Run your Member 2 Pipeline!
    modified_iq, insertions = synthesize_and_insert(
        combined_iq=combined_iq, 
        gap_list=gap_list, 
        fs=fs, sps=sps, beta=beta, 
        py_rng=rng, 
        insert_prob=0.5
    )
    
    # 3. Print the Ground Truth Log
    successful_injections = [r for r in insertions if r["inserted"]]
    print(f"Successfully injected {len(successful_injections)} packets!")
    
    if len(successful_injections) > 0:
        # Look at the very first successful injection
        test_packet = successful_injections[0]
        print("\n--- First Injection Receipt ---")
        print(f"Message Type: {test_packet['msg_type']}")
        print(f"Target SNR:   {test_packet['snr_db']:.2f} dB")
        print(f"Location:     Samples {test_packet['pkt_start']} to {test_packet['pkt_end']}")
        
        # 4. VISUAL TEST: Plot the Spectrogram
        print("\nGenerating Spectrogram...")
        
        # Grab a small window of audio around our injected packet
        start = max(0, test_packet["gap_start"] - 5000)
        end = min(len(modified_iq), test_packet["gap_end"] + 5000)
        window = modified_iq[start:end]
        
        f, t, Sxx = sig.spectrogram(window, fs=fs, nperseg=256, return_onesided=False)
        
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t * 1000, np.fft.fftshift(f) / 1e6, 
                       10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12), 
                       shading="auto", cmap='viridis')
        plt.title(f"Visual Test: {test_packet['msg_type']} Injected at {test_packet['snr_db']:.1f} dB")
        plt.ylabel("Frequency (MHz)")
        plt.xlabel("Time (ms)")
        plt.colorbar(label="Power (dB)")
        plt.tight_layout()
        plt.show()