## 1. `PROTOCOL_SPEC.md` — the design document

**What it is:** A written specification (not code) that defines what the fake "drone-to-drone" messages actually look like.

**What's inside it:**
- **5 message types**, each with a real purpose: BEACON (presence ping), ACK (acknowledgment), POSITION_REPORT (location data), COLLISION_WARNING (safety alert), TASK_ASSIGN (coordination command)
- **Exact byte sizes** for each message type's payload
- **The transport model**: simple, one-way, no-handshake messaging (like a walkie-talkie, not a phone call)
- **The packet structure**: every message is preceded by a fixed "preamble" (a sync signal) and then repeated several times back-to-back
- **The math behind why it's built this way**: originally, real messages turned out to be way too short (0.01–0.03ms) compared to what your team's existing pipeline was already tuned for (0.2–1.0ms packets). Adding the preamble and repeating the message solved that mismatch, bringing every message type to a consistent ~0.2ms — without needing to re-tune anything else in the already-validated pipeline

**Why it matters:** this is the "rulebook" everyone else builds against. Member 2 reads this to know what to synthesize; Member 3 reads it to understand what's being validated.

---

## 2. `message_schema.py` — the message-building code

**What it is:** Working Python code that actually implements the rules from the spec document.

**What it does:**
- **Creates messages** — builds a properly structured message object (type, sender, receiver, sequence number, payload, checksum)
- **Converts messages to raw bytes** (`serialize()`) and **back again** (`deserialize()`) — like packing a letter into an envelope and unpacking it
- **Builds the full "on-air" signal content** (`build_on_air_bits()`) — this is the actual function Member 2 needs: it takes a message and produces the complete bitstream (preamble + repeated copies) ready to be turned into a radio signal
- **Decodes a received on-air signal back into a message** (`parse_on_air_bits()`) — including figuring out which message type it is, and using "majority voting" across the repeated copies to recover the correct message even if some bits got corrupted along the way

**Why it matters:** I tested this thoroughly — all 5 message types round-trip correctly, corruption is detected properly, and I specifically tested that the redundancy (repeating the message) actually works: I flipped 5 random bits in a message and confirmed it still decoded correctly. This is the piece Member 2 will import and use directly.

---

## 3. `build_corpus.py` — the batch data-processing script

**What it is:** An automation script that expands your pool of drone recordings.

**What it does:**
- Takes a list of drone `.dat` files (you fill this list in)
- Runs each one through the **exact same cleaning process** from your original Task 1 — reading the raw signal, normalizing it, and detecting when the drone was actually transmitting vs. silent
- Saves each cleaned file automatically
- Builds a summary spreadsheet (`corpus_catalog.csv`) listing every recording — which drone, where the file is, how long it is, and what percentage of time it was active

**Why it matters:** right now you only have 2 drone recordings, which isn't enough for a trustworthy dataset. This script lets you process many more drone files automatically instead of manually repeating Task 1's steps by hand each time. Member 3 will use the resulting catalog to loop through all available drone pairs when building the final dataset at scale.

---

**How the three connect:** the spec document defines the rules → the message schema code implements those rules in a way Member 2 can build on → the corpus script makes sure there's enough real drone data available for Member 3 to actually build a large-scale dataset from.