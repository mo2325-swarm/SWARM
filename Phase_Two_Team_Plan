The Three Roles
Member 1 — Protocol & Data Architect

Responsible for: defining what the drones "say" to each other, and gathering enough source data to build a real dataset.

Tasks:

Design the inter-drone message protocol: decide on message types (e.g., presence beacon, acknowledgment, position report, warning, task assignment), their sizes, and how they're transmitted (simple, no-handshake, best-effort style).
Document this in a protocol specification file.
Build a small piece of code that can create and read these messages correctly, with a self-test to confirm it works.
Download additional drone recordings from DroneDetect — different drone models, same "clean" condition — to expand beyond the current 1 pair.
Reuse the Phase 1 cleaning process on each new recording.
Build a catalog file listing every available recording (drone model, file location, duration, activity level) for the rest of the team to use.

Hands off: the message-creation code → Member 2. The recording catalog → Member 3.

Member 2 — Signal Synthesis Engineer

Responsible for: turning the real protocol messages into actual radio signal data and embedding them realistically.

Tasks:

Take Member 1's message format and convert real message content into IQ signal data (the same modulation approach used in Phase 1, but now driven by real messages instead of random bits).
Add basic realism to the synthetic signal: slight frequency variation, background noise, and a range of signal strengths, so inserted messages aren't unrealistically clean.
Insert these messages into the real communication gaps identified in Phase 1's pipeline.
Re-run this on the original Air/Mavic Pro pair first to confirm correctness before scaling up.

Hands off: the finished message-insertion code → Member 3.

Member 3 — Dataset Assembly & Validation Lead

Responsible for: running the full pipeline at scale and proving the final dataset is trustworthy.

Tasks:

Loop through every drone pair in Member 1's catalog, running the TDMA-overlay step on each to produce multiple combined streams.
Run Member 2's message-insertion pipeline on every combined stream.
Merge everything into one final labeled dataset, tracking which source recording each example came from.
Check that there are enough labeled examples of each category to be statistically meaningful — not just a handful of samples.
Split the data for testing by source recording, not randomly, so results reflect real generalization rather than accidental memorization.
Run baseline classification tests and report results with proper statistical rigor (confidence intervals, multiple random trials).
Write the final documentation: what's in the dataset, how it was built, and its known limitations — clearly stating it is simulated.

Delivers: the final dataset, and the validation + documentation files.

Task Dependencies
Member 1 (protocol design)      ──────►  Member 2 (signal synthesis)
Member 1 (expanded recordings)  ──────►  Member 3 (batch assembly)
                                                │
Member 2 (signal synthesis)     ───────────────┘
                                                ▼
                                     Member 3 (final dataset +
                                       statistical validation)

Member 1's two tasks (protocol design, data gathering) are independent and can both start immediately.
Member 2 needs Member 1's protocol format, but can prototype early using a placeholder.
Member 3 needs both Member 1's expanded data and Member 2's finished synthesis code before the full run, but can build/test the batch logic early using the existing single drone pair.