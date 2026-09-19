# SWARM — Synthetic Dataset for Multi-Drone Coordination

A labeled, **simulated** RF dataset. We take two real single-drone recordings,
fake a TDMA-based "conversation" between them, find the genuine silent gaps, and
insert synthetic drone-to-drone messages into those gaps.

Sample recordings:
https://drive.google.com/drive/folders/13NiHfzwg64ev2NH91plwsvsKpimlIqsP?usp=sharing

## Phases
- **Phase 1** — clean two recordings → TDMA overlay → find gaps → insert a packet.
- **Phase 2** — real message protocol, more drone pairs, realistic synthesis, and a
  validated dataset built at scale.

## Repository layout
```
docs/                     team plans and diagrams
  Phase_One_Team_Plan.pdf
  Phase_Two_Team_Plan.md
  The_Seven_Tasks.png
  CHANGES_phase2_member3.md   what Member 3 fixed and why

task-1/  task-2/  task-3/          Phase 1 work (Members A/B/C)
task-1-phase-2/                    Member 1 — protocol + data catalog
  PROTOCOL_SPEC.md
  message_schema.py                <-- single source of truth for the protocol
  build_corpus.py
task-2-phase-2/                    Member 2 — signal synthesis + injection
  modulator_and_injector.py
  message_schema.py                (shim -> re-exports task-1-phase-2 copy)
```

## Roles (Phase 2)
- **Member 1** — Protocol & Data Architect
- **Member 2** — Signal Synthesis Engineer
- **Member 3** — Dataset Assembly & Validation Lead
