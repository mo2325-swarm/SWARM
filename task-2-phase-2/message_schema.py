"""
message_schema.py  (shim — do not edit)

The real protocol lives in ONE place: task-1-phase-2/message_schema.py
(Member 1's deliverable). This file used to be a full copy, which meant the two
copies could silently drift apart. It now just re-exports the canonical module,
so Member 2's scripts keep working when run from this folder while there is only
one source of truth for the protocol.
"""
import os
import importlib.util

_CANON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "task-1-phase-2", "message_schema.py")
)
_spec = importlib.util.spec_from_file_location("swarm_message_schema", _CANON)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export every public name from the canonical module.
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("_")})
