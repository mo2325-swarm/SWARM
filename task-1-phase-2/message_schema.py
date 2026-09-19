"""
message_schema.py  (v2 — matches PROTOCOL_SPEC.md v2)

Member 1 deliverable — Protocol & Data Architect

Defines the inter-drone message format, including the preamble + repetition
scheme that lands every message type inside the pipeline's existing,
validated 0.2-1.0ms packet-duration envelope (see PROTOCOL_SPEC.md for the
full derivation). Provides serialize()/deserialize() for a single message,
plus build_on_air_bits()/parse_on_air_bits() for the full preamble+repeats
packet that Member 2's waveform generator actually needs to modulate.
"""

from dataclasses import dataclass
from enum import Enum
import random


class MsgType(Enum):
    BEACON = "BEACON"
    ACK = "ACK"
    POSITION_REPORT = "POSITION_REPORT"
    COLLISION_WARNING = "COLLISION_WARNING"
    TASK_ASSIGN = "TASK_ASSIGN"


# Payload sizes in bytes, per PROTOCOL_SPEC.md.
PAYLOAD_SIZES = {
    MsgType.BEACON: 4,
    MsgType.ACK: 2,
    MsgType.POSITION_REPORT: 12,
    MsgType.COLLISION_WARNING: 8,
    MsgType.TASK_ASSIGN: 20,
}

# Repeat counts per type, computed in PROTOCOL_SPEC.md against the pipeline's
# real symbol rate (3 Msym/s) to land each message at ~0.20-0.23ms on air.
# NOTE: POSITION_REPORT uses 10 (not 9) repeats so that its on-air length is
# unique. With 9 repeats its body was 136*9 = 1224 bits, identical to BEACON's
# 72*17 = 1224, which made the two types impossible to tell apart on decode.
REPEAT_COUNTS = {
    MsgType.BEACON: 17,
    MsgType.ACK: 21,
    MsgType.POSITION_REPORT: 10,
    MsgType.COLLISION_WARNING: 12,
    MsgType.TASK_ASSIGN: 6,
}

# Fixed 16-symbol sync preamble, shared by every message. Expressed here as
# a fixed bit pattern (32 bits = 16 QPSK symbols @ 2 bits/symbol). Chosen as
# an alternating-ish pattern with poor self-correlation shift properties
# avoided (not a rigorous Barker/Gold code — good enough for a synthetic
# dataset's sync anchor, not a real RF sync sequence).
PREAMBLE_BITS = "11010010111001011101001011100101"[:32]

_MSG_TYPE_LIST = list(MsgType)  # fixed order, used for compact integer encoding


def crc8(data: bytes) -> int:
    """Simple CRC-8 (polynomial 0x07) integrity check over the given bytes."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def _bits_to_bytes(bits: str) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError(f"Bit string length {len(bits)} is not a multiple of 8")
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


@dataclass
class Message:
    msg_type: MsgType
    src_id: int          # 1 = D1, 2 = D2
    dst_id: int          # 0 = broadcast, or specific drone id
    seq_no: int          # increments per message, wraps at 256
    payload: bytes
    checksum: int = 0    # computed by serialize(); not meant to be set manually

    # -- single-message (one repeat) serialization -------------------------

    def serialize(self) -> bytes:
        """Pack this message into raw bytes: [type][src][dst][seq][payload][crc]."""
        expected_len = PAYLOAD_SIZES[self.msg_type]
        if len(self.payload) != expected_len:
            raise ValueError(
                f"{self.msg_type.value} expects {expected_len}-byte payload, "
                f"got {len(self.payload)} bytes"
            )
        header = bytes([
            _MSG_TYPE_LIST.index(self.msg_type),
            self.src_id & 0xFF,
            self.dst_id & 0xFF,
            self.seq_no & 0xFF,
        ])
        body = header + self.payload
        self.checksum = crc8(body)
        return body + bytes([self.checksum])

    @staticmethod
    def deserialize(data: bytes) -> "Message":
        """Unpack raw bytes (as produced by serialize()) back into a Message."""
        if len(data) < 5:
            raise ValueError("Data too short to be a valid message")
        if data[0] >= len(_MSG_TYPE_LIST):
            raise ValueError(f"Unknown message-type byte {data[0]}")
        msg_type = _MSG_TYPE_LIST[data[0]]
        src_id, dst_id, seq_no = data[1], data[2], data[3]
        payload = data[4:-1]
        checksum = data[-1]

        expected_len = PAYLOAD_SIZES[msg_type]
        if len(payload) != expected_len:
            raise ValueError(
                f"Corrupt message: {msg_type.value} expects {expected_len} "
                f"byte payload, got {len(payload)}"
            )
        recomputed = crc8(data[:-1])
        if recomputed != checksum:
            raise ValueError(
                f"Checksum mismatch: expected {checksum}, computed {recomputed}"
            )
        return Message(msg_type, src_id, dst_id, seq_no, payload, checksum)

    # -- on-air packet (preamble + N repeats) -------------------------------

    def build_on_air_bits(self) -> str:
        """Build the full on-air bitstream: preamble + N repeats of this message.

        This is what Member 2's waveform generator should feed into the QPSK/RRC
        modulator (2 bits per symbol), producing a packet whose duration lands
        in the pipeline's validated 0.2-1.0ms envelope. See PROTOCOL_SPEC.md.
        """
        single_bits = _bytes_to_bits(self.serialize())
        n = REPEAT_COUNTS[self.msg_type]
        return PREAMBLE_BITS + (single_bits * n)

    @staticmethod
    def parse_on_air_bits(bits: str) -> "Message":
        """Inverse of build_on_air_bits(): strip the preamble, recover the
        repeated message copies, majority-vote each bit across repeats
        (basic redundancy decoding), and deserialize the result.

        Raises ValueError if the preamble doesn't match or the recovered
        message fails its checksum.
        """
        if not bits.startswith(PREAMBLE_BITS):
            raise ValueError("Preamble mismatch — not a valid on-air packet")
        payload_bits = bits[len(PREAMBLE_BITS):]

        # We don't know msg_type yet, so we can't know the exact repeat unit
        # length without decoding once. Decode assuming the shortest possible
        # single-message length first (ACK, 7 bytes = 56 bits) up through the
        # longest (TASK_ASSIGN, 25 bytes = 200 bits), and pick the length that
        # evenly divides the remaining bits and yields a valid checksum after
        # majority voting.
        for msg_type, payload_size in PAYLOAD_SIZES.items():
            unit_bits = (payload_size + 5) * 8  # +5 = header(4) + crc(1)
            n = REPEAT_COUNTS[msg_type]
            expected_total = unit_bits * n
            if len(payload_bits) != expected_total:
                continue

            # Majority vote each bit position across the n repeats
            copies = [payload_bits[i * unit_bits:(i + 1) * unit_bits] for i in range(n)]
            voted = []
            for pos in range(unit_bits):
                ones = sum(1 for c in copies if c[pos] == "1")
                voted.append("1" if ones * 2 > n else "0")
            voted_bits = "".join(voted)

            try:
                return Message.deserialize(_bits_to_bytes(voted_bits))
            except (ValueError, IndexError):
                continue  # length matched but content didn't check out; try next type

        raise ValueError("Could not decode on-air packet against any known message type")


def make_random_message(
    msg_type: MsgType,
    rng: random.Random,
    src_id: int = 1,
    dst_id: int = 0,
    seq_no: int = 0,
) -> Message:
    """Generate a Message of the given type with a random (but valid-size) payload."""
    size = PAYLOAD_SIZES[msg_type]
    payload = bytes(rng.randint(0, 255) for _ in range(size))
    return Message(msg_type, src_id, dst_id, seq_no, payload)


def random_msg_type(rng: random.Random) -> MsgType:
    """Pick a uniformly random message type."""
    return rng.choice(_MSG_TYPE_LIST)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = random.Random(42)
    all_passed = True

    print("=== Test 1: single-message round-trip (serialize/deserialize) ===")
    for mt in MsgType:
        msg = make_random_message(mt, rng, src_id=1, dst_id=2, seq_no=7)
        raw = msg.serialize()
        back = Message.deserialize(raw)
        ok = (back.msg_type == msg.msg_type and back.payload == msg.payload
              and back.checksum == msg.checksum)
        all_passed &= ok
        print(f"[{'OK' if ok else 'FAIL'}] {mt.value:<18} wire_size={len(raw)}B")

    print("\n=== Test 2: checksum corruption is detected ===")
    msg = make_random_message(MsgType.BEACON, rng, src_id=1, dst_id=0, seq_no=1)
    raw = bytearray(msg.serialize())
    raw[4] ^= 0xFF
    try:
        Message.deserialize(bytes(raw))
        print("[FAIL] Corrupted message was NOT detected")
        all_passed = False
    except ValueError as e:
        print(f"[OK] Corruption correctly detected: {e}")

    print("\n=== Test 3: on-air packet build + parse (preamble + repeats) ===")
    for mt in MsgType:
        msg = make_random_message(mt, rng, src_id=2, dst_id=1, seq_no=3)
        on_air = msg.build_on_air_bits()
        n = REPEAT_COUNTS[mt]
        expected_bits = len(PREAMBLE_BITS) + (PAYLOAD_SIZES[mt] + 5) * 8 * n
        duration_ms = (len(on_air) / 2) / 3_000_000 * 1000  # 2 bits/symbol, 3 Msym/s

        recovered = Message.parse_on_air_bits(on_air)
        ok = (len(on_air) == expected_bits and recovered.msg_type == msg.msg_type
              and recovered.payload == msg.payload)
        all_passed &= ok
        print(f"[{'OK' if ok else 'FAIL'}] {mt.value:<18} "
              f"on_air_bits={len(on_air):<6} duration={duration_ms:.4f}ms "
              f"repeats={n}")

    print("\n=== Test 4: on-air decoding survives bit errors (redundancy check) ===")
    msg = make_random_message(MsgType.TASK_ASSIGN, rng, src_id=1, dst_id=2, seq_no=9)
    on_air = list(msg.build_on_air_bits())
    # Flip a handful of bits scattered through the payload region (not the preamble)
    flip_positions = rng.sample(range(len(PREAMBLE_BITS), len(on_air)), 5)
    for pos in flip_positions:
        on_air[pos] = "1" if on_air[pos] == "0" else "0"
    corrupted = "".join(on_air)
    try:
        recovered = Message.parse_on_air_bits(corrupted)
        ok = recovered.payload == msg.payload
        all_passed &= ok
        print(f"[{'OK' if ok else 'FAIL'}] Recovered correctly despite 5 flipped bits "
              f"out of {len(on_air)} (majority vote across {REPEAT_COUNTS[msg.msg_type]} repeats)")
    except ValueError as e:
        print(f"[FAIL] Could not recover from bit errors: {e}")
        all_passed = False

    print("\n" + ("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"))
