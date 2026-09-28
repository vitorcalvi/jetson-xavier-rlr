"""Small, hardware-independent pieces of the Phase F pipeline."""
from __future__ import annotations

from typing import Iterable, Sequence


def dedupe_detections(previous: Sequence[str] | None, current: Sequence[str]) -> tuple[list[str], bool]:
    """Return stable current labels and whether the scene changed."""
    normalized = list(dict.fromkeys(str(item) for item in current))
    return normalized, list(previous or ()) != normalized


def build_prompt(self_state: str, detections: Iterable[str], transcript: str) -> str:
    """Build a bounded prompt with a fixed, explicit self-state section."""
    vision = " ".join(detections) or "(none)"
    heard = transcript.strip() or "(no transcript)"
    return (
        "You are a concise, self-aware assistant operating on an embedded device. "
        "Use only the observations and state below; distinguish observations from guesses.\n"
        "<self-state>\n" + self_state.strip() + "\n</self-state>\n"
        "<vision>\n" + vision + "\n</vision>\n"
        "<user-audio>\n" + heard + "\n</user-audio>\n"
        "Reply in at most two short sentences."
    )


def pcm_frame_bytes(sample_rate: int, frame_ms: int = 20, sample_width: int = 2) -> int:
    """Number of bytes in one mono PCM frame of the requested duration."""
    if sample_rate <= 0 or frame_ms not in (10, 20, 30) or sample_width <= 0:
        raise ValueError("invalid PCM frame parameters")
    samples = sample_rate * frame_ms // 1000
    if samples * 1000 != sample_rate * frame_ms:
        raise ValueError("frame duration does not produce a whole number of samples")
    return samples * sample_width


def vad_chunks(pcm: bytes, sample_rate: int = 16000, frame_ms: int = 20) -> list[bytes]:
    """Split PCM into complete mono int16 frames, discarding a trailing partial frame."""
    size = pcm_frame_bytes(sample_rate, frame_ms)
    return [pcm[offset:offset + size] for offset in range(0, len(pcm) - size + 1, size)]
