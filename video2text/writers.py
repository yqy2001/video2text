"""Write transcripts to .txt and .srt."""

from typing import List

from .transcribe import Segment


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _label(seg: Segment) -> str:
    return f"Speaker {seg.speaker}: " if seg.speaker is not None else ""


def write_txt(segments: List[Segment], path: str, timestamps: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            prefix = f"[{_hms(seg.start)}] " if timestamps else ""
            f.write(f"{prefix}{_label(seg)}{seg.text}\n")


def write_srt(segments: List[Segment], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_srt_time(seg.start)} --> {_srt_time(seg.end)}\n{_label(seg)}{seg.text}\n\n")
