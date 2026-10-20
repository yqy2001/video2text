"""Speech-to-text with faster-whisper, on GPU when available, else CPU."""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import ctranslate2
from faster_whisper import WhisperModel, decode_audio
from opencc import OpenCC

log = logging.getLogger(__name__)

# Whisper tends to emit Traditional characters for Mandarin; a Simplified
# prompt nudges it toward Simplified output.
ZH_PROMPT = "以下是普通话的句子。"


@dataclass
class Segment:
    start: float
    end: float
    text: str


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def default_cpu_threads() -> int:
    """CPUs actually available to us. os.cpu_count() reports every host core,
    which badly oversubscribes containers with a CPU quota."""
    env = os.environ.get("OMP_NUM_THREADS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    try:  # cgroup v2 quota, e.g. "1900000 100000"
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def load_model(
    model: Optional[str],
    device: str,
    compute_type: Optional[str],
    cpu_threads: Optional[int] = None,
) -> WhisperModel:
    device = resolve_device(device)
    if device == "cuda":
        model = model or "large-v3"
        compute_type = compute_type or "float16"
        extra = {}
    else:
        # large-v3-turbo + int8 is far faster than large-v3 on CPU at
        # nearly the same accuracy.
        model = model or "large-v3-turbo"
        compute_type = compute_type or "int8"
        extra = {"cpu_threads": cpu_threads or default_cpu_threads()}
    log.info(
        "Loading model %s on %s (%s%s)", model, device, compute_type,
        f", {extra['cpu_threads']} threads" if extra else "",
    )
    return WhisperModel(model, device=device, compute_type=compute_type, **extra)


def transcribe(
    model: WhisperModel,
    audio_path: str,
    language: Optional[str] = None,
    vad: bool = True,
    simplify_zh: bool = True,
) -> Tuple[List[Segment], str]:
    """Transcribe `audio_path`. Returns (segments, language)."""
    audio = decode_audio(audio_path, sampling_rate=model.feature_extractor.sampling_rate)
    log.info("  audio length: %s", _fmt(len(audio) / model.feature_extractor.sampling_rate))

    if language is None:
        language, prob, _ = model.detect_language(audio, vad_filter=vad)
        log.info("  detected language: %s (p=%.2f)", language, prob)

    segments = _run(model, audio, language, vad)
    if not segments and vad:
        # VAD treats singing / speech over loud music as non-speech.
        log.info("  VAD found no speech, retrying without VAD")
        segments = _run(model, audio, language, vad=False)

    if language == "zh" and simplify_zh:
        # The prompt only steers the start; Whisper can drift into Traditional
        # mid-transcript, so convert deterministically.
        t2s = OpenCC("t2s")
        for seg in segments:
            seg.text = t2s.convert(seg.text)
    return segments, language


def _run(model: WhisperModel, audio, language: str, vad: bool) -> List[Segment]:
    segments_iter, _ = model.transcribe(
        audio,
        language=language,
        vad_filter=vad,
        initial_prompt=ZH_PROMPT if language == "zh" else None,
    )

    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append(Segment(seg.start, seg.end, text))
            if len(segments) % 50 == 0:
                log.info("  ... %d segments, at %s", len(segments), _fmt(seg.end))
    return segments


def _fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
