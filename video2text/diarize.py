"""Speaker diarization with sherpa-onnx (pyannote segmentation + CAM++ embeddings).

Models are ONNX files from sherpa-onnx's GitHub releases, so no Hugging Face
token is needed. They are downloaded once to ~/.cache/video2text/diarization.
"""

import logging
import os
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from faster_whisper import decode_audio

from .transcribe import Segment, default_cpu_threads

log = logging.getLogger(__name__)

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_SEG_ARCHIVE = f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
# CAM++ trained on Chinese + English speech.
_EMB_URL = (f"{_RELEASES}/speaker-recongition-models/"
            "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "video2text", "diarization")


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def _download(url: str, dest: str) -> None:
    log.info("  downloading %s", url)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def check_available() -> None:
    """Fail fast (before a long transcription) if diarization can't run."""
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        raise SystemExit("speaker labels need sherpa-onnx: pip install --user sherpa-onnx "
                         "(or pass --no-speakers)") from None
    ensure_models()


def ensure_models(cache_dir: str = CACHE_DIR):
    """Return (segmentation_model, embedding_model) paths, downloading if missing."""
    os.makedirs(cache_dir, exist_ok=True)
    seg = os.path.join(cache_dir, "pyannote-segmentation-3.0.onnx")
    emb = os.path.join(cache_dir, os.path.basename(_EMB_URL))
    if not os.path.exists(seg):
        archive = os.path.join(cache_dir, "segmentation.tar.bz2")
        _download(_SEG_ARCHIVE, archive)
        with tarfile.open(archive) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/model.onnx"))
            src = tar.extractfile(member)
            with open(seg + ".part", "wb") as out:
                out.write(src.read())
        os.replace(seg + ".part", seg)
        os.remove(archive)
    if not os.path.exists(emb):
        _download(_EMB_URL, emb)
    return seg, emb


def diarize(
    audio_path: str,
    num_speakers: Optional[int] = None,
    threshold: float = 0.6,
    threads: Optional[int] = None,
) -> List[Turn]:
    """Return speaker turns. `threshold` only applies when num_speakers is None."""
    import sherpa_onnx

    seg_model, emb_model = ensure_models()
    threads = threads or default_cpu_threads()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_model),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model, num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=threshold),
    )
    if not config.validate():
        raise RuntimeError("invalid speaker diarization config")
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)

    audio = decode_audio(audio_path, sampling_rate=sd.sample_rate)
    return [Turn(r.start, r.end, r.speaker) for r in sd.process(audio).sort_by_start_time()]


def assign_speakers(segments: List[Segment], turns: List[Turn]) -> None:
    """Label each segment with the speaker who talks most during it, numbering
    speakers 1, 2, ... in order of first appearance. With a single speaker the
    labels would be noise, so segments are left unlabeled.

    Whole segments, not words: faster-whisper's word_timestamps=True shifts its
    decoding windows and set off repetition loops in large-v3 on real interviews.
    """
    relabel = {}
    for seg in segments:
        talk = Counter()
        for t in turns:
            overlap = min(seg.end, t.end) - max(seg.start, t.start)
            if overlap > 0:
                talk[t.speaker] += overlap
        if talk:
            spk = talk.most_common(1)[0][0]
        elif turns:  # no overlap (e.g. a pause Whisper filled in): nearest turn
            mid = (seg.start + seg.end) / 2
            spk = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker
        else:
            continue
        seg.speaker = relabel.setdefault(spk, len(relabel) + 1)
    log.info("  found %d speaker(s)", len(relabel))
    if len(relabel) == 1:
        for seg in segments:
            seg.speaker = None
