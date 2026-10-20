"""Command-line entry point: video2text URL [URL ...]"""

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
import time

from .download import download_audio
from .transcribe import load_model, transcribe
from .writers import write_srt, write_txt

log = logging.getLogger("video2text")


def sanitize_filename(name: str, max_len: int = 150) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name).strip(" ._")
    return name[:max_len] or "transcript"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="video2text",
        description="Transcribe YouTube / Bilibili videos to text with Whisper.",
    )
    p.add_argument("urls", nargs="+", metavar="URL", help="YouTube or Bilibili video URL(s)")
    p.add_argument("-o", "--output-dir", default="transcripts", help="output directory (default: ./transcripts)")
    p.add_argument("-m", "--model", default=None,
                   help="Whisper model (default: large-v3 on GPU, large-v3-turbo on CPU; e.g. small, medium)")
    p.add_argument("--compute-type", default=None, help="default: float16 on GPU, int8 on CPU")
    p.add_argument("-l", "--language", default=None, help="language code, e.g. zh, en (default: auto-detect)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="default: auto")
    p.add_argument("--threads", type=int, default=None,
                   help="CPU threads (default: OMP_NUM_THREADS or available CPUs)")
    p.add_argument("--no-vad", action="store_true",
                   help="disable voice-activity filtering (VAD is retried off automatically if it finds no speech)")
    p.add_argument("--keep-traditional", action="store_true",
                   help="don't convert Chinese output to Simplified characters")
    p.add_argument("--srt", action="store_true", help="also write an .srt subtitle file")
    p.add_argument("--timestamps", action="store_true", help="prefix each .txt line with [hh:mm:ss]")
    p.add_argument("--cookies", help="cookies.txt file passed to yt-dlp")
    p.add_argument("--proxy", help="proxy URL passed to yt-dlp, e.g. http://127.0.0.1:7890")
    p.add_argument("--keep-audio", action="store_true", help="keep the downloaded audio in the output dir")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return p.parse_args(argv)


def process(url, model, args) -> None:
    with tempfile.TemporaryDirectory(prefix="video2text_") as tmp:
        log.info("Downloading %s", url)
        audio_path, title = download_audio(url, tmp, cookies=args.cookies, proxy=args.proxy)
        log.info("  title: %s", title)

        t0 = time.time()
        segments, language = transcribe(model, audio_path, args.language, vad=not args.no_vad,
                                        simplify_zh=not args.keep_traditional)
        log.info("  transcribed %d segments (%s) in %.1fs", len(segments), language, time.time() - t0)

        base = os.path.join(args.output_dir, sanitize_filename(title))
        write_txt(segments, base + ".txt", timestamps=args.timestamps)
        log.info("  wrote %s.txt", base)
        if args.srt:
            write_srt(segments, base + ".srt")
            log.info("  wrote %s.srt", base)
        if args.keep_audio:
            dest = base + os.path.splitext(audio_path)[1]
            shutil.move(audio_path, dest)
            log.info("  kept audio at %s", dest)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    if not args.verbose:
        # faster-whisper logs every VAD/processing step at INFO.
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.model, args.device, args.compute_type, args.threads)

    failed = 0
    for url in args.urls:
        try:
            process(url, model, args)
        except Exception as e:  # keep going with the remaining URLs
            failed += 1
            log.error("FAILED %s: %s", url, e)
            if args.verbose:
                log.exception(e)
    if failed:
        log.error("%d of %d URL(s) failed", failed, len(args.urls))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
