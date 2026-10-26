"""Download the audio track of a video with yt-dlp (no ffmpeg needed)."""

import logging
import os
from typing import Optional, Tuple

import yt_dlp

log = logging.getLogger(__name__)


class _Logger:
    """Route yt-dlp's own messages into our logger."""

    def debug(self, msg):
        if not msg.startswith("[debug] "):
            log.debug(msg)

    def info(self, msg):
        log.debug(msg)

    def warning(self, msg):
        log.warning(msg)

    def error(self, msg):
        log.error(msg)


def _progress_hook(d):
    if d["status"] == "downloading":
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        if total:
            pct = d["downloaded_bytes"] * 100 / total
            # Log roughly every 10% to keep stderr readable.
            bucket = int(pct // 10)
            if bucket != getattr(_progress_hook, "_last", -1):
                _progress_hook._last = bucket
                log.info("  downloading... %3.0f%%", pct)
    elif d["status"] == "finished":
        _progress_hook._last = -1
        log.info("  download finished")


def download_audio(
    url: str,
    workdir: str,
    cookies: Optional[str] = None,
    proxy: Optional[str] = None,
) -> Tuple[str, str]:
    """Download the best audio-only stream of `url` into `workdir`.

    Returns (audio_path, video_title).
    """
    opts = {
        # Prefer audio-only streams; fall back to a full file if none exist.
        "format": "bestaudio/best",
        "outtmpl": os.path.join(workdir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "logger": _Logger(),
        "progress_hooks": [_progress_hook],
        # YouTube's n-challenge needs a JS runtime (+ the yt-dlp-ejs package); yt-dlp
        # only enables deno by default, but the pods have node.
        "js_runtimes": {"deno": {}, "node": {}},
    }
    if cookies:
        opts["cookiefile"] = cookies
    if proxy:
        opts["proxy"] = proxy

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Bilibili multi-part videos come back as a playlist; take the first entry.
        if info.get("_type") == "playlist" and info.get("entries"):
            info = info["entries"][0]
        path = info.get("requested_downloads", [{}])[0].get("filepath") or ydl.prepare_filename(info)

    if not os.path.exists(path):
        raise RuntimeError(f"yt-dlp reported success but {path} does not exist")
    return path, info.get("title") or info.get("id") or "transcript"
