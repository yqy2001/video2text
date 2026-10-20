# video2text — notes for Claude

CLI that downloads a YouTube/Bilibili video's audio (yt-dlp) and transcribes it (faster-whisper).
Entry point: `video2text/cli.py`; run without installing via `PYTHONPATH=. python3 -m video2text URL`.

## Dev environment gotchas (company k8s pods; seen on multiple machines)

- `pip install` needs `--user` (system site-packages is read-only); console scripts go to `~/.local/bin`.
- huggingface.co returns 429 for the shared proxy IP → `export HF_ENDPOINT=https://hf-mirror.com`.
  Model downloads run ~1 MB/s, so large-v3 (~3 GB) takes ~50 min the first time on a new machine.
  For quick tests use `-m tiny`.
- NVIDIA driver 535 (CUDA 12.4): ctranslate2 4.8.x fails with `cudaErrorInvalidDevice: invalid device ordinal`;
  `pip install --user ctranslate2==4.5.0` works. Check `nvidia-smi` driver version on a new machine.
- Pods see ~180 CPUs but have a quota (`OMP_NUM_THREADS` is set, e.g. 19). Never size thread pools by
  `os.cpu_count()` — it made CPU inference ~40× slower. `default_cpu_threads()` in `transcribe.py` handles this.
- Outbound proxy exits in Beijing → some YouTube videos report "Video unavailable" (region block), not a bug.
  `dQw4w9WgXcQ` is playable for testing; `jNQXAC9IVRw` is not. Bilibili works directly.
- Bilibili returns HTTP 412 after a handful of quick downloads from the same IP; it clears after a while.
  Don't burn test runs re-downloading the same video — use `--keep-audio` and reuse the file.
- Installing faster-whisper bumps protobuf to 7.x, which conflicts with the preinstalled tensorflow-cpu.

## Behavior notes

- Silero VAD drops sung vocals entirely; `transcribe()` retries without VAD when VAD yields 0 segments.
- Chinese: the Simplified `initial_prompt` only steers the first window — Whisper drifts into Traditional
  mid-transcript — so `transcribe()` also converts with OpenCC t2s. VAD also drops the sung parts of
  concert/music videos that also have speech; `--no-vad` keeps them.
