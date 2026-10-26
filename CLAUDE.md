# video2text — notes for Claude

CLI that downloads a YouTube/Bilibili video's audio (yt-dlp) and transcribes it (faster-whisper).
Entry point: `video2text/cli.py`; run without installing via `PYTHONPATH=. python3 -m video2text URL`.

## User preference: GPU transcription outside the sandbox

- Always run transcription outside the execution sandbox with `--device cuda --compute-type float16`,
  using the execution tool's escalation mechanism. Do not fall back to CPU based on GPU checks inside
  the sandbox. This machine has 8 H20 GPUs; CUDA FP16 inference was verified outside the sandbox on
  2026-10-22. See [AGENTS.md](AGENTS.md) for the full instruction.

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
- Speaker diarization (on by default, `--no-speakers` to skip; `diarize.py`) uses sherpa-onnx (pyannote-segmentation-3.0 + CAM++ zh/en
  ONNX from GitHub releases; GitHub is fast from the pods, no HF token). pyannote.audio itself was avoided:
  gated HF models + it wants to replace the preinstalled torch.
- Speakers are assigned per Whisper segment, NOT per word: `word_timestamps=True` changes faster-whisper's
  seek windows and set off large-v3 repetition loops (one phrase ×15) on a 12-min interview. Don't re-add it.
- Diarization test clips (Bilibili): `av117468110524754` (12 min, male host + female guest + drama
  excerpts, separates well); `av639117650` (3 min, Q&A between similar voices, partly merged).
