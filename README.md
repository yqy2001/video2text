# video2text

Transcribe a YouTube or Bilibili video to a text file (speech-to-text).

- Downloads only the audio track with [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- Transcribes with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Works on GPU or CPU — no ffmpeg required

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
```

## Usage

```bash
video2text "https://www.youtube.com/watch?v=VIDEO_ID"
video2text "https://www.bilibili.com/video/BVxxxxxxxxxx" -l zh --srt
python -m video2text URL1 URL2 -o out/ --timestamps
```

Output goes to `./transcripts/<video title>.txt` (plus `.srt` with `--srt`).

| Option | Description |
|---|---|
| `-o, --output-dir` | Output directory (default `./transcripts`) |
| `-m, --model` | Whisper model; default `large-v3` on GPU, `large-v3-turbo` on CPU. Smaller = faster: `small`, `base`, `tiny` |
| `--compute-type` | Default `float16` on GPU, `int8` on CPU |
| `-l, --language` | Language code (`zh`, `en`, ...); auto-detected if omitted |
| `--device` | `auto` (default), `cuda`, or `cpu` |
| `--threads N` | CPU threads (default: `OMP_NUM_THREADS` or available CPUs) |
| `--no-vad` | Disable voice-activity filtering (it's retried off automatically if it finds no speech, e.g. songs) |
| `--keep-traditional` | Don't convert Chinese output to Simplified characters |
| `--srt` | Also write an `.srt` subtitle file |
| `--timestamps` | Prefix each line of the `.txt` with `[hh:mm:ss]` |
| `--cookies FILE` | Netscape `cookies.txt` for yt-dlp (login-only / age-restricted videos) |
| `--proxy URL` | Proxy for downloading, e.g. `http://127.0.0.1:7890` |
| `--keep-audio` | Keep the downloaded audio next to the transcript |

## Notes

- **No GPU?** The device is auto-detected. On CPU the tool uses `large-v3-turbo` with int8
  quantization on all cores. For faster (less accurate) results, try `-m small`.
- Models are downloaded from Hugging Face on first use (~1.5–3 GB for the large models).
  Set `HF_ENDPOINT=https://hf-mirror.com` if Hugging Face is unreachable.
- Chinese output is converted to Simplified characters (OpenCC); pass `--keep-traditional` to opt out.

## Troubleshooting

- **`429 Too Many Requests` / can't download model** — Hugging Face rate-limits shared IPs.
  Use the mirror: `export HF_ENDPOINT=https://hf-mirror.com`.
- **`cudaErrorInvalidDevice: invalid device ordinal`** — recent `ctranslate2` wheels need a newer
  NVIDIA driver. With driver 535 (CUDA 12.4), use `pip install ctranslate2==4.5.0`.
- **YouTube: `Video unavailable`** although the video exists — it's region-blocked for your IP
  (e.g. from mainland China). Use `--proxy` with an exit in another region, or `--cookies`.
- **Bilibili: `HTTP Error 412: Precondition Failed`** — Bilibili rate-limits IPs that download several
  videos quickly. Wait (minutes to an hour) or pass `--cookies` exported from a logged-in browser.
- **yt-dlp warns "No supported JavaScript runtime"** — YouTube extraction works without one for
  now, but installing [deno](https://deno.com) makes it more reliable.
