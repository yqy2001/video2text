# video2text

Transcribe a YouTube or Bilibili video to a text file (speech-to-text).

- Downloads only the audio track with [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- Transcribes with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Works on GPU or CPU — no ffmpeg required

## Setup on a new machine

Requires Python ≥ 3.9. A GPU is optional.

1. **Get the code and install** (use `--user` if system site-packages isn't writable; the
   `video2text` command then lands in `~/.local/bin`, so make sure that's on your `PATH`):

   ```bash
   git clone https://github.com/yqy2001/video2text.git   # or `git pull` if already cloned
   cd video2text
   pip install --user -e .
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **If Hugging Face is slow or rate-limited (HTTP 429)**, use the mirror. Add it to `~/.bashrc` so it
   persists:

   ```bash
   echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc
   ```

3. **GPU machines only — check the NVIDIA driver:**

   ```bash
   nvidia-smi | head -4    # look at "Driver Version"
   ```

   With an older driver (e.g. 535 / CUDA 12.4), recent `ctranslate2` wheels fail with
   `cudaErrorInvalidDevice`. Pin the last compatible version:

   ```bash
   pip install --user ctranslate2==4.5.0
   ```

4. **Smoke test with the tiny model** (75 MB download, takes seconds):

   ```bash
   video2text "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -m tiny -o /tmp/v2t_test
   ```

   The first log line shows the device: `Loading model tiny on cuda (float16)` on a GPU,
   or `on cpu (int8, N threads)` otherwise.

5. **Optional: pre-download the default model** so the first real run doesn't wait. It's about 3 GB,
   which can take close to an hour on a slow link:

   ```bash
   python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu')"        # GPU default
   python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu')"  # CPU default
   ```

   Models are cached in `~/.cache/huggingface/hub`. Copying that directory from another machine
   works too.

## Usage

```bash
video2text "https://www.youtube.com/watch?v=VIDEO_ID"
video2text "https://www.bilibili.com/video/BVxxxxxxxxxx" -l zh --srt
python -m video2text URL1 URL2 -o out/ --no-timestamps
```

Output goes to `./transcripts/<video title>.txt` (plus `.srt` with `--srt`). Each line is
`[hh:mm:ss] Speaker N: text`; pass `--no-timestamps` to drop the time and `--no-speakers` to drop
the speaker.

### Speaker labels

By default, each line is labeled with who is speaking (`[00:02:32] Speaker 2: ...`, also in the
`.srt`). If only one speaker is found, the labels are left off. This uses `sherpa-onnx` (installed
with the package). Its models (~35 MB, from GitHub, no Hugging Face token needed) download to
`~/.cache/video2text/diarization` on first use. It runs on CPU and takes ~15 s for a 3-minute video.
Pass `--no-speakers` to skip it.

- Speakers are numbered by first appearance. They're anonymous: you get "Speaker 2", not a name.
- If you know how many people talk, pass `--num-speakers N`. It's more reliable than estimating.
- Too many speakers (one person split in two)? Raise `--speaker-threshold` (default 0.6), e.g. 0.7.
  Too few (two people merged)? Lower it.
- Labels are per transcript line, so if a speaker change falls inside one Whisper segment, the whole
  line goes to whoever talks longer. Quick back-and-forth between similar voices (for example, two
  people of the same gender on one mic) can get merged into one speaker.

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
| `--no-speakers` | Don't label lines with the speaker (skips diarization) |
| `--num-speakers N` | Number of speakers, if known (default: estimate) |
| `--speaker-threshold X` | Speaker clustering threshold; lower = more speakers (default 0.6) |
| `--srt` | Also write an `.srt` subtitle file |
| `--no-timestamps` | Plain text: drop the `[hh:mm:ss]` prefix from each `.txt` line |
| `--cookies FILE` | Netscape `cookies.txt` for yt-dlp (login-only / age-restricted videos) |
| `--proxy URL` | Proxy for downloading, e.g. `http://127.0.0.1:7890` |
| `--keep-audio` | Keep the downloaded audio next to the transcript |

## Notes

- **No GPU?** The device is auto-detected. On CPU the tool uses `large-v3-turbo` with int8
  quantization on all available cores (respecting `OMP_NUM_THREADS` / container CPU quota). For faster (less accurate) results, try `-m small`.
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
