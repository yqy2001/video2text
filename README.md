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
| `--proxy URL` | Proxy for downloading, e.g. `http://127.0.0.1:7890` or `socks5h://127.0.0.1:1080` |
| `--keep-audio` | Keep the downloaded audio next to the transcript |

## YouTube through another machine (SSH tunnel)

If this machine can't reach YouTube (or YouTube blocks its IP), route only the download through a
machine that can. Transcription still runs locally on this machine's GPU. Below, **machine B** is the
one with YouTube access (e.g. your laptop) and **machine A** is the one that transcribes.

### First: SSH from B to A (once per pod)

On a Merlin/Arnold dev pod, the platform runs its own `sshd`, not the system one:

- **Address and port:** on A, run `echo $MERLIN_SSHD_HOST $MERLIN_SSHD_PORT`. The host is the pod's
  IPv6 address and the port is something like `10903`. The user is `tiger`. Both values change when
  the pod is recreated, so check them again on a new pod. B must be able to reach that IPv6 address
  (e.g. from the company network).
- **Authorized keys:** this `sshd` reads **`~/.arnold_ssh/authorized_keys`**. The usual
  `~/.ssh/authorized_keys` is ignored. There's no password login.

1. **On B: get a public key.** If `ls ~/.ssh/id_ed25519.pub` finds nothing, create one with
   `ssh-keygen -t ed25519`. Then print it with `cat ~/.ssh/id_ed25519.pub`.
2. **On A: authorize the key.** Open a terminal through the platform's web terminal or VS Code, then
   append B's key:

   ```bash
   echo 'ssh-ed25519 AAAA... you@laptop' >> ~/.arnold_ssh/authorized_keys
   ```

3. **On B: add a host alias** to `~/.ssh/config`, so you don't have to type the IPv6 address and port
   again. scp also needs IPv6 addresses in brackets (`tiger@[2605:...]:path`), and the alias avoids
   that:

   ```
   Host v2t
       HostName <MERLIN_SSHD_HOST>
       Port <MERLIN_SSHD_PORT>
       User tiger
       ServerAliveInterval 30
       ServerAliveCountMax 3
   ```

4. **On B: test it** with `ssh v2t hostname`. If the pod was recreated at the same address and you
   get `REMOTE HOST IDENTIFICATION HAS CHANGED`, remove the old host key with
   `ssh-keygen -R '[<host>]:<port>'`.

### What you need to do

1. **On B: open a reverse SOCKS tunnel to A** and keep it running (e.g. inside `tmux`).
   `-R` with only a port and no destination makes A's `127.0.0.1:1080` a SOCKS5 proxy that exits
   from B. This needs OpenSSH ≥ 7.6 on B.

   ```bash
   ssh -N -T -R 127.0.0.1:1080 -o ExitOnForwardFailure=yes v2t
   ```

   Without the alias, run
   `ssh -p <port> -N -T -R 127.0.0.1:1080 -o ExitOnForwardFailure=yes tiger@<IPv6 address>`.

2. **On A: check the tunnel.** The second command should print B's exit IP:

   ```bash
   ss -ltn | grep 1080
   curl -x socks5h://127.0.0.1:1080 https://ipinfo.io/json
   ```

3. **On B: export YouTube cookies** (needed when YouTube says "Sign in to confirm you're not a bot",
   which happens often from datacenter IPs). YouTube rotates the cookies of open sessions, so export
   them from a private window:
   1. Open a private/incognito window and log in to YouTube. Using a throwaway account is safer,
      because the file gives full access to the account.
   2. In that window, go to `https://www.youtube.com/robots.txt` and make it the only tab.
   3. Export cookies with a browser extension such as "Get cookies.txt LOCALLY" (allow it in
      incognito). Export from the **youtube.com** tab: a google.com tab gives only Google cookies.
   4. Close the private window right away.

   Copy the file into A's checkout, e.g. `scp cookies.txt v2t:/opt/tiger/video2text/`. It's
   gitignored.

4. **On A: check the cookies.** The cookies only count as a login if `LOGIN_INFO` is there:

   ```bash
   chmod 600 cookies.txt
   grep -c youtube.com cookies.txt     # should be well above 0
   grep -q LOGIN_INFO cookies.txt && echo logged-in || echo "NOT logged in - export again"
   ```

5. **On A, once: make sure YouTube's JS challenge can be solved.** Logged-in YouTube requires it,
   and yt-dlp needs the `yt-dlp-ejs` package (installed by `pip install --user -e .`) plus Node.js
   or deno:

   ```bash
   pip install --user -e .       # pulls in yt-dlp-ejs
   node --version || deno --version
   ```

6. **On A: transcribe.** Use `socks5h` (not `socks5`) so DNS is also resolved on B:

   ```bash
   video2text "https://www.youtube.com/watch?v=VIDEO_ID" \
     --proxy socks5h://127.0.0.1:1080 --cookies cookies.txt --keep-audio
   ```

   The tunnel only needs to stay up during the download. `--keep-audio` keeps the audio, so you can
   re-transcribe it without downloading it again.

Later, only steps 1 and 6 are needed. If the bot check comes back, the cookies have expired or been
rotated: go back to step 3.

### Errors along the way

| Error | Cause | Fix |
|---|---|---|
| `ssh v2t` hangs or `Permission denied (publickey)` | Pod address or port changed, or key not in `~/.arnold_ssh/authorized_keys` | Recheck `$MERLIN_SSHD_HOST`/`$MERLIN_SSHD_PORT` and the key |
| `remote port forwarding failed for listen port 1080` | An old tunnel still holds port 1080 on A | Kill the old `ssh` on B, or wait for it to time out |
| `ss` shows nothing on port 1080 | Tunnel is down | Restart step 1 on B |
| `Sign in to confirm you're not a bot` | YouTube distrusts the exit IP; no cookies, or cookies not logged in | Steps 3–4 (check `LOGIN_INFO`) |
| `n challenge solving failed` / `The page needs to be reloaded` | No JS challenge solver | Step 5 |
| `Video unavailable` | Region block at the exit IP | Use an exit in another region |

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
- **yt-dlp warns "No supported JavaScript runtime"** — install Node.js or [deno](https://deno.com).
  Logged-out YouTube may still work without one, but with `--cookies` it fails with "The page needs
  to be reloaded". The `yt-dlp-ejs` challenge solver is installed with the package. See
  [YouTube through another machine](#youtube-through-another-machine-ssh-tunnel).
