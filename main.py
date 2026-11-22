"""
Daily 2D Shorts bot
Idea + script + title + description (Gemini) -> voice (free TTS) -> 2D animation
(your own character images) -> upload to YouTube.
Runs on GitHub Actions, no PC needed.
"""
import asyncio
import datetime
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======================= SETTINGS (yahan badal sakte ho) =======================
VOICE = "en-US-GuyNeural"        # edge-tts voice (deadpan male). Aur voices: en-US-ChristopherNeural, en-GB-RyanNeural
LANG = "en"                      # gTTS backup voice language
FPS = 30
W, H = 1080, 1920                # vertical Shorts size
CHAR_SIZE = 1000                 # character picture size on screen
CHAR_Y = 300                     # character top position
PANEL_Y = 1290                   # black caption panel starts here
HOOK_COLOR = (0, 0, 0)
ACCENT = (255, 205, 0)           # yellow accent (last line + bar)
MODELS = [                       # tried in this order, first one that works is used
    os.environ.get("GEMINI_MODEL", "").strip(),
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]
SR = 24000                       # audio sample rate

ROOT = Path(__file__).parent
OUT = ROOT / "out"
USED = ROOT / "used_ideas.txt"

PROMPT = """You write scripts for a YouTube Shorts channel.
Format: a sleepy, deadpan cartoon teenager (black hoodie, messy black hair) narrates ONE absurd "What if..." idea in about 25 seconds.
Rules:
- Spoken lines only: 7 to 9 short lines, each max 12 words, about 55 to 70 words in total.
- Line 1 is a strong hook that starts with "What if".
- Escalate with 3 funny, surprising beats. Dry humor. Family friendly.
- The last line is a twist or punchline that also works as a loop back to line 1.
- No real people, brands or copyrighted characters.
- Do NOT repeat these earlier ideas: {used}
Return ONLY valid JSON with these keys:
"hook_text": max 28 characters, uppercase, shown on screen at the top,
"title": YouTube title, max 80 characters, curiosity driven, no hashtags,
"description": 2 short sentences about the video, then a blank line, then 4 hashtags including #shorts,
"tags": list of 6 to 10 keywords,
"lines": list of the spoken lines.
Today's date, for variety: {today}."""


# ============================== 1) SCRIPT (Gemini) ==============================
def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a:b + 1])


def check_script(d):
    lines = [str(x).strip() for x in d["lines"] if str(x).strip()]
    if not 4 <= len(lines) <= 14:
        raise ValueError("script has a bad number of lines")
    title = str(d["title"]).strip()[:95]
    desc = str(d.get("description", "")).strip()
    if "#shorts" not in desc.lower():
        desc += "\n\n#shorts"
    tags = [str(t).strip() for t in d.get("tags", []) if str(t).strip()][:12]
    hook = str(d.get("hook_text") or title).strip().upper()[:40]
    return {"title": title, "description": desc, "tags": tags, "hook": hook, "lines": lines}


def make_script(used_titles):
    import requests  # imported here so the rest works without it

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("GEMINI_API_KEY secret is missing")
    prompt = PROMPT.format(
        used="; ".join(used_titles[-40:]) or "none yet",
        today=datetime.date.today().isoformat(),
    )
    last_error = None
    for model in [m for m in MODELS if m]:
        for attempt in range(3):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0},
                    },
                    timeout=120,
                )
            except Exception as e:  # network problem
                last_error = f"{model}: {e}"
                time.sleep(10)
                continue
            if r.status_code == 200:
                try:
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    script = check_script(parse_json(text))
                    print(f"Script made with {model}")
                    return script
                except Exception as e:
                    last_error = f"{model}: bad answer ({e})"
                    time.sleep(5)
                    continue
            last_error = f"{model}: HTTP {r.status_code} {r.text[:200]}"
            if r.status_code in (429, 500, 503):
                time.sleep(20)  # busy / per-minute limit -> wait and retry
                continue
            break  # 400/403/404 -> try next model
    raise SystemExit(f"Gemini failed. Last error: {last_error}")


# ============================== 2) VOICE (free TTS) ==============================
def tts_line(text, mp3_path):
    try:
        import edge_tts

        asyncio.run(edge_tts.Communicate(text, VOICE, rate="-3%").save(str(mp3_path)))
        if Path(mp3_path).stat().st_size > 1000:
            return
    except Exception as e:
        print("edge-tts failed, using backup voice:", e)
    from gtts import gTTS

    gTTS(text, lang=LANG).save(str(mp3_path))


def decode(path):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768


def make_audio(lines, workdir):
    pieces = [np.zeros(int(SR * 0.35), dtype=np.float32)]
    t = 0.35
    timeline = []
    gap = np.zeros(int(SR * 0.30), dtype=np.float32)
    for i, line in enumerate(lines):
        mp3 = Path(workdir) / f"line{i}.mp3"
        tts_line(line, mp3)
        x = decode(mp3)
        timeline.append([t, t + len(x) / SR, line])
        pieces += [x, gap]
        t += len(x) / SR + 0.30
    pieces.append(np.zeros(int(SR * 0.5), dtype=np.float32))
    audio = np.concatenate(pieces)
    # each caption stays until the next one starts
    for i in range(len(timeline) - 1):
        timeline[i][1] = timeline[i + 1][0]
    timeline[-1][1] = len(audio) / SR
    return audio, timeline


def write_wav(path, audio):
    data = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


# ============================== 3) ANIMATION ==============================
def font(size):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def wrap(draw, text, fnt, max_w):
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if draw.textlength(t, font=fnt) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _mask(im):
    return (np.array(im).min(axis=2) < 140).astype(np.float32)


def _best_shift(ref, mov, lim, center=(0, 0)):
    """Where must `mov` be pasted so its dark lines overlap `ref` the most? (FFT trick)"""
    p = 1
    while p < ref.shape[0] + mov.shape[0]:
        p *= 2

    def pad(a):
        z = np.zeros((p, p), np.float32)
        z[: a.shape[0], : a.shape[1]] = a
        return z

    c = np.fft.irfft2(np.fft.rfft2(pad(ref)) * np.conj(np.fft.rfft2(pad(mov))), s=(p, p))
    union = ref.sum() + mov.sum() - c
    cy, cx = center
    best = (-1.0, 0, 0)
    for dy in range(cy - lim, cy + lim + 1):
        for dx in range(cx - lim, cx + lim + 1):
            v = c[dy % p, dx % p] / max(1e-6, union[dy % p, dx % p])
            if v > best[0]:
                best = (v, dx, dy)
    return best


def fit_to(ref_img, im):
    """Resize + move `im` so it sits exactly on top of `ref_img` (no jumping)."""
    size, q = ref_img.size[0], 4
    small = lambda img, s=1.0: img.resize((round(size / q * s), round(size / q * s)), Image.BILINEAR)
    rm = _mask(small(ref_img))
    best = None
    for s in np.arange(0.90, 1.101, 0.02):
        v, dx, dy = _best_shift(rm, _mask(small(im, s)), 36)
        if best is None or v > best[0]:
            best = (v, s, dx, dy)
    _, s, dx, dy = best
    big = im.resize((round(size * s), round(size * s)), Image.LANCZOS)
    _, dx2, dy2 = _best_shift(_mask(ref_img), _mask(big), 10, (dy * q, dx * q))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(big, (dx2, dy2))
    return canvas


def closed_from_open(open_img):
    """Make the closed-mouth face from the open-mouth face (same drawing, only the mouth changes)."""
    size = open_img.size[0]
    a = np.array(open_img).astype(int)
    pink = (a[:, :, 0] > 140) & (a[:, :, 0] - a[:, :, 1] > 35) & (a[:, :, 0] - a[:, :, 2] > 25)
    ys, xs = np.where(pink)
    if len(xs) < 200:
        return None
    cx, cy = np.median(xs), np.median(ys)
    sel = (abs(xs - cx) < 120) & (abs(ys - cy) < 120)
    x0, x1, y0, y1 = xs[sel].min(), xs[sel].max(), ys[sel].min(), ys[sel].max()
    out = open_img.copy()
    d = ImageDraw.Draw(out)
    px, py_top, py_bot = int(0.032 * size), int(0.032 * size), int(0.026 * size)
    d.rectangle((x0 - px, y0 - py_top, x1 + px, y1 + py_bot), fill=(255, 255, 255))
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    half, thick = int(0.03 * size), max(4, int(0.006 * size))
    d.line((mx - half, my, mx + half, my), fill=(25, 25, 25), width=thick)
    return out


def load_character():
    size = CHAR_SIZE
    raw = {}
    for name in ("open", "closed", "blink"):
        files = sorted(ROOT.glob(f"char_{name}.*"))
        if not files:
            raise SystemExit(f"Image char_{name} not found in the repo")
        im = Image.open(files[0]).convert("RGB")
        w, h = im.size
        m = int(min(w, h) * 0.012)  # cut thin border lines
        im = im.crop((m, m, w - m, h - m)).resize((size, size), Image.LANCZOS)
        a = np.array(im)
        a[(a > 238).all(axis=2)] = 255  # near-white -> pure white
        raw[name] = Image.fromarray(a)
    ref = raw["open"]
    closed = closed_from_open(ref)
    if closed is None:  # backup: use your own closed-mouth picture
        closed = fit_to(ref, raw["closed"])
    return {"open": ref, "closed": closed, "blink": fit_to(ref, raw["blink"])}


def build_base(hook):
    base = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(base)
    size = 92
    while True:
        f = font(size)
        rows = wrap(d, hook, f, W - 140)
        if len(rows) <= 3 or size <= 56:
            break
        size -= 8
    y = 90
    for row in rows:
        tw = d.textlength(row, font=f)
        d.text(((W - tw) / 2, y), row, font=f, fill=HOOK_COLOR)
        y += int(size * 1.2)
    d.rectangle(((W - 260) // 2, y + 10, (W + 260) // 2, y + 26), fill=ACCENT)
    return base


def build_panels(timeline):
    ph = H - PANEL_Y
    panels = []
    for i, (_, _, text) in enumerate(timeline):
        img = Image.new("RGB", (W, ph), (0, 0, 0))
        d = ImageDraw.Draw(img)
        size = 74
        while True:
            f = font(size)
            rows = wrap(d, text, f, W - 140)
            if len(rows) * size * 1.25 <= ph - 220 or size <= 44:
                break
            size -= 4
        total = len(rows) * size * 1.25
        y = 140 + (ph - 140 - total) / 2
        color = ACCENT if i == len(timeline) - 1 else (255, 255, 255)
        for row in rows:
            tw = d.textlength(row, font=f)
            d.text(((W - tw) / 2, y), row, font=f, fill=color)
            y += size * 1.25
        panels.append(img)
    return panels, Image.new("RGB", (W, ph), (0, 0, 0))


def render_video(audio, timeline, script, out_path, wav_path):
    chars = load_character()
    base = build_base(script["hook"])
    panels, empty_panel = build_panels(timeline)

    n = int(math.ceil(len(audio) / SR * FPS))
    hop = SR / FPS
    rms = np.array([np.sqrt(np.mean(audio[int(i * hop): int((i + 1) * hop)] ** 2) + 1e-12) for i in range(n)])
    loud = rms[rms > 1e-3]
    ref = np.percentile(loud, 90) if len(loud) else 1.0
    opened = rms > 0.28 * ref
    smooth = opened.copy()
    for i in range(1, n - 1):  # remove 1-frame flicker
        smooth[i] = (int(opened[i - 1]) + int(opened[i]) + int(opened[i + 1])) >= 2
    opened = smooth

    rnd = random.Random(7)
    blink = set()
    t = rnd.uniform(1.0, 2.0)
    while t < n / FPS:
        for k in range(4):
            blink.add(int(t * FPS) + k)
        t += rnd.uniform(2.8, 4.5)

    write_wav(wav_path, audio)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", str(wav_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cx = (W - CHAR_SIZE) // 2
    idx = 0
    for i in range(n):
        sec = i / FPS
        while idx < len(timeline) - 1 and sec >= timeline[idx][1]:
            idx += 1
        panel = panels[idx] if sec >= timeline[0][0] else empty_panel
        if opened[i]:
            face = chars["open"]
        elif i in blink:
            face = chars["blink"]
        else:
            face = chars["closed"]
        dy = int(5 * math.sin(2 * math.pi * sec * 0.6)) - (3 if opened[i] else 0)
        frame = base.copy()
        frame.paste(face, (cx, CHAR_Y + dy))
        frame.paste(panel, (0, PANEL_Y))
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"Video ready: {out_path} ({n / FPS:.1f} sec)")


# ============================== 4) YOUTUBE UPLOAD ==============================
def upload(video_path, script):
    cid = os.environ.get("YT_CLIENT_ID", "").strip()
    sec = os.environ.get("YT_CLIENT_SECRET", "").strip()
    tok = os.environ.get("YT_REFRESH_TOKEN", "").strip()
    if not (cid and sec and tok):
        print("YouTube keys not set -> upload skipped (video is in the artifact).")
        return None
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        None,
        refresh_token=tok,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cid,
        client_secret=sec,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    yt = build("youtube", "v3", credentials=creds)
    privacy = (os.environ.get("PRIVACY") or "private").strip().lower()
    if privacy not in ("private", "unlisted", "public"):
        privacy = "private"
    title = script["title"]
    if "#shorts" not in title.lower() and len(title) <= 90:
        title += " #shorts"
    body = {
        "snippet": {
            "title": title,
            "description": script["description"],
            "tags": script["tags"],
            "categoryId": "23",  # Comedy
        },
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }
    req = yt.videos().insert(
        part="snippet,status", body=body, media_body=MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    )
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    print(f"Uploaded ({privacy}): https://youtube.com/shorts/{resp['id']}")
    return resp["id"]


# ============================== MAIN ==============================
def main():
    OUT.mkdir(exist_ok=True)
    used = USED.read_text(encoding="utf-8").splitlines() if USED.exists() else []
    script = make_script(used)
    print("TITLE:", script["title"])
    for line in script["lines"]:
        print(" -", line)
    with tempfile.TemporaryDirectory() as tmp:
        audio, timeline = make_audio(script["lines"], tmp)
        video = OUT / "video.mp4"
        render_video(audio, timeline, script, video, Path(tmp) / "voice.wav")
    (OUT / "meta.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    upload(video, script)
    with USED.open("a", encoding="utf-8") as f:
        f.write(script["title"] + "\n")


if __name__ == "__main__":
    main()
