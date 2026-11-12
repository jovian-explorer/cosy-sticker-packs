#!/usr/bin/env python3
"""Add sticker packs to catalog.json.

    python tools/add_pack.py <folder-of-webp | file.wastickers> "Pack name"

Validates every sticker against WhatsApp's limits, re-encodes the ones that fail
(Pillow, ffmpeg as fallback for animated files), splits them into packs of at most
30 (static and animated kept apart), makes a 96x96 tray icon per pack and updates
plain/catalog.json. Then it encrypts everything (AES-256-GCM, key in tools/packs.key)
into <name>.enc files in the repo root plus index.enc. Commit the .enc files and push.
plain/ and tools/packs.key are gitignored and never leave the PC. Other commands:

    python tools/add_pack.py --refresh     re-hash all pack files, bump image_data_version of changed packs
    python tools/add_pack.py --encrypt     only re-encrypt plain/ into the .enc files
    python tools/add_pack.py --check       validate plain/, change nothing

Needs: pip install pillow cryptography. ffmpeg on PATH is optional.
The key is 32 random bytes, base64, one line, in tools/packs.key (or env COSY_PACKS_KEY).
"""
import argparse
import base64
import hashlib
import hmac
import os
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image, ImageSequence

SIZE = 512
STATIC_MAX = 100 * 1024
ANIM_MAX = 500 * 1024
FRAME_MIN_MS = 8
ANIM_TOTAL_MAX_MS = 10_000
PACK_MAX = 30
PACK_MIN = 3
TRAY_SIZE = 96
TRAY_MAX = 50 * 1024
DEFAULT_EMOJI = "\U0001F642"

ROOT = Path(__file__).resolve().parent.parent
PLAIN = ROOT / "plain"
KEY_FILE = ROOT / "tools" / "packs.key"
INDEX = "index.enc"


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]


def git_blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def frames_of(img):
    """Return (list of RGBA frames, list of durations ms)."""
    frames, durs = [], []
    for f in ImageSequence.Iterator(img):
        frames.append(f.convert("RGBA"))
        durs.append(int(f.info.get("duration", 100)))
    return frames, durs


def inspect(data):
    """Return (animated, problems) for one webp byte string."""
    problems = []
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # not a readable image at all
        return False, ["unreadable: %s" % e]
    if img.format != "WEBP":
        problems.append("not a webp (%s)" % img.format)
    animated = getattr(img, "n_frames", 1) > 1
    if img.size != (SIZE, SIZE):
        problems.append("size %dx%d, need 512x512" % img.size)
    if animated:
        if len(data) > ANIM_MAX:
            problems.append("%d KB, animated limit 500 KB" % (len(data) // 1024))
        _, durs = frames_of(img)
        if min(durs) < FRAME_MIN_MS:
            problems.append("frame duration under 8 ms")
        if sum(durs) > ANIM_TOTAL_MAX_MS:
            problems.append("animation longer than 10 s")
    elif len(data) > STATIC_MAX:
        problems.append("%d KB, static limit 100 KB" % (len(data) // 1024))
    return animated, problems


def fit_canvas(frame, shrink=1.0):
    """Scale to fit 512x512 keeping aspect ratio, centred on a transparent canvas.
    shrink < 1 makes the artwork smaller inside the 512x512 frame (last resort for size)."""
    frame = frame.convert("RGBA")
    scale = min(SIZE / frame.width, SIZE / frame.height) * shrink
    w, h = max(1, round(frame.width * scale)), max(1, round(frame.height * scale))
    frame = frame.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.paste(frame, ((SIZE - w) // 2, (SIZE - h) // 2), frame)
    return canvas


def encode_static(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img.seek(0)
    for shrink in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        canvas = fit_canvas(img, shrink)
        for q in (90, 80, 70, 60, 50, 40, 30, 20, 10):
            buf = io.BytesIO()
            canvas.save(buf, "WEBP", quality=q, alpha_quality=min(q, 80), method=6)
            if buf.tell() <= STATIC_MAX:
                return buf.getvalue()
    raise ValueError("cannot get static sticker under 100 KB")


def encode_animated_pillow(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    frames, durs = frames_of(img)
    durs = [max(FRAME_MIN_MS, d) for d in durs]
    # drop frames from the end until the loop is 10 s or shorter
    while sum(durs) > ANIM_TOTAL_MAX_MS and len(frames) > 2:
        frames.pop()
        durs.pop()
    for shrink in (1.0, 0.8, 0.6):
        fitted = [fit_canvas(f, shrink) for f in frames]
        for q in (80, 65, 50, 40, 30, 20, 10):
            buf = io.BytesIO()
            fitted[0].save(buf, "WEBP", save_all=True, append_images=fitted[1:],
                           duration=durs, loop=0, quality=q, alpha_quality=min(q, 80), method=4)
            if buf.tell() <= ANIM_MAX:
                return buf.getvalue()
    raise ValueError("cannot get animated sticker under 500 KB")


def encode_animated_ffmpeg(src_path):
    if not shutil.which("ffmpeg"):
        raise ValueError("ffmpeg not on PATH")
    for fps, q in ((15, 60), (12, 45), (10, 30), (8, 20)):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.webp"
            vf = ("fps=%d,scale=512:512:force_original_aspect_ratio=decrease:flags=lanczos,"
                  "pad=512:512:(ow-iw)/2:(oh-ih)/2:color=0x00000000,format=yuva420p" % fps)
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src_path), "-t", "9.9", "-vf", vf,
                   "-c:v", "libwebp_anim", "-loop", "0", "-q:v", str(q), "-lossless", "0", str(out)]
            if subprocess.run(cmd).returncode == 0 and out.exists() and out.stat().st_size <= ANIM_MAX:
                return out.read_bytes()
    raise ValueError("ffmpeg could not get the file under the limits")


def fix(name, data, src_path, animated):
    """Re-encode one failing sticker; returns new bytes."""
    if not animated:
        return encode_static(data)
    try:
        out = encode_animated_pillow(data)
        if not inspect(out)[1]:
            return out
    except Exception as e:
        print("    pillow failed (%s), trying ffmpeg" % e)
    return encode_animated_ffmpeg(src_path)


def tray_png(data):
    img = Image.open(io.BytesIO(data))
    img.seek(0)
    img = img.convert("RGBA").resize((TRAY_SIZE, TRAY_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    if buf.tell() > TRAY_MAX:
        raise ValueError("tray icon over 50 KB")
    return buf.getvalue()


def chunk(items):
    """Split into groups of at most 30, never leaving a group under 3."""
    n = len(items)
    if n == 0:
        return []
    if n < PACK_MIN:
        raise ValueError("need at least %d stickers of one kind, got %d" % (PACK_MIN, n))
    groups = -(-n // PACK_MAX)
    base, extra = divmod(n, groups)
    out, i = [], 0
    for g in range(groups):
        k = base + (1 if g < extra else 0)
        out.append(items[i:i + k])
        i += k
    return out


def load_catalog():
    path = PLAIN / "catalog.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"sticker_packs": []}


def save_catalog(cat):
    text = json.dumps(cat, ensure_ascii=False, indent=1) + "\n"
    PLAIN.mkdir(exist_ok=True)
    tmp = PLAIN / "catalog.json.tmp"
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(PLAIN / "catalog.json")


def collect(src, workdir):
    """Return a sorted list of (name, path) of webp files from a folder or .wastickers."""
    src = Path(src)
    if src.is_file():
        if src.suffix.lower() not in (".wastickers", ".zip"):
            sys.exit("not a folder, .wastickers or .zip: %s" % src)
        out = Path(workdir) / "unzipped"
        with zipfile.ZipFile(src) as z:
            z.extractall(out)
        src = out
    files = [p for p in src.rglob("*.webp") if p.is_file()]
    # a .wastickers also holds a tray image; its tray is a webp named like the pack icon
    files = [p for p in files if not re.fullmatch(r"(tray|try)[^/]*", p.stem, re.I)]
    if not files:
        sys.exit("no .webp files found in %s" % src)
    return sorted(files, key=natural_key)


def next_identifier(cat):
    nums = [int(p["identifier"]) for p in cat["sticker_packs"] if p["identifier"].isdigit()]
    return max(nums, default=0) + 1


def add(src, name, publisher, emoji, dry):
    if len(name) > 100:
        sys.exit("pack name too long")
    cat = load_catalog()
    with tempfile.TemporaryDirectory() as work:
        files = collect(src, work)
        static, anim = [], []
        report = {"ok": 0, "fixed": 0}
        for p in files:
            data = p.read_bytes()
            animated, problems = inspect(data)
            if problems:
                print("  fix  %s: %s" % (p.name, "; ".join(problems)))
                try:
                    data = fix(p.name, data, p, animated)
                except Exception as e:
                    sys.exit("  FAILED %s: %s" % (p.name, e))
                animated, problems = inspect(data)
                if problems:
                    sys.exit("  FAILED %s after re-encoding: %s" % (p.name, "; ".join(problems)))
                report["fixed"] += 1
            else:
                report["ok"] += 1
            (anim if animated else static).append((p.name, data))
        print("%d valid, %d re-encoded; %d static, %d animated" %
              (report["ok"], report["fixed"], len(static), len(anim)))

        kinds = [(g, False) for g in chunk(static)] + [(g, True) for g in chunk(anim)]
        # a pack name carries a number only when the upload became more than one pack
        pack_names = [name] if len(kinds) == 1 else ["%s %d" % (name, i + 1) for i in range(len(kinds))]
        ident = next_identifier(cat)
        for (group, animated), pname in zip(kinds, pack_names):
            pid = str(ident)
            ident += 1
            entry = {
                "identifier": pid,
                "name": pname,
                "publisher": publisher,
                "animated_sticker_pack": animated,
                "tray_image_file": "tray.png",
                "tray_sha": git_blob_sha(tray_png(group[0][1])),
                "image_data_version": "1",
                "stickers": [],
            }
            tray = tray_png(group[0][1])
            out_dir = PLAIN / pid
            if not dry:
                out_dir.mkdir(parents=True, exist_ok=False)
                (out_dir / "tray.png").write_bytes(tray)
            for i, (_, data) in enumerate(group, 1):
                fname = "s%02d.webp" % i
                if not dry:
                    (out_dir / fname).write_bytes(data)
                entry["stickers"].append({"image_file": fname, "emojis": [emoji], "sha": git_blob_sha(data)})
            cat["sticker_packs"].append(entry)
            print("pack %s: %s, %d %s stickers" % (pid, pname, len(group), "animated" if animated else "static"))
    if dry:
        print("dry run, nothing written")
    else:
        save_catalog(cat)
        encrypt()
        print("Done. Next: git add -A && git commit -m 'Add %s' && git push" % name)


def refresh():
    cat = load_catalog()
    for p in cat["sticker_packs"]:
        d = PLAIN / p["identifier"]
        changed = False
        tray = git_blob_sha((d / p["tray_image_file"]).read_bytes())
        if p.get("tray_sha") not in (None, tray):
            changed = True
        p["tray_sha"] = tray
        for s in p["stickers"]:
            sha = git_blob_sha((d / s["image_file"]).read_bytes())
            if s.get("sha") not in (None, sha):
                changed = True
            s["sha"] = sha
        if changed:
            p["image_data_version"] = str(int(p["image_data_version"]) + 1)
            print("pack %s changed, image_data_version -> %s" % (p["identifier"], p["image_data_version"]))
    save_catalog(cat)
    encrypt()


def load_key():
    raw = os.environ.get("COSY_PACKS_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not raw:
        sys.exit("no key: put the base64 key in tools/packs.key (ask the owner) or set COSY_PACKS_KEY")
    key = base64.b64decode(raw)
    if len(key) != 32:
        sys.exit("key must be 32 bytes")
    return key


def seal(key, data):
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, data, None)


def unseal(key, blob):
    return AESGCM(key).decrypt(blob[:12], blob[12:], None)


def enc_name(key, rel, sha):
    """Opaque file name; changes whenever the content changes, so a stale CDN copy is never mixed in."""
    return hmac.new(key, ("%s:%s" % (rel, sha)).encode(), hashlib.sha256).hexdigest()[:40] + ".enc"


def encrypt():
    """Make the .enc files and index.enc in the repo root from plain/. Unchanged files keep their bytes."""
    key = load_key()
    files = {}
    for f in sorted(PLAIN.rglob("*")):
        if f.is_file() and not f.name.endswith(".tmp"):
            rel = f.relative_to(PLAIN).as_posix()
            data = f.read_bytes()
            name = enc_name(key, rel, git_blob_sha(data))
            files[rel] = name
            if not (ROOT / name).exists():
                (ROOT / name).write_bytes(seal(key, data))
    for old in ROOT.glob("*.enc"):
        if old.name != INDEX and old.name not in files.values():
            old.unlink()
    index = json.dumps({"version": 1, "files": files}, sort_keys=True).encode()
    same = False
    if (ROOT / INDEX).exists():
        try:
            same = unseal(key, (ROOT / INDEX).read_bytes()) == index
        except Exception:
            pass
    if not same:
        (ROOT / INDEX).write_bytes(seal(key, index))
    print("encrypted %d files, index %s" % (len(files), "unchanged" if same else "updated"))


def check():
    cat = load_catalog()
    bad = 0
    for p in cat["sticker_packs"]:
        d = PLAIN / p["identifier"]
        n = len(p["stickers"])
        if not PACK_MIN <= n <= PACK_MAX:
            print("pack %s: %d stickers" % (p["identifier"], n))
            bad += 1
        try:
            t = Image.open(d / p["tray_image_file"])
            if not (24 <= t.width <= 512 and 24 <= t.height <= 512) or (d / p["tray_image_file"]).stat().st_size > TRAY_MAX:
                print("pack %s: bad tray" % p["identifier"])
                bad += 1
        except Exception as e:
            print("pack %s: tray unreadable (%s)" % (p["identifier"], e))
            bad += 1
        for s in p["stickers"]:
            data = (d / s["image_file"]).read_bytes()
            animated, problems = inspect(data)
            if animated != p["animated_sticker_pack"]:
                problems.append("animated flag mismatch")
            if git_blob_sha(data) != s.get("sha"):
                problems.append("sha differs from catalog (run --refresh)")
            if not 1 <= len(s["emojis"]) <= 3:
                problems.append("needs 1 to 3 emojis")
            if problems:
                bad += 1
                print("pack %s %s: %s" % (p["identifier"], s["image_file"], "; ".join(problems)))
    print("%d packs, %d stickers, %d problems" %
          (len(cat["sticker_packs"]), sum(len(p["stickers"]) for p in cat["sticker_packs"]), bad))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="folder of .webp files or a .wastickers file")
    ap.add_argument("name", nargs="?", help="pack name")
    ap.add_argument("--publisher", default="Keshav")
    ap.add_argument("--emoji", default=DEFAULT_EMOJI, help="emoji given to every sticker (edit catalog.json later for per-sticker emoji)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--encrypt", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if check() else 0)
    if a.encrypt:
        return encrypt()
    if a.refresh:
        return refresh()
    if not a.source or not a.name:
        ap.error("give a source and a pack name")
    add(a.source, a.name, a.publisher, a.emoji, a.dry_run)


if __name__ == "__main__":
    main()
