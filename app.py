import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
import webbrowser
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE = Path(__file__).parent
CACHE = BASE / "cache"
EXPORTS = BASE / "exports"
CACHE.mkdir(exist_ok=True)
EXPORTS.mkdir(exist_ok=True)
CACHE_VER = 2
SPK_FILE = CACHE / "speakers.json"
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg", ".ts"}
ESPEAK_LANG = {"fr": "fr-fr", "en": "en-us", "es": "es", "de": "de", "it": "it", "pt": "pt", "nl": "nl"}
VOWELS = set("aeiouyɑɛɔøœəɐɪʊɨɯ")
IPA_DROP = str.maketrans("", "", "ˈˌː‿͡.‖| ")
IPA_MAP = str.maketrans({"ɑ": "a", "ɡ": "g", "ʁ": "r"})
PORT = 8765
SR = 16000

HALLU = re.compile(r"sous-?titr|amara\.org|radio-canada|merci d'avoir regard|abonnez-vous", re.I)
VAD = dict(threshold=0.4, min_silence_duration_ms=300, speech_pad_ms=250)

ALIGN_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
FRAME = 0.02
CHUNK_MAX = 20.0
CHUNK_GAP = 1.5
CHUNK_PAD = 0.6
JOIN_GAP = 0.16
LEAD_IN = 0.08
LEAD_OUT = 0.08
BAND_SLACK = 0.6
POOLS = {
    "a": "a ɑ æ", "ɑ": "a ɑ", "e": "e ɛ", "ɛ": "ɛ e æ", "i": "i ɪ", "u": "u ʊ", "o": "o ɔ", "ɔ": "ɔ o", "y": "y ʏ",
    "ə": "ə ø œ ɵ", "ø": "ø ə œ y", "œ": "œ ə ø ɛ", "ʁ": "ʁ r x ɾ ɹ χ",
    "ɔ̃": "ɔ̃ õ o ɔ", "ɑ̃": "ɑ̃ ã ɔ̃ a ɑ", "ɛ̃": "ɛ̃ ẽ ɑ̃ ã a ɛ æ", "œ̃": "œ̃ ɛ̃ ẽ a ɛ",
}
WORD_GAP = 0.4
ENV_WIN = 320
ENV_HOP = 80
SNAP_MAX = 0.3
SNAP_DB = 8.0
MIN_CONF = 0.0

SPK_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
SPK_ONNX = "voxceleb_resnet34_LM.onnx"
TURN_GAP = 0.35
TURN_MAX = 8.0
TURN_MIN_EMB = 0.5
SPK_LONG = 1.2
SPK_THR = 0.8

if sys.platform == "win32" and not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
    for cand in (r"C:\Program Files\eSpeak NG\libespeak-ng.dll", r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll"):
        if os.path.exists(cand):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = cand
            break

try:
    import torch
except Exception as e:
    print("torch unavailable:", e)

LOCK = threading.Lock()
VIDEOS = {}
BACKENDS = {}
MODEL = {"key": None, "obj": None}
ALIGNER = {}
EMBEDDER = {}
SPK = {"thr": SPK_THR, "names": {}, "labels": {}}
JOB = {"running": False, "done": 0, "total": 0, "current": "", "stage": "", "errors": [], "ipa_ok": True}

app = FastAPI()


def norm_text(s):
    s = s.lower().replace("œ", "oe").replace("æ", "ae")
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z0-9]", "", s)


def norm_ipa(s):
    s = unicodedata.normalize("NFC", s).translate(IPA_DROP).translate(IPA_MAP)
    return unicodedata.normalize("NFC", s)


def clean_token(w):
    return re.sub(r"[^\w'’-]", "", w.lower()).strip("-")


def get_backend(code):
    if code in BACKENDS:
        return BACKENDS[code]
    try:
        from phonemizer.backend import EspeakBackend
        be = EspeakBackend(code, with_stress=False, language_switch="remove-flags", preserve_punctuation=False)
    except Exception as e:
        print("Phonemizer unavailable:", e)
        be = None
        JOB["ipa_ok"] = False
    BACKENDS[code] = be
    return be


def phonemize_phones(tokens, lang):
    be = get_backend(ESPEAK_LANG.get(lang or "fr", "fr-fr"))
    if be is None:
        return None
    from phonemizer.separator import Separator
    idx = [i for i, t in enumerate(tokens) if t]
    out = [[] for _ in tokens]
    if not idx:
        return out
    with LOCK:
        res = be.phonemize([tokens[i] for i in idx], separator=Separator(phone=" ", word="|"), strip=True)
    for i, r in zip(idx, res):
        out[i] = r.replace("|", " ").split()
    return out


def phonemize_query(q, lang):
    tokens = [clean_token(t) for t in q.split()]
    phones = phonemize_phones(tokens, lang)
    if phones is None:
        return ""
    return "".join(norm_ipa("".join(p)) for p in phones)


def looks_like_ipa(q):
    return any(ord(c) > 0x024F or unicodedata.combining(c) for c in q)


def build_text_stream(words):
    s, t0, t1, wid, ws, we, cf = [], [], [], [], [], [], []
    for i, w in enumerate(words):
        txt = w["t"]
        if not txt:
            continue
        weights = [2.0 if ch in VOWELS else 1.0 for ch in txt]
        total = sum(weights)
        dur = w["t1"] - w["t0"]
        acc = 0.0
        for k, (ch, wt) in enumerate(zip(txt, weights)):
            a = w["t0"] + dur * acc / total
            acc += wt
            s.append(ch)
            t0.append(a)
            t1.append(w["t0"] + dur * acc / total)
            wid.append(i)
            ws.append(k == 0)
            we.append(k == len(txt) - 1)
            cf.append(w["c"])
    return {"s": "".join(s), "t0": t0, "t1": t1, "wid": wid, "ws": ws, "we": we, "c": cf}


def build_ipa_stream(words):
    s, t0, t1, wid, ws, we, cf = [], [], [], [], [], [], []
    for i, w in enumerate(words):
        chars = [(ch, a, b, c) for sym, a, b, c in w["ph"] for ch in norm_ipa(sym)]
        for k, (ch, a, b, c) in enumerate(chars):
            s.append(ch)
            t0.append(a)
            t1.append(b)
            wid.append(i)
            ws.append(k == 0)
            we.append(k == len(chars) - 1)
            cf.append(c)
    return {"s": "".join(s), "t0": t0, "t1": t1, "wid": wid, "ws": ws, "we": we, "c": cf}


def load_video(vid, path, data):
    words = []
    for w in data["words"]:
        t0 = float(w["t0"])
        t1 = max(float(w["t1"]), t0 + 0.04)
        words.append({"w": w["w"], "t0": t0, "t1": t1, "t": norm_text(w["w"]), "c": float(w.get("c", 1.0)), "ph": w.get("ph", []), "turn": -1})
    turns = data.get("turns", [])
    for k, (a, b) in enumerate(turns):
        for w in words[a:b + 1]:
            w["turn"] = k
    ef = CACHE / f"{vid}.npy"
    emb = np.load(ef) if ef.exists() else np.zeros((0, 256), np.float32)
    VIDEOS[vid] = {
        "id": vid,
        "path": str(path),
        "name": Path(path).name,
        "words": words,
        "turns": turns,
        "emb": emb,
        "text": build_text_stream(words),
        "ipa": build_ipa_stream(words),
    }


def vid_id(p):
    st = p.stat()
    return hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-800:])


def get_model(size):
    if MODEL["key"] != size:
        from faster_whisper import WhisperModel
        MODEL["obj"] = WhisperModel(size, device="auto", compute_type="auto")
        MODEL["key"] = size
    return MODEL["obj"]


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def merge_words(raw):
    out = []
    for w in raw:
        txt = w.word.strip()
        if not txt:
            continue
        if out and (txt[0] in "'’" or out[-1]["w"][-1] in "'’" or not re.search(r"\w", txt)):
            out[-1]["w"] += txt
            out[-1]["t1"] = round(w.end, 3)
        else:
            out.append({"w": txt, "t0": round(w.start, 3), "t1": round(w.end, 3)})
    return [w for w in out if re.search(r"\w", w["w"])]


def transcribe(model, audio, lang):
    segs, info = model.transcribe(
        audio,
        language=None if lang == "auto" else lang,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=VAD,
        condition_on_previous_text=False,
        hallucination_silence_threshold=1.5,
        beam_size=5,
    )
    out = []
    for seg in segs:
        if HALLU.search(seg.text):
            continue
        words = merge_words(seg.words or [])
        if words:
            out.append({"t0": seg.start, "t1": seg.end, "words": words})
    return out, info.language


def get_aligner():
    if not ALIGNER:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        vocab = json.loads(Path(hf_hub_download(ALIGN_MODEL, "vocab.json")).read_text(encoding="utf-8"))
        ALIGNER.update(torch=torch, dev=dev, vocab=vocab, blank=vocab["<pad>"], model=Wav2Vec2ForCTC.from_pretrained(ALIGN_MODEL).to(dev).eval())
    return ALIGNER


def phone_ids(ph, vocab):
    for cand in (ph, ph.replace("ː", "")):
        if cand in vocab:
            fam = POOLS.get(cand.replace("ː", ""), "").split()
            ids = [vocab[cand]] + [vocab[x] for f in fam for x in (f, f + "ː") if x in vocab and x != cand]
            return [tuple(ids)]
    return [(vocab[c],) for c in ph if c in vocab]


def emissions(al, x):
    torch = al["torch"]
    x = (x - x.mean()) / (x.std() + 1e-7)
    with torch.inference_mode():
        logits = al["model"](torch.from_numpy(x)[None].to(al["dev"])).logits[0]
    return torch.log_softmax(logits.float(), -1).cpu().numpy()


def ctc_path(lp, toks, blank, bands=None):
    T, S = lp.shape[0], len(toks)
    if S == 0 or T < S:
        return None
    ext = np.full(2 * S + 1, blank)
    ext[1::2] = [ids[0] for ids in toks]
    N = len(ext)
    em = lp[:, ext].astype(np.float64)
    cols = {}
    for k, ids in enumerate(toks):
        if len(ids) > 1:
            if ids not in cols:
                cols[ids] = np.logaddexp.reduce(lp[:, list(ids)].astype(np.float64), axis=1)
            em[:, 2 * k + 1] = cols[ids]
    neg = -1e30
    if bands is not None:
        for k, spans in enumerate(bands):
            ok = np.zeros(T, bool)
            for a, b in spans:
                ok[max(0, a):max(0, b)] = True
            em[~ok, 2 * k + 1] = neg
    skip = np.zeros(N, bool)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    d = np.full(N, neg)
    d[0], d[1] = em[0, 0], em[0, 1]
    bp = np.zeros((T, N), np.int8)
    rng = np.arange(N)
    for t in range(1, T):
        cand = np.stack([d, np.concatenate(([neg], d[:-1])), np.where(skip, np.concatenate(([neg, neg], d[:-2])), neg)])
        k = cand.argmax(0)
        d = cand[k, rng] + em[t]
        bp[t] = k
    s = N - 1 if d[N - 1] >= d[N - 2] else N - 2
    if d[s] < neg / 2:
        return ctc_path(lp, toks, blank) if bands is not None else None
    path = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path, em


def chunk_segments(segs):
    chunks, cur = [], []
    for s in segs:
        if cur and (s["t0"] - cur[-1]["t1"] > CHUNK_GAP or s["t1"] - cur[0]["t0"] > CHUNK_MAX):
            chunks.append(cur)
            cur = []
        cur.append(s)
    if cur:
        chunks.append(cur)
    return chunks


def word_band(w, lo):
    a, b = w["t0"] - lo, w["t1"] - lo
    d = 0.3 + 0.12 * len(w["_ph"])
    spans = [(a - BAND_SLACK, a + d), (b - d, b + BAND_SLACK)] if b - a > d + 0.5 else [(a - BAND_SLACK, b + BAND_SLACK)]
    return [(int(x / FRAME), int(y / FRAME) + 1) for x, y in spans]


def envelope(audio):
    e = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))
    idx = np.arange(0, max(1, len(audio) - ENV_WIN), ENV_HOP)
    return 10 * np.log10((e[np.minimum(idx + ENV_WIN, len(audio))] - e[idx]) / ENV_WIN + 1e-10)


def snap_start(db, t, limit):
    i1 = int(t * SR / ENV_HOP)
    i0 = int(max(limit, t - SNAP_MAX) * SR / ENV_HOP)
    seg = db[i0:i1]
    if len(seg) < 3:
        return max(limit, t - LEAD_IN)
    level, floor = db[max(0, i1 - 4):i1 + 12].max(), seg.min()
    if level - floor < SNAP_DB:
        return max(limit, t - LEAD_IN)
    quiet = np.flatnonzero(seg <= floor + 0.3 * (level - floor))
    return max(limit, (i0 + quiet[-1]) * ENV_HOP / SR)


def snap_end(db, t, limit):
    i0 = int(t * SR / ENV_HOP)
    i1 = int(min(limit, t + SNAP_MAX) * SR / ENV_HOP)
    seg = db[i0:i1]
    if len(seg) < 3:
        return min(limit, t + LEAD_OUT)
    level, floor = db[max(0, i0 - 12):i0 + 4].max(), seg.min()
    if level - floor < SNAP_DB:
        return min(limit, t + LEAD_OUT)
    quiet = np.flatnonzero(seg <= floor + 0.3 * (level - floor))
    return min(limit, (i0 + quiet[0]) * ENV_HOP / SR + ENV_WIN / SR)


def align(audio, segs, lang):
    words = [w for s in segs for w in s["words"]]
    phones = phonemize_phones([clean_token(w["w"]) for w in words], lang)
    if phones is None:
        for w in words:
            w.update(c=1.0, ph=[])
        return False
    al = get_aligner()
    db = envelope(audio)
    for w, ph in zip(words, phones):
        w["_ph"] = ph
    floor = 0.0
    for chunk in chunk_segments(segs):
        cw = [w for s in chunk for w in s["words"]]
        lo = max(0.0, chunk[0]["t0"] - CHUNK_PAD, floor)
        hi = min(len(audio) / SR, chunk[-1]["t1"] + CHUNK_PAD)
        toks, owner, bands = [], [], []
        for wi, w in enumerate(cw):
            band = word_band(w, lo)
            for pi, ph in enumerate(w["_ph"]):
                for tid in phone_ids(ph, al["vocab"]):
                    toks.append(tid)
                    owner.append((wi, pi))
                    bands.append(band)
        res = ctc_path(emissions(al, audio[int(lo * SR):int(hi * SR)]), toks, al["blank"], bands) if hi - lo > 0.1 else None
        if res is None:
            for w in cw:
                w.update(c=0.0, ph=[])
            continue
        path, em = res
        spans = {}
        for k, key in enumerate(owner):
            fr = np.flatnonzero(path == 2 * k + 1)
            sc = float(np.exp(em[fr, 2 * k + 1]).mean())
            a, b = lo + fr[0] * FRAME, lo + (fr[-1] + 1) * FRAME
            if key in spans:
                spans[key] = [spans[key][0], b, (spans[key][2] + sc) / 2]
            else:
                spans[key] = [a, b, sc]
        stray = []
        for wi, w in enumerate(cw):
            keys = [(wi, pi) for pi in range(len(w["_ph"])) if (wi, pi) in spans]
            runs = [[k] for k in keys[:1]]
            for p, n in zip(keys, keys[1:]):
                if spans[n][0] - spans[p][1] > WORD_GAP:
                    runs.append([])
                runs[-1].append(n)
            if len(runs) > 1:
                main = max(runs, key=lambda r: (len(r), sum(spans[k][2] for k in r)))
                stray += [(k, main, r[0] < main[0]) for r in runs if r is not main for k in r]
        gone = {k for k, _, _ in stray}
        order = [v for k, v in spans.items() if k not in gone]
        raw = [(v[0], v[1]) for v in order]
        for i in range(len(order) - 1):
            p, n = order[i], order[i + 1]
            if raw[i + 1][0] - raw[i][1] <= JOIN_GAP:
                p[1] = n[0] = (raw[i][1] + raw[i + 1][0]) / 2
            else:
                p[1] = snap_end(db, raw[i][1], raw[i + 1][0])
                n[0] = snap_start(db, raw[i + 1][0], raw[i][1])
                if p[1] > n[0]:
                    p[1] = n[0] = (p[1] + n[0]) / 2
        order[0][0] = snap_start(db, raw[0][0], lo)
        order[-1][1] = snap_end(db, raw[-1][1], hi)
        floor = order[-1][1]
        for k, main, before in stray:
            edge = spans[main[0]][0] if before else spans[main[-1]][1]
            spans[k] = [edge, edge, 0.0]
        for wi, w in enumerate(cw):
            ph = [[sym, *np.round(spans[(wi, pi)], 3).tolist()] for pi, sym in enumerate(w["_ph"]) if (wi, pi) in spans]
            if ph:
                w.update(t0=ph[0][1], t1=ph[-1][2], c=round(float(np.mean([p[3] for p in ph])), 3), ph=ph)
            else:
                w.update(c=0.0, ph=[])
    for w in words:
        w.pop("_ph", None)
    return True


def mel_bank():
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    pts = np.linspace(mel(20.0), mel(SR / 2), 82)
    fm = mel(np.arange(257) * SR / 512.0)
    fb = np.zeros((80, 257))
    for i in range(80):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((fm - l) / (c - l), (r - fm) / (r - c)))
    fb[:, 256] = 0.0
    return fb


def fbank(x):
    x = x.astype(np.float64) * 32768.0
    n = 1 + (len(x) - 400) // 160
    fr = x[np.arange(400)[None] + 160 * np.arange(n)[:, None]]
    fr = fr - fr.mean(1, keepdims=True)
    fr = np.concatenate([fr[:, :1] * 0.03, fr[:, 1:] - 0.97 * fr[:, :-1]], 1) * np.hamming(400)
    feats = np.log(np.maximum(np.abs(np.fft.rfft(fr, 512)) ** 2 @ EMBEDDER["fb"].T, 1.1920929e-07))
    return (feats - feats.mean(0)).astype(np.float32)


def get_embedder():
    if not EMBEDDER:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        opt = ort.SessionOptions()
        opt.log_severity_level = 3
        sess = ort.InferenceSession(hf_hub_download(SPK_REPO, SPK_ONNX), opt, providers=["CPUExecutionProvider"])
        EMBEDDER.update(sess=sess, inp=sess.get_inputs()[0].name, fb=mel_bank())
    return EMBEDDER


def make_turns(words):
    turns, start = [], 0
    for i in range(1, len(words) + 1):
        if i == len(words) or words[i]["t0"] - words[i - 1]["t1"] > TURN_GAP or words[i - 1]["w"][-1] in ".?!…" or words[i]["t1"] - words[start]["t0"] > TURN_MAX:
            turns.append([start, i - 1])
            start = i
    return turns


def embed_turns(audio, words, turns):
    em = get_embedder()
    out = np.zeros((len(turns), 256), np.float32)
    for k, (a, b) in enumerate(turns):
        t0, t1 = words[a]["t0"], words[b]["t1"]
        if t1 - t0 < TURN_MIN_EMB:
            mid = (t0 + t1) / 2
            t0, t1 = mid - TURN_MIN_EMB / 2, mid + TURN_MIN_EMB / 2
        seg = audio[max(0, int(t0 * SR)):int(t1 * SR)]
        if len(seg) < 400:
            continue
        e = em["sess"].run(None, {em["inp"]: fbank(seg)[None]})[0][0]
        out[k] = e / (np.linalg.norm(e) + 1e-9)
    return out


def is_custom(name):
    return not re.fullmatch(r"Voice \d+", name)


def save_speakers():
    SPK_FILE.write_text(json.dumps({"thr": SPK["thr"], "names": {str(k): v for k, v in SPK["names"].items()}, "labels": SPK["labels"]}, ensure_ascii=False), encoding="utf-8")


def load_speakers():
    if SPK_FILE.exists():
        d = json.loads(SPK_FILE.read_text(encoding="utf-8"))
        SPK.update(thr=d["thr"], names={int(k): v for k, v in d["names"].items()}, labels=d["labels"])


def labels_ok():
    return all(len(SPK["labels"].get(vid, [])) == len(v["turns"]) for vid, v in VIDEOS.items())


def recluster(thr=None):
    from scipy.cluster.hierarchy import fcluster, linkage
    if thr is not None:
        SPK["thr"] = thr
    items = [(vid, k) for vid, v in VIDEOS.items() for k in range(len(v["turns"]))]
    if not items:
        SPK.update(names={}, labels={})
        save_speakers()
        return
    E = np.concatenate([v["emb"] for v in VIDEOS.values() if len(v["turns"])])
    dur = np.array([VIDEOS[vid]["words"][VIDEOS[vid]["turns"][k][1]]["t1"] - VIDEOS[vid]["words"][VIDEOS[vid]["turns"][k][0]]["t0"] for vid, k in items])
    old = [SPK["labels"].get(vid, [None] * (k + 1))[k] if k < len(SPK["labels"].get(vid, [])) else None for vid, k in items]
    long = np.flatnonzero((dur >= SPK_LONG) & (np.abs(E).sum(1) > 0))
    if len(long) < 2:
        long = np.arange(len(items))
    if len(long) > 5000:
        long = long[np.argsort(-dur[long])[:5000]]
    lab = fcluster(linkage(E[long], "average", "cosine"), SPK["thr"], "distance") if len(long) > 1 else np.ones(len(long), int)
    groups = [long[lab == g] for g in np.unique(lab)]
    kept = [g for g in groups if len(g) >= 3 and dur[g].sum() >= 4.0] or [max(groups, key=len)]
    cents = np.stack([E[g].mean(0) / (np.linalg.norm(E[g].mean(0)) + 1e-9) for g in kept])
    assign = (E @ cents.T).argmax(1)
    for gi, g in enumerate(kept):
        assign[g] = gi
    by_name, names, remap, n = {}, {}, {}, 0
    for gi in np.argsort([-dur[assign == g].sum() for g in range(len(kept))]):
        members = np.flatnonzero(assign == gi)
        votes = Counter(old[m] for m in members if old[m] is not None)
        name = None
        if votes:
            oid, cnt = votes.most_common(1)[0]
            oname = SPK["names"].get(oid)
            if oname and is_custom(oname) and cnt >= 0.5 * len(members):
                name = oname
        if name and name in by_name:
            remap[gi] = by_name[name]
            continue
        sid = len(names) + 1
        if not name:
            n += 1
            name = f"Voice {n}"
        names[sid] = name
        by_name[name] = sid
        remap[gi] = sid
    labels = {}
    for (vid, k), g in zip(items, assign):
        labels.setdefault(vid, []).append(remap[int(g)])
    SPK.update(names=names, labels=labels)
    save_speakers()


def process_video(p, vid, model_size, lang):
    wav = CACHE / f"{vid}.wav"
    try:
        JOB["stage"] = "extracting audio"
        run(["ffmpeg", "-y", "-i", str(p), "-vn", "-ac", "1", "-ar", str(SR), str(wav)])
        audio = load_wav(wav)
    finally:
        wav.unlink(missing_ok=True)
    JOB["stage"] = "transcribing"
    segs, language = transcribe(get_model(model_size), audio, lang)
    JOB["stage"] = "aligning"
    aligned = align(audio, segs, language)
    words = [w for s in segs for w in s["words"] if not aligned or w["c"] >= MIN_CONF]
    JOB["stage"] = "voices"
    turns = make_turns(words)
    np.save(CACHE / f"{vid}.npy", embed_turns(audio, words, turns))
    data = {"v": CACHE_VER, "path": str(p), "lang": language, "aligned": aligned, "words": words, "turns": turns}
    (CACHE / f"{vid}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def read_cache(jf):
    if not jf.exists():
        return None
    data = json.loads(jf.read_text(encoding="utf-8"))
    return data if data.get("v") == CACHE_VER else None


def index_job(folder, model_size, lang):
    try:
        vids = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_EXT)
        JOB.update(running=True, total=len(vids), done=0, current="", stage="", errors=[], ipa_ok=True)
        for p in vids:
            JOB["current"] = p.name
            try:
                vid = vid_id(p)
                data = read_cache(CACHE / f"{vid}.json") or process_video(p, vid, model_size, lang)
                load_video(vid, p, data)
            except Exception as e:
                JOB["errors"].append(f"{p.name}: {e}")
            JOB["done"] += 1
        if not labels_ok():
            JOB["stage"] = "grouping voices"
            recluster()
    except Exception as e:
        JOB["errors"].append(str(e))
    finally:
        JOB.update(running=False, current="", stage="")


class IndexReq(BaseModel):
    path: str
    model: str = "large-v3"
    lang: str = "fr"


@app.post("/api/index")
def api_index(req: IndexReq):
    if JOB["running"]:
        raise HTTPException(409, "Indexing already running")
    if not Path(req.path).is_dir():
        raise HTTPException(400, "Folder not found")
    JOB.update(running=True, done=0, total=0)
    threading.Thread(target=index_job, args=(req.path, req.model, req.lang), daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def api_status():
    return {**JOB, "videos": len(VIDEOS), "words": sum(len(v["words"]) for v in VIDEOS.values())}


def turn_span(v, k):
    a, b = v["turns"][k]
    return v["words"][a]["t0"], v["words"][b]["t1"]


@app.get("/api/speakers")
def api_speakers():
    stats = {sid: {"id": sid, "name": name, "turns": 0, "dur": 0.0, "samples": []} for sid, name in SPK["names"].items()}
    for vid, v in VIDEOS.items():
        for k, sid in enumerate(SPK["labels"].get(vid, [])):
            if sid not in stats or k >= len(v["turns"]):
                continue
            t0, t1 = turn_span(v, k)
            st = stats[sid]
            st["turns"] += 1
            st["dur"] += t1 - t0
            st["samples"].append({"vid": vid, "start": round(t0, 3), "end": round(t1, 3)})
    for st in stats.values():
        st["dur"] = round(st["dur"], 1)
        st["samples"] = sorted(st["samples"], key=lambda s: s["start"] - s["end"])[:4]
    return {"thr": SPK["thr"], "speakers": sorted(stats.values(), key=lambda s: -s["dur"])}


class RenameReq(BaseModel):
    id: int
    name: str


@app.post("/api/speakers/rename")
def api_rename(req: RenameReq):
    name = req.name.strip()
    if req.id not in SPK["names"] or not name:
        raise HTTPException(400, "Unknown voice or empty name")
    twin = next((sid for sid, n in SPK["names"].items() if sid != req.id and n.lower() == name.lower()), None)
    if twin is None:
        SPK["names"][req.id] = name
    else:
        SPK["labels"] = {vid: [twin if s == req.id else s for s in labs] for vid, labs in SPK["labels"].items()}
        del SPK["names"][req.id]
    save_speakers()
    return api_speakers()


class ReclusterReq(BaseModel):
    thr: float


@app.post("/api/speakers/recluster")
def api_recluster(req: ReclusterReq):
    if JOB["running"]:
        raise HTTPException(409, "Indexing running")
    recluster(min(0.95, max(0.2, req.thr)))
    return api_speakers()


def find(stream, q, whole):
    out = []
    if not q or not stream["s"]:
        return out
    s = stream["s"]
    pos = 0
    while True:
        i = s.find(q, pos)
        if i < 0:
            break
        j = i + len(q) - 1
        pos = i + 1
        if j + 1 < len(s) and unicodedata.combining(s[j + 1]):
            continue
        if not whole or (stream["ws"][i] and stream["we"][j]):
            out.append((i, j))
    return out


def make_hit(v, stream, i, j, pad, kind):
    a, b = stream["wid"][i], stream["wid"][j]
    words = v["words"]
    labs = SPK["labels"].get(v["id"], [])
    turn = words[a]["turn"]
    spk = labs[turn] if 0 <= turn < len(labs) else 0
    return {
        "vid": v["id"],
        "name": v["name"],
        "start": round(max(0.0, stream["t0"][i] - pad), 3),
        "end": round(stream["t1"][j] + pad, 3),
        "before": " ".join(w["w"] for w in words[max(0, a - 3):a]),
        "match": " ".join(w["w"] for w in words[a:b + 1]),
        "after": " ".join(w["w"] for w in words[b + 1:b + 4]),
        "kind": kind,
        "conf": round(float(np.mean(stream["c"][i:j + 1])), 3),
        "spk": spk,
    }


@app.get("/api/search")
def api_search(q: str, mode: str = "auto", whole: bool = False, pad: int = 30, lang: str = "fr", limit: int = 600, spk: str = "", minconf: float = 0.0, sort: str = "conf"):
    q = q.strip()
    if not q:
        return {"total": 0, "hits": [], "facets": {}}
    p = pad / 1000.0
    if mode == "auto":
        mode = "ipa" if looks_like_ipa(q) else "both"
    qt = norm_text(q) if mode in ("text", "both") else ""
    qp = ""
    if mode == "ipa":
        qp = norm_ipa(q)
    elif mode == "both":
        qp = phonemize_query(q, lang)
    hits = []
    for v in VIDEOS.values():
        seen = []
        for i, j in find(v["ipa"], qp, whole):
            h = make_hit(v, v["ipa"], i, j, p, "phonetic")
            seen.append(h["start"])
            hits.append(h)
        for i, j in find(v["text"], qt, whole or (mode == "both" and bool(v["ipa"]["s"]))):
            h = make_hit(v, v["text"], i, j, p, "text")
            if any(abs(h["start"] - s) < 0.12 for s in seen):
                continue
            hits.append(h)
    hits = [h for h in hits if h["conf"] >= minconf]
    facets = Counter(h["spk"] for h in hits)
    want = {int(x) for x in spk.split(",") if x.strip().lstrip("-").isdigit()}
    if want:
        hits = [h for h in hits if h["spk"] in want]
    if sort == "time":
        hits.sort(key=lambda h: (h["name"], h["start"]))
    else:
        hits.sort(key=lambda h: (-h["conf"], h["name"], h["start"]))
    return {"total": len(hits), "hits": hits[:limit], "ipa": qp, "facets": facets}


@app.get("/media/{vid}")
def media(vid: str):
    v = VIDEOS.get(vid)
    if not v:
        raise HTTPException(404)
    return FileResponse(v["path"])


class Clip(BaseModel):
    vid: str
    start: float
    end: float
    label: str = ""


class ExportReq(BaseModel):
    clips: list[Clip]
    mode: str = "clips"
    fmt: str = "mp4"
    name: str = "ytp"


def cut(src, start, dur, out, fmt, norm):
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}"]
    if fmt == "wav":
        cmd += ["-vn", "-c:a", "pcm_s16le"]
        if norm:
            cmd += ["-ar", "48000", "-ac", "2"]
    else:
        if norm:
            cmd += ["-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k"]
        if norm:
            cmd += ["-ar", "48000", "-ac", "2"]
    cmd.append(str(out))
    run(cmd)


def slug(s):
    return re.sub(r"[^\w-]+", "_", s, flags=re.UNICODE).strip("_")[:40] or "clip"


@app.post("/api/export")
def api_export(req: ExportReq):
    if not req.clips:
        raise HTTPException(400, "No clips")
    ext = "wav" if req.fmt == "wav" else "mp4"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = slug(req.name)
    try:
        if req.mode == "concat":
            with tempfile.TemporaryDirectory() as tmp:
                lines = []
                for n, c in enumerate(req.clips):
                    v = VIDEOS[c.vid]
                    seg = Path(tmp) / f"{n:04d}.{ext}"
                    cut(v["path"], c.start, max(0.05, c.end - c.start), seg, ext, True)
                    lines.append("file '" + str(seg).replace("\\", "/").replace("'", "'\\''") + "'")
                lst = Path(tmp) / "list.txt"
                lst.write_text("\n".join(lines), encoding="utf-8")
                out = EXPORTS / f"{base}_{stamp}.{ext}"
                run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
            return {"file": out.name}
        paths = []
        for n, c in enumerate(req.clips):
            v = VIDEOS[c.vid]
            out = EXPORTS / f"{base}_{stamp}_{n + 1:03d}_{slug(c.label)}.{ext}"
            cut(v["path"], c.start, max(0.05, c.end - c.start), out, ext, False)
            paths.append(out)
        if len(paths) == 1:
            return {"file": paths[0].name}
        zp = EXPORTS / f"{base}_{stamp}.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for pth in paths:
                z.write(pth, pth.name)
        return {"file": zp.name}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/download/{fname}")
def api_download(fname: str):
    f = EXPORTS / Path(fname).name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, filename=f.name)


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


def preload_cache():
    for jf in CACHE.glob("*.json"):
        if jf == SPK_FILE:
            continue
        try:
            data = read_cache(jf)
            if data and Path(data["path"]).exists():
                load_video(jf.stem, data["path"], data)
        except Exception as e:
            print("Skipped", jf.name, e)
    load_speakers()
    if VIDEOS and not labels_ok():
        recluster()


if __name__ == "__main__":
    preload_cache()
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
