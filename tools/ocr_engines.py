"""OCR for emulator screenshots: engines behind one interface, and the image preparation that makes them work.

An N64 frame is 320x240 (or 640x480) with blurry, stylised, often tiny text on busy backgrounds. The engines
are trained on documents and street signs, so they need help: the picture is enlarged, sharpened and shown to
them in several versions (as it is, with contrast stretched, inverted - white-on-dark and dark-on-light text are
both common), and what each version finds is merged. `read_screen()` does all of that.

    from ocr_engines import read_screen
    for box in read_screen(PIL.Image.open("screen.png")):
        print(box.text, box.conf, box.x0, box.y0, box.x1, box.y1)    # coordinates of the original picture

Engines (whatever is installed; `pip install rapidocr-onnxruntime easyocr winrt-Windows.Media.Ocr ...`):
  rapid   RapidOCR (PaddleOCR models on onnxruntime): fast on a CPU
  easy    EasyOCR (CRAFT + CRNN, PyTorch): slower, strong on stylised text; uses a CUDA GPU when there is one
  win     Windows' own OCR engine (Windows.Media.Ocr): no model to download
"""
import asyncio
import dataclasses
import re
import threading
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclasses.dataclass
class Box:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float
    engine: str = ""
    variant: str = ""
    alts: list = dataclasses.field(default_factory=list)  # the other readings of the same text (other versions / engines)

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def w(self):
        return self.x1 - self.x0


def norm(s: str) -> str:
    """Upper case letters and digits only: what two readings of the same word have in common."""
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


# ---------------------------------------------------------------------------
# Engines

class _Engine:
    name = ""

    def read(self, img: Image.Image) -> List[Box]:
        raise NotImplementedError


class RapidEngine(_Engine):
    name = "rapid"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()
        self._lock = threading.Lock()

    def read(self, img):
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1]  # RapidOCR wants BGR
        with self._lock:
            res, _ = self._ocr(arr)
        out = []
        for pts, text, conf in res or []:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            out.append(Box(text, float(conf), min(xs), min(ys), max(xs), max(ys), self.name))
        return out


class EasyEngine(_Engine):
    name = "easy"

    def __init__(self, gpu: Optional[bool] = None):
        import easyocr
        import torch
        if gpu is None:
            gpu = torch.cuda.is_available()
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self._lock = threading.Lock()
        self.gpu = gpu

    def read(self, img):
        arr = np.asarray(img.convert("RGB"))
        with self._lock:
            res = self._reader.readtext(arr, paragraph=False, width_ths=0.7, text_threshold=0.5, low_text=0.3)
        out = []
        for pts, text, conf in res:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            out.append(Box(text, float(conf), min(xs), min(ys), max(xs), max(ys), self.name))
        return out


class WinEngine(_Engine):
    name = "win"

    def __init__(self):
        from winrt.windows.media.ocr import OcrEngine
        self._OcrEngine = OcrEngine
        self._engine = OcrEngine.try_create_from_user_profile_languages()
        if self._engine is None:
            raise RuntimeError("Windows has no OCR language installed")
        self._lock = threading.Lock()

    async def _recognize(self, img):
        from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.storage.streams import DataWriter
        rgba = img.convert("RGBA")
        bgra = np.asarray(rgba)[:, :, [2, 1, 0, 3]].tobytes()
        w = DataWriter()
        w.write_bytes(bgra)
        sb = SoftwareBitmap.create_copy_with_alpha_from_buffer(w.detach_buffer(), BitmapPixelFormat.BGRA8, rgba.width,
                                                               rgba.height, BitmapAlphaMode.PREMULTIPLIED)
        return await self._engine.recognize_async(sb)

    def read(self, img):
        with self._lock:
            res = asyncio.run(self._recognize(img))
        out = []
        for line in res.lines:
            ws = list(line.words)
            if not ws:
                continue
            x0 = min(w.bounding_rect.x for w in ws)
            y0 = min(w.bounding_rect.y for w in ws)
            x1 = max(w.bounding_rect.x + w.bounding_rect.width for w in ws)
            y1 = max(w.bounding_rect.y + w.bounding_rect.height for w in ws)
            out.append(Box(line.text, 0.8, x0, y0, x1, y1, self.name))  # Windows gives no confidence
        return out


_LOCAL = threading.local()  # one engine of each kind per thread: the engines are not thread safe


def get_engine(name: str) -> _Engine:
    engines = getattr(_LOCAL, "engines", None)
    if engines is None:
        engines = _LOCAL.engines = {}
    if name not in engines:
        engines[name] = {"rapid": RapidEngine, "easy": EasyEngine, "win": WinEngine}[name]()
    return engines[name]


def available_engines() -> List[str]:
    ok = []
    for n in ("rapid", "easy", "win"):
        try:
            get_engine(n)
            ok.append(n)
        except Exception:  # not installed / no language
            pass
    return ok


# ---------------------------------------------------------------------------
# Image preparation

def _stretch(img: Image.Image) -> Image.Image:
    """Contrast stretched to the full range (2nd-98th percentile), per channel."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    lo = np.percentile(a, 2, axis=(0, 1))
    hi = np.percentile(a, 98, axis=(0, 1))
    a = np.clip((a - lo) / np.maximum(hi - lo, 1) * 255, 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def variants(img: Image.Image, scale: int):
    """(name, picture) pairs: the enlarged screen as it is, contrast stretched, inverted."""
    big = img.convert("RGB").resize((img.width * scale, img.height * scale), Image.LANCZOS)
    sharp = big.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    yield "raw", sharp
    yield "stretch", _stretch(sharp)
    yield "invert", ImageOps.invert(_stretch(sharp))


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _overlap_small(a: Box, b: Box) -> float:
    """Intersection over the smaller box: 1 when one lies inside the other."""
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    small = min(a.w * a.h, b.w * b.h)
    return ix * iy / small if small > 0 else 0.0


def merge(boxes: List[Box]) -> List[Box]:
    """The same text found by several versions / engines becomes one box. Boxes that lie on the same place form a
    cluster; its text is the reading most of them agree on (weighted by confidence), the other readings are kept in
    `alts` - a word the best reading garbled is often right in another one."""
    boxes = [b for b in sorted(boxes, key=lambda b: -b.conf) if norm(b.text)]
    clusters: List[List[Box]] = []
    for b in boxes:
        for c in clusters:
            if _overlap_small(b, c[0]) > 0.55 and abs(b.cy - c[0].cy) < max(b.h, c[0].h) * 0.6:
                c.append(b)
                break
        else:
            clusters.append([b])
    out: List[Box] = []
    for c in clusters:
        votes = {}
        for b in c:
            votes[norm(b.text)] = votes.get(norm(b.text), 0.0) + b.conf
        # the reading with the most weight; ties go to the more confident / longer one
        best = max(c, key=lambda b: (votes[norm(b.text)], b.conf, len(b.text)))
        rep = dataclasses.replace(best)
        rep.alts = [b.text for b in c if norm(b.text) != norm(best.text)]
        rep.x0 = min(b.x0 for b in c if norm(b.text) == norm(best.text))
        rep.x1 = max(b.x1 for b in c if norm(b.text) == norm(best.text))
        rep.y0 = min(b.y0 for b in c if norm(b.text) == norm(best.text))
        rep.y1 = max(b.y1 for b in c if norm(b.text) == norm(best.text))
        out.append(rep)
    out.sort(key=lambda b: (round(b.cy / 8), b.x0))
    return out


def read_one(img: Image.Image, engine: str, scale: int, vname: str) -> List[Box]:
    """One engine on one version of the picture, boxes in the coordinates of the picture as given."""
    eng = get_engine(engine)
    for name, v in variants(img, scale):
        if name != vname:
            continue
        out = eng.read(v)
        for b in out:
            b.x0 /= scale; b.x1 /= scale; b.y0 /= scale; b.y1 /= scale
            b.variant = f"{vname}x{scale}"
        return out
    return []


def read_screen(img: Image.Image, engines=("rapid",), scales=(3,), variant_names=None, min_conf=0.0) -> List[Box]:
    """All the text found on a screenshot, in the coordinates of the picture as given."""
    found: List[Box] = []
    for en in engines:
        eng = get_engine(en)
        for scale in scales:
            for vname, v in variants(img, scale):
                if variant_names and vname not in variant_names:
                    continue
                for b in eng.read(v):
                    if b.conf < min_conf:
                        continue
                    b.x0 /= scale; b.x1 /= scale; b.y0 /= scale; b.y1 /= scale
                    b.variant = f"{vname}x{scale}"
                    found.append(b)
    return merge(found)
