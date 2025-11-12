#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Batch cover cropper with content-aware framing + safe label area
# NumPy 2.0–compatible.
# v5_clean:
# - Auto presets: "product" vs "people" (weights, moat, zoom guard).
# - Mobile (750x750): strict crop. Desktop (1920x1080): letterbox allowed.
# - File size limit: <= max_bytes (default 1MB) via quality binary search (JPG/WEBP).
# - PNGs авто-конвертятся в JPG, если не влезают в лимит.
# - Учитываются отступы и охранная зона под лейбл.
# - python "E:\Other\Jornal\batch_cover_cropper_v5_clean.py" --in "E:\Other\Jornal\input" --out1 "E:\Other\Jornal\out\desktop_1920x1080" --out2 "E:\Other\Jornal\out\mobile_750x750" --format jpg --quality 92 --mode auto --max_bytes 1000000 --min_quality 45

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List
import re
import io
import numpy as np
from PIL import Image
import cv2

# ---------- Utils ----------

def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    mn = float(arr.min())
    mx = float(arr.max())
    rng = mx - mn
    if rng < 1e-6:
        return np.zeros_like(arr, dtype="float32")
    return (arr - mn) / (rng + 1e-6)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def imread_any(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    elif im.mode != "RGB":
        return im.convert("RGB")
    return im

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return np.array(img)[:, :, ::-1].copy()

# ---------- Specs ----------

@dataclass
class Spec:
    name: str
    W: int
    H: int
    margin_left: int
    margin_right: int
    margin_top: int
    margin_bottom: int
    label_left: int
    label_bottom: int
    label_w: int
    label_h: int
    allow_letterbox: bool

def make_spec_desktop(label_w: int = 380, label_h: int = 120) -> Spec:
    return Spec(
        name="desktop_1920x1080",
        W=1920, H=1080,
        margin_left=80, margin_right=80, margin_top=148, margin_bottom=72,
        label_left=947, label_bottom=289, label_w=label_w, label_h=label_h,
        allow_letterbox=True
    )

def make_spec_mobile(label_w: int = 240, label_h: int = 90) -> Spec:
    return Spec(
        name="mobile_750x750",
        W=750, H=750,
        margin_left=64, margin_right=64, margin_top=126, margin_bottom=380,
        label_left=330, label_bottom=665, label_w=label_w, label_h=label_h,
        allow_letterbox=False
    )

# ---------- Detectors ----------

def compute_saliency(img_cv: np.ndarray) -> np.ndarray:
    sal = None
    try:
        if hasattr(cv2, "saliency"):
            sal_obj = cv2.saliency.StaticSaliencyFineGrained_create()
            ok, sal = sal_obj.computeSaliency(img_cv)
            if ok and sal is not None:
                sal = sal.astype("float32")
    except Exception:
        sal = None
    if sal is None:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        sal = normalize01(mag)
    sal = cv2.GaussianBlur(sal, (0,0), 3)
    sal = normalize01(sal)
    return sal

def compute_text_like_mask(img_cv: np.ndarray, scale_limit: int = 1024) -> np.ndarray:
    h, w = img_cv.shape[:2]
    scale = 1.0
    small = img_cv
    if max(h, w) > scale_limit:
        scale = scale_limit / float(max(h, w))
        small = cv2.resize(img_cv, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    try:
        mser = cv2.MSER_create(_delta=5, _min_area=60, _max_area=2000)
    except TypeError:
        mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for pts in regions:
        cv2.fillPoly(mask, [pts], 255)
    mask = cv2.medianBlur(mask, 5)
    if scale != 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask.astype("float32")/255.0

def compute_face_boxes(img_cv: np.ndarray) -> List[tuple]:
    boxes = []
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return boxes
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            boxes.append((x, y, w, h))
    except Exception:
        pass
    return boxes

def face_mask_from_boxes(img_cv: np.ndarray, boxes: List[tuple]) -> np.ndarray:
    mask = np.zeros(img_cv.shape[:2], dtype=np.float32)
    for (x, y, w, h) in boxes:
        cv2.rectangle(mask, (x, y), (x+w, y+h), 1.0, thickness=-1)
    return mask

# ---------- Presets ----------

@dataclass
class Weights:
    saliency: float
    text: float
    face: float

@dataclass
class Preset:
    name: str
    weights: Weights
    edge_moat_px_desktop: int
    edge_moat_px_mobile: int
    label_penalty_boost_mobile: float
    zoom_guard_mobile: float
    zoom_guard_desktop: float

PRODUCT_PRESET = Preset(
    name="product",
    weights=Weights(saliency=0.45, text=0.40, face=0.15),
    edge_moat_px_desktop=28,
    edge_moat_px_mobile=56,
    label_penalty_boost_mobile=1.6,
    zoom_guard_mobile=0.82,
    zoom_guard_desktop=0.70
)

PEOPLE_PRESET = Preset(
    name="people",
    weights=Weights(saliency=0.50, text=0.15, face=0.35),
    edge_moat_px_desktop=20,
    edge_moat_px_mobile=36,
    label_penalty_boost_mobile=1.2,
    zoom_guard_mobile=0.70,
    zoom_guard_desktop=0.65
)

PRODUCT_HINTS = re.compile(r"(product|pack|packshot|bottle|tube|jar|box|cream|gel|serum|lotion|sunscreen|spf|bioderma|atoderm|sensibio|sebium|cicabio|photoderm|esthederm|etat)", re.I)

def choose_preset_auto(img_cv: np.ndarray, filename: str) -> Preset:
    faces = compute_face_boxes(img_cv)
    if faces:
        return PEOPLE_PRESET
    text_mask = compute_text_like_mask(img_cv)
    text_ratio = float(text_mask.mean())
    name_hint = bool(PRODUCT_HINTS.search(filename or ""))
    if text_ratio > 0.022 or name_hint:
        return PRODUCT_PRESET
    return PEOPLE_PRESET

def importance_map(img_cv: np.ndarray, preset: Preset) -> np.ndarray:
    sal = compute_saliency(img_cv)
    txt = compute_text_like_mask(img_cv)
    face = face_mask_from_boxes(img_cv, compute_face_boxes(img_cv))
    imp = preset.weights.saliency * sal + preset.weights.text * txt + preset.weights.face * face
    return normalize01(imp)

# ---------- Integral image & sums ----------

def integral_image(m: np.ndarray) -> np.ndarray:
    return cv2.integral(m)[1:, 1:]

def rect_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x2, y2 = x+w-1, y+h-1
    A = ii[y2, x2]
    B = ii[y-1, x2] if y > 0 else 0.0
    C = ii[y2, x-1] if x > 0 else 0.0
    D = ii[y-1, x-1] if (x>0 and y>0) else 0.0
    return float(A - B - C + D)

# ---------- Crop chooser ----------

@dataclass
class SpecCrop:
    x: int; y: int; w: int; h: int
    used_letterbox: bool
    score: float

def choose_crop(img: Image.Image, spec: Spec, preset: Preset,
                grid_x: int, grid_y: int,
                edge_moat_px: int,
                label_penalty_boost: float,
                zoom_guard_scale_min: float) -> SpecCrop:
    W0, H0 = img.size
    ar = spec.W / spec.H
    img_cv = pil_to_cv(img)
    imp = importance_map(img_cv, preset)
    ii = integral_image(imp)

    safe_x1 = spec.margin_left
    safe_y1 = spec.margin_top
    safe_x2 = spec.W - spec.margin_right
    safe_y2 = spec.H - spec.margin_bottom

    label_w = spec.label_w
    label_h = spec.label_h
    label_x1 = spec.label_left
    label_y2 = spec.H - spec.label_bottom
    label_y1 = label_y2 - label_h

    max_w = min(W0, int(H0 * ar))
    max_h = int(max_w / ar)
    if max_h > H0:
        max_h = H0
        max_w = int(max_h * ar)

    best_score = -1e18
    best = (0, 0, max_w, max_h)

    scale_min = zoom_guard_scale_min if not spec.allow_letterbox else preset.zoom_guard_desktop
    for scale in np.linspace(1.0, scale_min, 12):
        cw = int(max_w * scale)
        ch = int(cw / ar)
        if ch > H0:
            ch = H0
            cw = int(ch * ar)
        if cw < 200 or ch < 200:
            continue

        xs = np.linspace(0, W0 - cw, num=grid_x).astype(int)
        ys = np.linspace(0, H0 - ch, num=grid_y).astype(int)

        for x in xs:
            for y in ys:
                sx = cw / spec.W
                sy = ch / spec.H

                lx1 = int(x + label_x1 * sx)
                ly1 = int(y + label_y1 * sy)
                lw  = max(1, int(label_w * sx))
                lh  = max(1, int(label_h * sy))

                fx1 = int(x + safe_x1 * sx)
                fy1 = int(y + safe_y1 * sy)
                fx2 = int(x + safe_x2 * sx)
                fy2 = int(y + safe_y2 * sy)
                fw = max(1, fx2 - fx1)
                fh = max(1, fy2 - fy1)

                moat_inset_x = int(edge_moat_px * sx)
                moat_inset_y = int(edge_moat_px * sy)
                ix1 = fx1 + moat_inset_x
                iy1 = fy1 + moat_inset_y
                ix2 = fx2 - moat_inset_x
                iy2 = fy2 - moat_inset_y
                iw = max(1, ix2 - ix1)
                ih = max(1, iy2 - iy1)

                inside_safe  = rect_sum(ii, fx1, fy1, fw, fh)
                in_label     = rect_sum(ii, lx1, ly1, lw, lh)
                inside_inner = rect_sum(ii, ix1, iy1, iw, ih)
                moat         = inside_safe - inside_inner

                score = inside_inner - 1.2*moat - (2.0 * label_penalty_boost) * in_label
                if score > best_score:
                    best_score = score
                    best = (x, y, cw, ch)

    used_letterbox = False
    if spec.allow_letterbox:
        total_full = rect_sum(ii, 0, 0, W0, H0)
        best_included = rect_sum(ii, best[0], best[1], best[2], best[3])
        loss_ratio = 1.0 - (best_included / (total_full + 1e-6))
        used_letterbox = loss_ratio > 0.35

    return SpecCrop(best[0], best[1], best[2], best[3], used_letterbox, best_score)

def compose_output(img: Image.Image, spec: Spec, dec: SpecCrop) -> Image.Image:
    if dec.used_letterbox and spec.allow_letterbox:
        W, H = spec.W, spec.H
        safe_w = W - spec.margin_left - spec.margin_right
        safe_h = H - spec.margin_top - spec.margin_bottom
        scale = min(safe_w / img.width, safe_h / img.height)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        def bg_color_from_image(pil_im: Image.Image):
            w, h = pil_im.size
            patch = []
            for (cx, cy) in [(0,0), (w-1,0), (0,h-1), (w-1,h-1)]:
                box = (max(cx-20,0), max(cy-20,0), min(cx+20,w), min(cy+20,h))
                arr = np.array(pil_im.crop(box)).reshape(-1,3)
                patch.append(np.median(arr, axis=0))
            med = np.median(np.vstack(patch), axis=0).astype(int)
            return (int(med[0]), int(med[1]), int(med[2]))
        color = bg_color_from_image(img)
        canvas = Image.new("RGB", (W, H), color)
        x0 = spec.margin_left + (safe_w - new_w)//2
        y0 = spec.margin_top + (safe_h - new_h)//2
        canvas.paste(resized, (x0, y0))
        return canvas
    crop = img.crop((dec.x, dec.y, dec.x+dec.w, dec.y+dec.h))
    return crop.resize((spec.W, spec.H), Image.LANCZOS)

# ---------- Size-limited saving ----------

def _save_to_bytes(img: Image.Image, fmt: str, quality: int) -> bytes:
    bio = io.BytesIO()
    fmt_up = fmt.upper()
    if fmt_up in ("JPG", "JPEG"):
        img.save(bio, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)
    elif fmt_up == "WEBP":
        img.save(bio, format="WEBP", quality=quality, method=6)
    elif fmt_up == "PNG":
        img.save(bio, format="PNG", optimize=True)
    else:
        img.save(bio, format=fmt_up)
    return bio.getvalue()

def save_with_limit(img: Image.Image, path: str, fmt: str, target_bytes: int,
                    start_quality: int = 92, min_quality: int = 45) -> None:
    fmt_norm = "jpg" if fmt.lower() == "jpeg" else fmt.lower()
    if fmt_norm in ("jpg", "webp"):
        lo, hi = min_quality, start_quality
        best_bytes = None
        while lo <= hi:
            mid = (lo + hi) // 2
            data = _save_to_bytes(img, fmt_norm, mid)
            if len(data) <= target_bytes:
                best_bytes = data
                lo = mid + 1
            else:
                hi = mid - 1
        if best_bytes is None:
            best_bytes = _save_to_bytes(img, fmt_norm, min_quality)
        with open(path, "wb") as f:
            f.write(best_bytes)
    else:
        data = _save_to_bytes(img, fmt_norm, start_quality)
        if len(data) <= target_bytes:
            with open(path, "wb") as f:
                f.write(data)
        else:
            alt = os.path.splitext(path)[0] + ".jpg"
            save_with_limit(img, alt, "jpg", target_bytes, start_quality, min_quality)

# ---------- Pipeline ----------

def process_one(img: Image.Image, src_name: str,
                spec_desktop: Spec, spec_mobile: Spec,
                mode: str):
    img_cv = pil_to_cv(img)
    if mode == "product":
        preset = PRODUCT_PRESET
    elif mode == "people":
        preset = PEOPLE_PRESET
    else:
        preset = choose_preset_auto(img_cv, src_name)

    dec1 = choose_crop(img, spec_desktop, preset,
                       grid_x=12, grid_y=8,
                       edge_moat_px=preset.edge_moat_px_desktop,
                       label_penalty_boost=1.0,
                       zoom_guard_scale_min=preset.zoom_guard_desktop)
    out1 = compose_output(img, spec_desktop, dec1)

    dec2 = choose_crop(img, spec_mobile, preset,
                       grid_x=16, grid_y=16,
                       edge_moat_px=preset.edge_moat_px_mobile,
                       label_penalty_boost=preset.label_penalty_boost_mobile,
                       zoom_guard_scale_min=preset.zoom_guard_mobile)
    dec2.used_letterbox = False
    out2 = compose_output(img, spec_mobile, dec2)
    return out1, out2, preset.name

def process_folder(input_dir: str, out1: str, out2: str,
                   label1_size: Tuple[int,int], label2_size: Tuple[int,int],
                   fmt: str, quality: int, mode: str,
                   max_bytes: int, min_quality: int):
    spec1 = make_spec_desktop(label1_size[0], label1_size[1])
    spec2 = make_spec_mobile(label2_size[0], label2_size[1])
    ensure_dir(out1); ensure_dir(out2)
    supported = (".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff")
    idx = 0
    for root, _, files in os.walk(input_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in supported:
                continue
            src_path = os.path.join(root, fn)
            try:
                img = imread_any(src_path)
                out_img1, out_img2, preset_name = process_one(img, fn, spec1, spec2, mode)
                rel = os.path.relpath(src_path, input_dir)
                base, _ = os.path.splitext(rel)
                safe_base = base.replace(os.sep, "__")
                p1 = os.path.join(out1, f"{safe_base}.{fmt}")
                p2 = os.path.join(out2, f"{safe_base}.{fmt}")
                save_with_limit(out_img1, p1, fmt, target_bytes=max_bytes,
                                start_quality=quality, min_quality=min_quality)
                save_with_limit(out_img2, p2, fmt, target_bytes=max_bytes,
                                start_quality=quality, min_quality=min_quality)
                idx += 1
                print(f"[OK] {fn} [{preset_name}]")
            except Exception as e:
                print(f"[FAIL] {fn}: {e}")
    print(f"Done. Processed {idx} images.")

def parse_size(s: str) -> Tuple[int,int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Size must be like 380x120")

def main():
    ap = argparse.ArgumentParser(description="Batch cropper with auto presets and size limit")
    ap.add_argument("--in", dest="input_dir", required=True)
    ap.add_argument("--out1", dest="out1", required=True)
    ap.add_argument("--out2", dest="out2", required=True)
    ap.add_argument("--label1_size", type=parse_size, default="380x120")
    ap.add_argument("--label2_size", type=parse_size, default="240x90")
    ap.add_argument("--format", dest="fmt", default="jpg", choices=["jpg","jpeg","png","webp"])
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--mode", choices=["auto","product","people"], default="auto")
    ap.add_argument("--max_bytes", type=int, default=1_000_000)
    ap.add_argument("--min_quality", type=int, default=45)
    args = ap.parse_args()

    ensure_dir(args.out1); ensure_dir(args.out2)
    process_folder(args.input_dir, args.out1, args.out2,
                   args.label1_size, args.label2_size,
                   args.fmt, args.quality, args.mode,
                   args.max_bytes, args.min_quality)

if __name__ == "__main__":
    main()
