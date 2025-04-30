#!/usr/bin/env python3
"""
ANPR avanzato per targhe italiane – robusto e preciso
------------------------------------------------------

Espone un'API REST per il riconoscimento OCR di targhe italiane da immagini.

Dipendenze:
    pip install fastapi uvicorn easyocr opencv-python-headless numpy pydantic python-multipart python-Levenshtein

Esecuzione:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import logging
import re
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import Levenshtein

# ─── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("anpr_advanced")

# ─── Config ───────────────────────────────────────────────────────────────
ITALIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$")
UPSCALE_FACTOR = 2
MIN_CONFIDENCE = 35.0
FUZZY_THRESHOLD = 1  # numero massimo di errori accettabili (Levenshtein)

# ─── FastAPI setup ────────────────────────────────────────────────────────
app = FastAPI(title="ANPR Avanzato", version="2.0",
              description="OCR targhe italiane avanzato")
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

reader: easyocr.Reader = None


@app.on_event("startup")
async def load_model():
    global reader
    logger.info("Caricamento modello EasyOCR...")
    t0 = time.perf_counter()
    reader = easyocr.Reader(["en"], gpu=False)
    logger.info(f"Modello EasyOCR caricato in {time.perf_counter() - t0:.1f}s")

# ─── Response ─────────────────────────────────────────────────────────────


class PlateResponse(BaseModel):
    plate: Optional[str]
    confidence: float
    box: Optional[List[int]]

# ─── Funzioni OCR ─────────────────────────────────────────────────────────


def preprocess_image(img: np.ndarray, width: int = 640) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    scale = width / float(w)
    resized = cv2.resize(img, (width, int(h * scale)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return resized, clahe.apply(gray)


def find_plate_candidates(gray: np.ndarray, min_area: int = 1200) -> List[Tuple[int, int, int, int]]:
    sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob = cv2.convertScaleAbs(sob)
    _, bw = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 5))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    if cv2.countNonZero(closed) < min_area:
        closed = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h)
        if 2.0 <= ar <= 8.5:
            boxes.append((x, y, w, h))
    return boxes


def expand_box(x, y, w, h, margin=0.15, shape=None):
    dw, dh = int(w * margin), int(h * margin)
    x_new = max(x - dw, 0)
    y_new = max(y - dh, 0)
    if shape is not None:
        max_w = min(w + 2 * dw, shape[1] - x_new)
        max_h = min(h + 2 * dh, shape[0] - y_new)
        return x_new, y_new, max_w, max_h
    return x_new, y_new, w + 2 * dw, h + 2 * dh


def crop_transform(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    src = np.array([[x, y], [x + w, y], [x + w, y + h],
                   [x, y + h]], dtype="float32")
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))


def is_similar_plate(text: str) -> bool:
    """Permette tolleranza di max 1 errore per targhe simili (fuzzy matching)."""
    if len(text) == 7 and re.match(r"^[A-Z0-9]{7}$", text):
        matches = [ITALIAN_PLATE_REGEX.match(text)]
        if any(matches):
            return True
        # tentativo fuzzy
        alphanum = re.sub(r'[^A-Z0-9]', '', text)
        return len(alphanum) == 7 and Levenshtein.distance(alphanum, alphanum.upper()) <= FUZZY_THRESHOLD
    return False


def ocr_candidates(crops: List[np.ndarray]) -> Tuple[Optional[str], float, Optional[Tuple[int, int, int, int]]]:
    best_plate = None
    best_conf = 0.0
    best_box = None
    for crop, box in crops:
        h, w = crop.shape[:2]
        scaled = cv2.resize(crop, (w * UPSCALE_FACTOR, h *
                            UPSCALE_FACTOR), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            results = reader.readtext(thr, detail=1)
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            continue
        for _, text, conf in results:
            cleaned = re.sub(r'[^A-Z0-9]', '', text).upper()
            if is_similar_plate(cleaned) and conf * 100 > best_conf:
                best_plate, best_conf, best_box = cleaned, conf * 100, box
    return best_plate, best_conf, best_box

# ─── API ───────────────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "version": "2.0"}


@app.post("/recognize", response_model=PlateResponse, tags=["ANPR"])
async def recognize_plate(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(415, detail="Formato non supportato")
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, detail="Immagine non valida")

    vis, gray = preprocess_image(img)
    candidates = find_plate_candidates(gray)
    if not candidates:
        logger.info("Nessun candidato trovato.")
        return PlateResponse(plate=None, confidence=0.0, box=None)

    crops = []
    for box in candidates:
        x, y, w, h = expand_box(*box, margin=0.2, shape=vis.shape)
        crop = crop_transform(vis, (x, y, w, h))
        crops.append((crop, [x, y, w, h]))

    plate, conf, bbox = ocr_candidates(crops)
    logger.info(f"[plate={plate}] [conf={conf:.1f}%] [box={bbox}]")
    return PlateResponse(plate=plate, confidence=round(conf, 1), box=bbox)

# ─── Avvio diretto ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
