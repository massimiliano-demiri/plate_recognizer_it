#!/usr/bin/env python3
"""
ANPR Server per targhe italiane
-------------------------------

Espone un’API REST che riceve immagini via POST e risponde
con la targa riconosciuta, la confidenza e la bounding box.

Dipendenze:
    pip install fastapi uvicorn easyocr opencv-python-headless numpy pydantic python-multipart

Esempio di utilizzo:
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

# ─── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("anpr_server")

# ─── Regex per targhe italiane (AA123BB) ───────────────────────────────────
ITALIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$")

# ─── Response model ────────────────────────────────────────────────────────


class PlateResponse(BaseModel):
    plate: Optional[str]
    confidence: float
    box: Optional[List[int]]  # [x, y, w, h]


# ─── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(
    title="ANPR Server Italiano",
    description="Riconoscimento targhe italiane in locale",
    version="1.1"
)

# ─── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── EasyOCR reader placeholder ────────────────────────────────────────────
reader: easyocr.Reader = None


@app.on_event("startup")
async def load_easyocr_model():
    logger.info("[startup] Loading EasyOCR model...")
    t0 = time.perf_counter()
    global reader
    reader = easyocr.Reader(["en"], gpu=False)
    logger.info(f"[startup] EasyOCR loaded in {time.perf_counter() - t0:.1f}s")

# ─── Preprocess grayscale with CLAHE ────────────────────────────────────────


def preprocess_gray(frame: np.ndarray, width: int = 640) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    scale = width / float(w)
    resized = cv2.resize(frame, (width, int(h * scale)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return resized, clahe.apply(gray)

# ─── Find plate region via Sobel+morpho + fallback Canny ──────────────────


def find_plate_region(gray: np.ndarray,
                      min_area: int = 1500,
                      ar_range: Tuple[float, float] = (2.0, 8.0)) -> Optional[Tuple[int, int, int, int]]:
    def detect_edges(img):
        sob = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sob = cv2.convertScaleAbs(sob)
        _, thr = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thr

    def detect_canny(img):
        return cv2.Canny(img, 50, 150)

    bw = detect_edges(gray)
    morphed = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    cand = morphed
    if cv2.countNonZero(cand) < 1000:
        cand = detect_canny(gray)

    cnts, _ = cv2.findContours(
        cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    best_area = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w/float(h) if h > 0 else 0
        if ar_range[0] <= ar <= ar_range[1] and area > best_area:
            best_area, best_box = area, (x, y, w, h)
    return best_box

# ─── Perspective crop ─────────────────────────────────────────────────────


def four_point_transform(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    src = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype="float32")
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))

# ─── OCR and filter ───────────────────────────────────────────────────────


def ocr_plate(crop: np.ndarray) -> Tuple[Optional[str], float]:
    raw = reader.readtext(crop, detail=1)
    logger.debug(f"[ocr] raw={raw}")
    cands: List[Tuple[str, float]] = []
    for _, text, conf in raw:
        plate = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        if ITALIAN_PLATE_REGEX.match(plate):
            cands.append((plate, conf*100))
    return max(cands, key=lambda x: x[1]) if cands else (None, 0.0)

# ─── Health-check ─────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
async def health(): return {"status": "ok"}

# ─── Main endpoint ────────────────────────────────────────────────────────


@app.post("/recognize", response_model=PlateResponse, tags=["ANPR"])
async def recognize_plate(file: UploadFile = File(...)):
    logger.info("[recognize] Received file: %s", file.filename)
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Formato non supportato")
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Impossibile decodificare immagine")
    vis, gray = preprocess_gray(frame)
    rect = find_plate_region(gray)
    if not rect:
        return {"plate": None, "confidence": 0.0, "box": None}
    x, y, w, h = rect
    crop = four_point_transform(vis, rect)
    plate, conf = ocr_plate(crop)
    return {"plate": plate, "confidence": round(conf, 1), "box": [x, y, w, h]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
