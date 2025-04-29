#!/usr/bin/env python3
"""
ANPR Server per targhe italiane
-------------------------------

Espone un’API REST che riceve immagini via POST e risponde
con la targa riconosciuta, la confidenza e la bounding box.

Dipendenze:
    pip install fastapi uvicorn easyocr opencv-python-headless numpy pydantic

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
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ─── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ─── EasyOCR reader (solo CPU) ─────────────────────────────────────────────
reader = easyocr.Reader(["en"], gpu=False)

# ─── Regex per targhe italiane (AA123BB) ───────────────────────────────────
ITALIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$")

# ─── Response model ────────────────────────────────────────────────────────


class PlateResponse(BaseModel):
    plate: Optional[str]
    confidence: float
    box: Optional[List[int]]  # [x, y, w, h]


# ─── App FastAPI ───────────────────────────────────────────────────────────
app = FastAPI(
    title="ANPR Server Italiano",
    description="Riconoscimento targhe italiane in locale",
    version="1.0"
)

# ─── Helper: preprocessing ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://portal.gioblu.it"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def preprocess_gray(
    frame: np.ndarray,
    width: int = 640
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ridimensiona l’immagine a `width` px (mantiene proporzioni),
    converte in scala di grigi ed applica CLAHE.
    """
    h, w = frame.shape[:2]
    scale = width / float(w)
    resized = cv2.resize(frame, (width, int(h * scale)))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return resized, gray_eq

# ─── Helper: trova regione targa ───────────────────────────────────────────


def find_plate_region(
    gray: np.ndarray,
    min_area: int = 4500
) -> Optional[Tuple[int, int, int, int]]:
    """
    Rileva contorni tramite Sobel+Otsu+morfologia,
    ritorna il bounding box (x,y,w,h) del contorno più grande
    compatibile con aspect-ratio targhe italiane.
    """
    # 1) Sobel orizzontale
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    # 2) Otsu threshold
    _, bw = cv2.threshold(
        grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # 3) Morfologia per unire i caratteri
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
    # 4) Contorni
    cnts, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    best_box = None
    best_area = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h) if h > 0 else 0
        # solo proporzioni in [3.5, 6.0]
        if 3.5 <= ar <= 6.0 and area > best_area:
            best_area = area
            best_box = (x, y, w, h)
    return best_box

# ─── Helper: prospettica + ritaglio ────────────────────────────────────────


def four_point_transform(
    img: np.ndarray,
    rect: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Applica transform prospettica su rect=(x,y,w,h) e restituisce il crop.
    """
    x, y, w, h = rect
    src = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype="float32")
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))

# ─── Helper: OCR targhe ────────────────────────────────────────────────────


def ocr_plate(crop: np.ndarray) -> Tuple[Optional[str], float]:
    """
    Applica EasyOCR sul crop, filtra i risultati con regex targa,
    ritorna (plate, confidence) del candidato migliore.
    """
    raw = reader.readtext(crop, detail=1)
    cands: List[Tuple[str, float]] = []
    for _, text, conf in raw:
        plate = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        if ITALIAN_PLATE_REGEX.match(plate):
            cands.append((plate, conf * 100))
    if not cands:
        return None, 0.0
    # prendi quello a conf più alta
    return max(cands, key=lambda x: x[1])

# ─── Endpoint: health check ───────────────────────────────────────────────


@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "ANPR Server in esecuzione"}

# ─── Endpoint: riconoscimento ─────────────────────────────────────────────


@app.post(
    "/recognize",
    response_model=PlateResponse,
    tags=["ANPR"],
    summary="Riconosce la targa da un’immagine",
)
async def recognize_plate(
    file: UploadFile = File(..., description="Immagine JPEG o PNG")
):
    # validation content-type
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Formato non supportato")

    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=400, detail="Impossibile decodificare l’immagine")

    t0 = time.perf_counter()
    vis, gray = preprocess_gray(frame)
    rect = find_plate_region(gray)

    if rect:
        x, y, w, h = rect
        crop = four_point_transform(vis, rect)
        plate, conf = ocr_plate(crop)
        box = [x, y, w, h]
    else:
        plate, conf, box = None, 0.0, None

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logging.info(
        f"Processed in {elapsed_ms:.1f} ms – Plate={plate} ({conf:.1f}%) Box={box}")

    return {
        "plate": plate,
        "confidence": round(conf, 1),
        "box": box
    }

# ─── Avvio standalone ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
