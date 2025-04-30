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
# Accetta anche eventuali spazi o trattini, che verranno rimossi in post
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
    version="1.0"
)

# ─── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in produzione specificare gli origin autorizzati
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── EasyOCR reader placeholder ────────────────────────────────────────────
reader: easyocr.Reader = None


@app.on_event("startup")
async def load_easyocr_model():
    """
    Carica il modello EasyOCR in startup per ridurre picchi di memoria all'avvio.
    """
    global reader
    logger.info("[startup] Inizio caricamento modello EasyOCR...")
    t0 = time.perf_counter()
    reader = easyocr.Reader(["en"], gpu=False)
    elapsed = time.perf_counter() - t0
    logger.info(f"[startup] Modello EasyOCR caricato in {elapsed:.1f}s")

# ─── Helper: preprocessing ─────────────────────────────────────────────────


def preprocess_gray(
    frame: np.ndarray,
    width: int = 640
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ridimensiona, converte in scala di grigi ed applica CLAHE.
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
    min_area: int = 2000,
    ar_range: Tuple[float, float] = (2.5, 7.0)
) -> Optional[Tuple[int, int, int, int]]:
    """
    Rileva contorni tramite Sobel+Otsu+morfologia,
    ritorna bounding box di maggiore area compatibile.
    """
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box, best_area = None, 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h) if h > 0 else 0
        if ar_range[0] <= ar <= ar_range[1] and area > best_area:
            best_area, best_box = area, (x, y, w, h)
    return best_box

# ─── Helper: prospettiva + ritaglio ────────────────────────────────────────


def four_point_transform(
    img: np.ndarray,
    rect: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Applica transform prospettica e restituisce il crop rect.
    """
    x, y, w, h = rect
    src = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype="float32")
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))

# ─── Helper: OCR targhe ────────────────────────────────────────────────────


def ocr_plate(crop: np.ndarray) -> Tuple[Optional[str], float]:
    """
    Applica EasyOCR e filtra risultati con regex.
    """
    raw = reader.readtext(crop, detail=1)
    logger.debug(f"[ocr_plate] raw results: {raw}")
    candidates: List[Tuple[str, float]] = []
    for _, text, conf in raw:
        plate = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        if ITALIAN_PLATE_REGEX.match(plate):
            candidates.append((plate, conf * 100))
    if not candidates:
        return None, 0.0
    best = max(candidates, key=lambda x: x[1])
    return best

# ─── Endpoint: health check ───────────────────────────────────────────────


@app.get("/", tags=["Health"])
async def health_check():
    logger.info("[health] OK")
    return {"status": "ok", "message": "ANPR Server in esecuzione"}

# ─── Endpoint: riconoscimento ─────────────────────────────────────────────


@app.post(
    "/recognize",
    response_model=PlateResponse,
    tags=["ANPR"],
    summary="Riconosce la targa da un’immagine"
)
async def recognize_plate(
    file: UploadFile = File(..., description="Immagine JPEG o PNG")
):
    logger.info(f"[recognize] Ricevuto file, content_type={file.content_type}")
    if file.content_type not in ("image/jpeg", "image/png"):
        logger.warning(
            f"[recognize] Formato non supportato: {file.content_type}")
        raise HTTPException(status_code=415, detail="Formato non supportato")

    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    logger.debug(
        f"[recognize] frame shape: {None if frame is None else frame.shape}")
    if frame is None:
        logger.error("[recognize] Impossibile decodificare l'immagine")
        raise HTTPException(
            status_code=400, detail="Impossibile decodificare l’immagine")

    vis, gray = preprocess_gray(frame)
    logger.debug(f"[recognize] preprocessed gray shape: {gray.shape}")
    rect = find_plate_region(gray)
    logger.debug(f"[recognize] find_plate_region ➞ {rect}")

    if rect:
        x, y, w, h = rect
        crop = four_point_transform(vis, rect)
        # Salva per debug locale
        cv2.imwrite("/tmp/last_crop.jpg", crop)
        plate, conf = ocr_plate(crop)
        box = [x, y, w, h]
        logger.info(
            f"[recognize] Plate detected: {plate}, Conf={conf:.1f}%, Box={box}")
    else:
        plate, conf, box = None, 0.0, None
        logger.info("[recognize] Nessuna regione targa trovata")

    return {"plate": plate, "confidence": round(conf, 1), "box": box}

# ─── Avvio standalone ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
