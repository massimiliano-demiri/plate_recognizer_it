#!/usr/bin/env python3
"""
ANPR Server per targhe italiane (versione avanzata con debug)
-------------------------------------------------------------

Espone un’API REST che riceve immagini via POST e risponde
con la targa riconosciuta, la confidenza, la bounding box e i tentativi OCR.

Dipendenze:
    pip install fastapi uvicorn easyocr opencv-python-headless numpy pydantic python-multipart

Esempio di utilizzo:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import logging
import re
import time
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("anpr_server")

# Regex per targhe italiane (AA123BB)
ITALIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$")

# Configurazione OCR
UPSCALE_FACTOR = 2
MIN_CONFIDENCE = 40.0

# Response model


class PlateResponse(BaseModel):
    plate: Optional[str]
    confidence: float
    box: Optional[List[int]]
    attempts: List[Dict[str, str]]


# FastAPI app
app = FastAPI(
    title="ANPR Server Italiano",
    description="Riconoscimento targhe italiane avanzato in locale",
    version="1.5"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EasyOCR reader placeholder
reader: easyocr.Reader = None


@app.on_event("startup")
async def load_easyocr_model():
    global reader
    logger.info("[startup] Caricamento modello EasyOCR...")
    t0 = time.perf_counter()
    reader = easyocr.Reader(["en", "it"], gpu=False)
    logger.info(
        f"[startup] EasyOCR caricato in {time.perf_counter() - t0:.1f}s")


def preprocess_gray(frame: np.ndarray, width: int = 640) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    scale = width / float(w)
    resized = cv2.resize(frame, (width, int(h * scale)))

    # Denoising (meglio in condizioni notturne o rumorose)
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, 10, 10, 7, 21)

    # Bilanciamento automatico del bianco
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    balanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Correzione gamma per immagini troppo scure/chiare
    gamma = 1.5  # più alta per schiarire
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma *
                     255 for i in np.arange(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(balanced, table)

    gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    return gamma_corrected, gray


def find_plate_region(gray: np.ndarray, min_area: int = 1000, ar_range: Tuple[float, float] = (2.0, 8.0)) -> Optional[Tuple[int, int, int, int]]:
    sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob = cv2.convertScaleAbs(sob)
    _, bw = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern)
    if cv2.countNonZero(closed) < min_area:
        closed = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h) if h > 0 else 0
        if ar_range[0] <= ar <= ar_range[1]:
            boxes.append((x, y, w, h))
    return max(boxes, key=lambda b: b[2]*b[3]) if boxes else None


def four_point_transform(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    pad = int(0.1 * w)  # padding laterale 10%
    x = max(0, x - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    src = np.array([[x, y], [x + w, y], [x + w, y + h],
                   [x, y + h]], dtype="float32")
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))


def normalize_plate(plate: str) -> str:
    return plate.replace("0", "O").replace("1", "I")


def ocr_plate(crop: np.ndarray) -> Tuple[Optional[str], float, List[Dict[str, str]]]:
    h, w = crop.shape[:2]
    scaled = cv2.resize(crop, (w * UPSCALE_FACTOR, h *
                        UPSCALE_FACTOR), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    # Diverse versioni dell'immagine per OCR
    _, binarized = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binarized)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    equalized = cv2.equalizeHist(gray)
    light_mask = cv2.inRange(gray, 200, 255)
    masked = cv2.bitwise_and(gray, gray, mask=light_mask)

    inputs = [
        ("binarized", binarized),
        ("inverted", inverted),
        ("enhanced", enhanced),
        ("equalized", equalized),
        ("light_masked", masked)
    ]

    candidates: List[Tuple[str, float]] = []
    attempts: List[Dict[str, str]] = []

    for label, img in inputs:
        try:
            results = reader.readtext(img, detail=1)
            for _, text, conf in results:
                raw = text
                plate = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                plate = normalize_plate(plate)
                conf_pct = round(conf * 100, 1)
                attempts.append({
                    "source": label,
                    "raw": raw,
                    "normalized": plate,
                    "confidence": f"{conf_pct:.1f}%"
                })
                if ITALIAN_PLATE_REGEX.match(plate):
                    candidates.append((plate, conf_pct))
        except Exception as e:
            logger.error("[ocr] errore OCR: %s", e)

    if not candidates:
        return None, 0.0, attempts

    plate, conf = max(candidates, key=lambda x: x[1])
    return (plate, conf, attempts) if conf >= MIN_CONFIDENCE else (None, conf, attempts)


@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok"}


@app.post("/recognize", response_model=PlateResponse, tags=["ANPR"])
async def recognize_plate(file: UploadFile = File(..., description="JPEG/PNG image")):
    logger.info("[recognize] ricezione file %s", file.filename)
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Formato non supportato")
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400, detail="Impossibile decodificare immagine")

    vis, gray = preprocess_gray(img)

    full_plate, full_conf, full_attempts = ocr_plate(vis)
    if full_plate:
        logger.info(
            f"[recognize] (full) plate={full_plate} conf={full_conf:.1f}%")
        return {"plate": full_plate, "confidence": round(full_conf, 1), "box": None, "attempts": full_attempts}

    rect = find_plate_region(gray)
    if rect:
        x, y, w, h = rect
        crop = four_point_transform(vis, rect)
        plate, conf, crop_attempts = ocr_plate(crop)
        logger.info(
            f"[recognize] (crop) plate={plate} conf={conf:.1f}% box={rect}")
        return {"plate": plate, "confidence": round(conf, 1), "box": [x, y, w, h], "attempts": crop_attempts}

    logger.info("[recognize] Nessuna targa trovata")
    return {"plate": None, "confidence": 0.0, "box": None, "attempts": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
