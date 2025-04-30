#!/usr/bin/env python3
"""
ANPR Server per targhe italiane (versione smart con correzione luce e OCR avanzato)
--------------------------------------------------------------------------

Espone un’API REST che riceve immagini via POST e risponde
con la targa riconosciuta, la confidenza, la bounding box e i tentativi OCR.

Dipendenze:
    pip install fastapi uvicorn easyocr opencv-python-headless numpy pydantic python-multipart

Esecuzione:
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
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smart_anpr")

ITALIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{2,4}[A-Z]{1,3}$")

MIN_CONFIDENCE = 30.0
UPSCALE_FACTOR = 2


class PlateResponse(BaseModel):
    plate: Optional[str]
    confidence: float
    box: Optional[List[int]]
    attempts: List[Dict[str, str]]


app = FastAPI(title="Smart ANPR Server", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reader: easyocr.Reader = None


@app.on_event("startup")
async def load_ocr():
    global reader
    logger.info("Caricamento modello OCR...")
    t0 = time.perf_counter()
    reader = easyocr.Reader(["en", "it"], gpu=False)
    logger.info(f"Modello OCR caricato in {time.perf_counter() - t0:.2f}s")


def adjust_gamma(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma *
                     255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def auto_white_balance(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(lab[:, :, 1])
    avg_b = np.average(lab[:, :, 2])
    lab[:, :, 1] -= ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    lab[:, :, 2] -= ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_variants(img: np.ndarray) -> Dict[str, np.ndarray]:
    variants = {}
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants["gray"] = gray
    variants["clahe"] = clahe.apply(gray)
    variants["gamma"] = adjust_gamma(gray, 1.5)
    variants["inverted"] = cv2.bitwise_not(gray)
    variants["binarized"] = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return variants


def normalize_plate(plate: str) -> str:
    return plate.replace("0", "O").replace("1", "I")


def extract_candidates(img_variants: Dict[str, np.ndarray]) -> Tuple[Optional[str], float, List[Dict[str, str]]]:
    attempts = []
    candidates = []
    for label, img in img_variants.items():
        try:
            results = reader.readtext(img, detail=1)
            for _, text, conf in results:
                raw = text
                plate = re.sub(r'[^A-Za-z0-9]', '', raw).upper()
                plate = normalize_plate(plate)
                conf_pct = round(conf * 100, 1)
                is_valid = bool(ITALIAN_PLATE_REGEX.match(plate))
                attempts.append({"source": label, "raw": raw, "normalized": plate,
                                "confidence": f"{conf_pct:.1f}%", "valid": str(is_valid)})
                if is_valid:
                    candidates.append((plate, conf_pct))
        except Exception as e:
            logger.warning(f"[OCR:{label}] Errore: {e}")
    if not candidates:
        return None, 0.0, attempts
    best_plate, best_conf = max(candidates, key=lambda x: x[1])
    return (best_plate, best_conf, attempts) if best_conf >= MIN_CONFIDENCE else (None, best_conf, attempts)


def detect_plate_regions(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, thresh = cv2.threshold(
        sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        ar = w / h if h > 0 else 0
        if area > 1500 and 2 < ar < 8:
            boxes.append((x, y, w, h))
    return boxes


def extract_region(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return img[y:y+h, x:x+w]


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/recognize", response_model=PlateResponse)
async def recognize(file: UploadFile = File(...)):
    logger.info(f"[recognize] Ricevuto: {file.filename}")
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Formato non supportato")

    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Immagine non valida")

    img = auto_white_balance(img)
    full_attempts_img = preprocess_variants(img)
    full_plate, full_conf, full_attempts = extract_candidates(
        full_attempts_img)
    if full_plate:
        logger.info(f"[full image] plate={full_plate} conf={full_conf:.1f}%")
        return {"plate": full_plate, "confidence": round(full_conf, 1), "box": None, "attempts": full_attempts}

    regions = detect_plate_regions(img)
    for rect in regions:
        region = extract_region(img, rect)
        region_attempts = preprocess_variants(region)
        plate, conf, crop_attempts = extract_candidates(region_attempts)
        if plate:
            logger.info(f"[region] plate={plate} conf={conf:.1f}% box={rect}")
            return {"plate": plate, "confidence": round(conf, 1), "box": list(rect), "attempts": crop_attempts}

    logger.info("[recognize] Nessuna targa rilevata")
    return {"plate": None, "confidence": 0.0, "box": None, "attempts": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
