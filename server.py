#!/usr/bin/env python3
"""
ANPR Server per targhe italiane - Versione Migliorata
--------------------------------------------------

Espone un’API REST che riceve immagini via POST e risponde
con la targa riconosciuta, la confidenza e la bounding box.

Migliorie implementate:
- Rilevamento regione 2-stage con EAST text detector e Sobel+MSER fallback
- Preprocessing multi-variant (CLAHE, AdaptiveThreshold, Bilateral Filter)
- Super-resolution upscaling opzionale via OpenCV DNN SuperRes
- Ensemble OCR: EasyOCR (GPU se disponibile) + Tesseract fallback
- Filtraggio avanzato candidate con punteggi ponderati
- Caching del modello OCR & detector
- Logging dettagliato e metriche di performance

Dipendenze:
    pip install fastapi uvicorn easyocr pytesseract opencv-python-headless opencv-contrib-python numpy pydantic python-multipart
    apt-get install tesseract-ocr

Esempio di utilizzo:
    uvicorn anpr_server_improved:app --host 0.0.0.0 --port 8000
"""

import logging
import re
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np
import easyocr
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("anpr_server_improved")

# ─── Regex per targhe italiane (AA123BB) ───────────────────────────────────
ITALIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$")

# ─── Configurazione OCR ─────────────────────────────────────────────────────
UPSCALE_FACTOR = 2  # fattore di ingrandimento dinamico
MIN_CONFIDENCE = 40.0  # soglia minima accettabile
USE_GPU_OCR = True  # EasyOCR su GPU se disponibile

# ─── Response model ────────────────────────────────────────────────────────


class PlateResponse(BaseModel):
    plate: Optional[str]
    confidence: float
    box: Optional[List[int]]  # [x, y, w, h]


# ─── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(
    title="ANPR Server Italiano Migliorato",
    description="Riconoscimento targhe italiane con metodi avanzati",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Modelli caricati in startup ──────────────────────────────────────────
reader: easyocr.Reader = None
east_net: cv2.dnn_Net = None


@app.on_event("startup")
async def load_models():
    global reader, east_net
    logger.info("[startup] Caricamento modelli in corso...")
    t0 = time.perf_counter()

    # EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=USE_GPU_OCR)

    # EAST text detector
    east_model_path = 'frozen_east_text_detection.pb'
    east_net = cv2.dnn.readNet(east_model_path)

    logger.info(
        f"[startup] Modelli caricati in {time.perf_counter() - t0:.1f}s")

# ─── Health-check ─────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": app.version}

# ─── Utility: Super-Resolution (opzionale) ─────────────────────────────────
try:
    from cv2 import dnn_superres
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel('ESPCN_x2.pb')
    sr.setModel('espcn', 2)

    def upscale(img: np.ndarray) -> np.ndarray:
        return sr.upsample(img)
    logger.info("[startup] SuperRes modello caricato")
except Exception:
    def upscale(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (img.shape[1]*UPSCALE_FACTOR, img.shape[0]*UPSCALE_FACTOR), interpolation=cv2.INTER_CUBIC)
    logger.warning("[startup] SuperRes non disponibile, uso resize standard")

# ─── Rilevamento regione targa con EAST detector ─────────────────────────


def detect_text_regions_east(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Ritorna bounding box candidate rilevate da EAST text detector."""
    H, W = image.shape[:2]
    newW, newH = (320, 320)
    rW, rH = W/float(newW), H/float(newH)
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (newW, newH), (123.68, 116.78, 103.94), True, False)
    east_net.setInput(blob)
    (scores, geometry) = east_net.forward(
        ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

    rects = []
    confidences = []
    for y in range(scores.shape[2]):
        for x in range(scores.shape[3]):
            score = scores[0, 0, y, x]
            if score < 0.5:
                continue
            offsetX, offsetY = x*4.0, y*4.0
            angle = geometry[0, 4, y, x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            endX = int(offsetX + cos *
                       geometry[0, 1, y, x] + sin * geometry[0, 2, y, x])
            endY = int(offsetY - sin *
                       geometry[0, 1, y, x] + cos * geometry[0, 2, y, x])
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append(
                (int(startX*rW), int(startY*rH), int(w*rW), int(h*rH)))
            confidences.append(float(score))
    picks = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)
    boxes = [rects[i[0]] for i in picks]
    return boxes

# ─── OCR multipasso e ensemble ──────────────────────────────────────────────


def ocr_ensemble(crop: np.ndarray) -> Tuple[Optional[str], float]:
    """Applica diversi preprocess e OCR engines, aggrega i risultati."""
    variants = []
    # Original upscaled
    img_up = upscale(crop)
    variants.append(img_up)
    # CLAHE
    lab = cv2.cvtColor(img_up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    variants.append(cv2.cvtColor(cv2.merge((clahe, a, b)), cv2.COLOR_LAB2BGR))
    # Adaptive Thresh
    gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    variants.append(cv2.cvtColor(cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5), cv2.COLOR_GRAY2BGR))
    # Bilateral filter
    variants.append(cv2.bilateralFilter(img_up, 9, 75, 75))

    candidates = {}
    for var in variants:
        # EasyOCR
        try:
            results = reader.readtext(var, detail=1)
            for _, text, conf in results:
                plate = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                if ITALIAN_PLATE_REGEX.match(plate):
                    score = conf*100 * 0.7
                    candidates[plate] = max(candidates.get(plate, 0), score)
        except Exception as e:
            logger.debug(f"[ocr_ez] errore OCR EasyOCR: {e}")
        # Tesseract
        try:
            cfg = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            txt = pytesseract.image_to_string(var, config=cfg)
            plate = re.sub(r'[^A-Za-z0-9]', '', txt).upper()
            if ITALIAN_PLATE_REGEX.match(plate):
                conf_t = np.mean(pytesseract.image_to_data(
                    var, config=cfg, output_type=pytesseract.Output.DICT)['conf'])
                score = conf_t * 0.3
                candidates[plate] = max(candidates.get(plate, 0), score)
        except Exception as e:
            logger.debug(f"[ocr_tes] errore OCR Tesseract: {e}")

    if not candidates:
        return None, 0.0
    best_plate, best_score = max(candidates.items(), key=lambda x: x[1])
    if best_score < MIN_CONFIDENCE:
        return None, best_score
    return best_plate, best_score

# ─── Endpoint principale ───────────────────────────────────────────────────


@app.post("/recognize", response_model=PlateResponse, tags=["ANPR"])
async def recognize_plate(file: UploadFile = File(..., description="JPEG/PNG image")):
    logger.info(f"[recognize] Ricevuto file {file.filename}")
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Formato non supportato")
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400, detail="Impossibile decodificare immagine")

    t_start = time.perf_counter()
    # 1) rilevamento testo con EAST
    boxes = detect_text_regions_east(img)
    # fallback Sobel+MSER
    if not boxes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create(_min_area=500, _max_area=10000)
        regs, _ = mser.detectRegions(gray)
        boxes = [cv2.boundingRect(r) for r in regs]
    best_box, best_plate, best_conf = None, None, 0.0
    for rect in boxes:
        x, y, w, h = rect
        roi = img[y:y+h, x:x+w]
        plate, conf = ocr_ensemble(roi)
        if plate and conf > best_conf:
            best_conf, best_plate, best_box = conf, plate, rect

    duration = time.perf_counter() - t_start
    logger.info(
        f"[recognize] plate={best_plate} conf={best_conf:.1f}% box={best_box} time={duration:.2f}s")
    if not best_box:
        return {"plate": None, "confidence": 0.0, "box": None}
    x, y, w, h = best_box
    return {"plate": best_plate, "confidence": round(best_conf, 1), "box": [x, y, w, h]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("anpr_server_improved:app",
                host="0.0.0.0", port=8000, reload=True)
