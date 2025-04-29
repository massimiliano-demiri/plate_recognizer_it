#!/usr/bin/env python3
"""
ANPR Server per targhe italiane
-------------------------------

Espone un’API REST che riceve immagini via POST e risponde
con la targa riconosciuta, la confidenza e la bounding box.

Dipendenze:
    pip install fastapi uvicorn easyocr opencv-python numpy

Esempio di utilizzo:
    uvicorn server:app --host 0.0.0.0 --port 8000

POST /recognize
    Form-field “file”: file immagine JPEG/PNG
Risposta JSON:
    {
      "plate": "AA123BB",
      "confidence": 93.5,
      "box": [x, y, w, h]
    }
Oppure, se non trovata:
    {
      "plate": null,
      "confidence": 0.0,
      "box": null
    }
"""

import re
import io
import time
from typing import Optional

import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Inizializza il reader EasyOCR (solo CPU)
reader = easyocr.Reader(['en'], gpu=False)

# Regex per targhe italiane (AA123BB)
ITALIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{3}[A-Z]{2}$')

app = FastAPI(title="ANPR Server Italiano",
              description="Riconoscimento targhe italiane in locale",
              version="1.0")


def preprocess_gray(frame: np.ndarray, width: int = 640) -> (np.ndarray, np.ndarray):
    """Ridimensiona a width, converte a gray + CLAHE."""
    h, w = frame.shape[:2]
    scale = width / float(w)
    frame_resized = cv2.resize(frame, (width, int(h * scale)))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return frame_resized, gray_eq


def find_plate_region(gray: np.ndarray, min_area: int = 4500) -> Optional[tuple]:
    """Trova il rettangolo più grande compatibile con targhe italiane."""
    # Sobel orizzontale
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad)
    # Otsu threshold
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
    # Contorni
    cnts, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h) if h > 0 else 0
        if 3.5 <= ar <= 6.0 and area > best_area:
            best_area = area
            best = (x, y, w, h)
    return best


def four_point_transform(img: np.ndarray, rect: tuple) -> np.ndarray:
    """Ritaglia in prospettiva il rettangolo rect=(x,y,w,h)."""
    x, y, w, h = rect
    src = np.array([[x, y], [x + w, y], [x + w, y + h],
                   [x, y + h]], dtype="float32")
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))


def ocr_plate(crop: np.ndarray) -> (Optional[str], float):
    """Applica EasyOCR e ritorna (plate, confidence)."""
    results = reader.readtext(crop, detail=1)
    cands = []
    for _, text, conf in results:
        plate = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        if ITALIAN_PLATE_REGEX.match(plate):
            cands.append((plate, conf * 100))
    if not cands:
        return None, 0.0
    return max(cands, key=lambda x: x[1])


@app.post("/recognize")
async def recognize_plate(image: UploadFile = File(...)):
    """Endpoint che riceve un’immagine e risponde con plate, confidence, box."""
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Formato non supportato")

    data = await image.read()
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
        box = [int(x), int(y), int(w), int(h)]
    else:
        plate, conf, box = None, 0.0, None

    elapsed = (time.perf_counter() - t0) * 1000
    logging.info(
        f"Processed in {elapsed:.1f} ms – Plate: {plate} ({conf:.1f}%) Box: {box}")

    return JSONResponse({
        "plate": plate,
        "confidence": round(conf, 1),
        "box": box
    })


@app.get("/")
def health_check():
    return {"status": "ok", "message": "ANPR Server in esecuzione"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
