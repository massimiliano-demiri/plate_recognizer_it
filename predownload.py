# predownload.py
import easyocr

# Scarica e mette in cache il modello 'en' di EasyOCR
easyocr.Reader(['en'], gpu=False)
