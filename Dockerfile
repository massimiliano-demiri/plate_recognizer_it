FROM python:3.10-slim

# Dipendenze di sistema per OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice
COPY . .

EXPOSE 8000

# Avvia uvicorn sul modulo server.py (app = FastAPI())
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
