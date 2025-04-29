FROM python:3.10-slim
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
