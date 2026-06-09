# Minimal CPU image that serves the trained model via the Gradio demo.
# Models are mounted/copied in at build time from ./models and ./artifacts.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch keeps the image small; the served model is LightGBM anyway.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY app.py ./
COPY models/ ./models/
COPY artifacts/ ./artifacts/

EXPOSE 7860
# Default to the Gradio demo; override CMD to run the FastAPI service instead:
#   docker run ... uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["python", "app.py"]
