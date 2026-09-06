# Dummy remote AI judge: small Python/FastAPI service.
FROM python:3.11-slim

WORKDIR /app

# curl is handy for debugging but not required; keep the image lean.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# Model registry (alias -> modality). The judge reads this to decide whether to
# attach a rendered chart image (vision) vs send text only.
COPY models.yaml ./models.yaml

EXPOSE 5000

# uvicorn on 0.0.0.0:5000 so Kayenta (in-network) can reach POST /judge.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
