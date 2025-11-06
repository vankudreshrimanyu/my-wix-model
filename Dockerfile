# Optional: use if you want to deploy with Docker
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy model file at build time OR mount via volume / download at runtime
# COPY xception_torch_best.pt /app/

COPY app.py /app/

ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
