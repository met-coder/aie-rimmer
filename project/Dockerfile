FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

ENV PORT=8000
ENV LOG_LEVEL=INFO
EXPOSE 8000
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
