FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000
ENV LOG_LEVEL=INFO
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.service:app --host 0.0.0.0 --port $PORT --log-level $LOG_LEVEL"]