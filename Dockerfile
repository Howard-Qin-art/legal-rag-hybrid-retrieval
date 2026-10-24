FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# sentence-transformers 会拉 torch（约 2GB），默认不装。
# 需要真实句向量时：docker compose build --build-arg WITH_TORCH=true api
ARG WITH_TORCH=false
RUN if [ "$WITH_TORCH" = "true" ]; then pip install --no-cache-dir sentence-transformers; fi

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
