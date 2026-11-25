FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 required for LightGBM OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY docs ./docs
COPY data ./data

RUN pip install --upgrade pip && \
    pip install -e ".[ui]"

EXPOSE 8000 8501

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
