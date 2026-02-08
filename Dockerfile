FROM python:3.11-slim

WORKDIR /app

# System libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

COPY --chown=user . .

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]