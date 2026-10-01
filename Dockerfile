FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Runtime libraries only — no C++/CMake build toolchain needed because dlib
# comes from the prebuilt dlib-bin wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libopenblas0-pthread \
    liblapack3 \
    unixodbc \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
       | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
       > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps face-recognition==1.3.0

COPY . .

RUN mkdir -p /app/data/encodings /app/data/uploads

ENV PORT=10000

CMD ["sh", "-c", "gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:${PORT} Script:app"]
