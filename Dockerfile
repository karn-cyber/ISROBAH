# HIMADRI — single-service deploy (FastAPI serves the API + the React build).
# Runtime data footprint is tiny (~17 MB): web/dist + data/outputs/real_faustini.
# The multi-GB raw DFSAR/DEM inputs are NOT needed at runtime (see .dockerignore).
FROM python:3.12

# Full python:3.12 (Debian) image: it ships the system shared libraries the
# rasterio / pyproj / geopandas wheels link against at runtime (libexpat,
# libsqlite3, etc.). The -slim image omits these and fails with
# "libexpat.so.1: cannot open shared object file".
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the precomputed real run + the built frontend
COPY src/ ./src/
COPY config/ ./config/
COPY web/dist/ ./web/dist/
COPY data/outputs/ ./data/outputs/

ENV PYTHONPATH=/app/src
ENV PORT=8000
EXPOSE 8000

# Hosts (Render/Railway/HF/Fly) inject $PORT; default 8000 locally.
CMD ["sh", "-c", "uvicorn himadri.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
