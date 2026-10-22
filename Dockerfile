FROM docker.io/library/python:3.14-slim

# XGBoost needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Listen on all interfaces inside the container; docker-compose.yml only
# publishes the port on the host's 127.0.0.1.
ENV HP_CHECKER_HOST=0.0.0.0 \
    HP_CHECKER_PORT=5487
EXPOSE 5487

# Flask's dev server stops cleanly on Ctrl+C (SIGINT) but ignores SIGTERM as PID 1
STOPSIGNAL SIGINT
CMD ["python", "app.py"]
