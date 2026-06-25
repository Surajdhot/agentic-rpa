# Conductor — single-container image running the Streamlit UI.
# Uses python:3.11-slim and installs Chromium (with OS deps) via Playwright.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Install Python deps, then the Chromium browser plus its system libraries.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

COPY . .

# Run as a non-root user; give it the browsers and a writable output sandbox.
RUN useradd --create-home --uid 1000 conductor \
    && mkdir -p /app/output \
    && chown -R conductor:conductor /app /ms-playwright
USER conductor

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
