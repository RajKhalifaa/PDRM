# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Chrome for Kaleido
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    unzip \
    xvfb \
    dos2unix \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libxss1 \
    libxtst6 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/* \
    || echo "Chrome install failed, continuing without Chrome"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Copy and set permissions for entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
COPY healthcheck.py /app/healthcheck.py

# Fix line endings for scripts and set permissions
RUN dos2unix /app/entrypoint.sh /app/healthcheck.py || true
RUN chmod +x /app/entrypoint.sh /app/healthcheck.py

# Create necessary directories
RUN mkdir -p /app/data

# Create a user to run the application (security best practice for ECS)
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Set environment variables for ECS compatibility
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
ENV DISPLAY=:99
ENV KALEIDO_EXECUTABLE_PATH=/usr/bin/google-chrome-stable

# Health check for ECS
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 /app/healthcheck.py

# Run the application using entrypoint script
CMD ["/app/entrypoint.sh"]
