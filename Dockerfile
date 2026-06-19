# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install basic system dependencies required by OpenCV & Matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY server.py .
COPY best_densenet_model.keras .
COPY templates/ templates/
COPY static/ static/

# Expose Flask default port
EXPOSE 5000

# Set environment configurations
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Use Gunicorn as the WSGI server for production deployment.
# Increased timeout to 120 seconds to allow the Keras model to load on boot.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app", "--timeout", "120"]
