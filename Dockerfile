# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application code to /app
COPY app /app

# Create necessary directories
RUN mkdir -p /app/staticFiles/uploads

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE $PORT

# Set working directory to the Flask app
WORKDIR /app

# Run the application
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT app:app"]