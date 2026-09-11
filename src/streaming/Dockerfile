# Use an official lightweight Python runtime image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy project package definitions first to leverage Docker cache layering
COPY requirements.txt .

# Install dependencies cleanly without local caching
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the internal port that FastAPI runs on
EXPOSE 8000

# Spin up Uvicorn to host the app, mapping to 0.0.0.0 so external networks can connect
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
