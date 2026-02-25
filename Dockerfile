FROM python:3.10-slim

# Install system dependencies (OpenCV requires libgl1)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the backend requirements (if any) or directly install dependencies
# We will install known dependencies based on main.py imports
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code and models
COPY backend/ ./backend/
COPY models/ ./models/

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
