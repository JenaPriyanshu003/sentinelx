# Use a pre-built PyTorch CPU image — saves ~15 min of pip install time!
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install remaining dependencies (torch is already included in base image)
COPY requirements.render.txt .
RUN pip install --no-cache-dir -r requirements.render.txt

# Copy the backend code and models
COPY backend/ ./backend/
COPY models/ ./models/

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
