#!/bin/bash

# Ensure we shut down both if you hit CTRL+C
trap 'kill %1' SIGINT

echo "Starting SentinelX Backend on http://127.0.0.1:8000"
source venv/bin/activate
# Run backend in the background
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting SentinelX Frontend on http://127.0.0.1:5173"
cd frontend
# Run frontend in the foreground
npm run dev
