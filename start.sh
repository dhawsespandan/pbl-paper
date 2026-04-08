#!/bin/bash
set -e

# Start Python model service
cd agrisense-ai-main/model-service
uvicorn main:app --host 127.0.0.1 --port 8000 &
MODEL_PID=$!

# Start Node.js API server
cd ../server
node app.js &
SERVER_PID=$!

# Start React frontend
cd ../client
npm run dev

# Cleanup on exit
kill $MODEL_PID $SERVER_PID 2>/dev/null
