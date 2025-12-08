#!/bin/bash

# TRELLIS Server Startup Script

echo "========================================"
echo "   TRELLIS 3D Generation Server"
echo "========================================"
echo ""

# Check if required files exist
if [ ! -f "server.py" ]; then
    echo "Error: server.py not found!"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Get the server IP address
echo "Detecting server IP address..."
SERVER_IP=$(hostname -I | awk '{print $1}')
echo "Server IP: $SERVER_IP"
echo ""

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q -r server_requirements.txt

echo ""
echo "========================================"
echo "Starting server..."
echo "========================================"
echo ""
echo "Server will be accessible at:"
echo "  - Local:   http://localhost:8000"
echo "  - Network: http://$SERVER_IP:8000"
echo ""
echo "API Documentation will be available at:"
echo "  - http://$SERVER_IP:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================"
echo ""

# Start the server
python server.py

