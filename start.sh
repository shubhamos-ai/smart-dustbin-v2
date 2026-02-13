#!/bin/bash

echo "Activating virtual environment..."
source /home/shubhamos/Documents/shubhamos3.11/bin/activate

echo "Changing directory..."
cd /home/shubhamos/Documents/jaydeep || exit

echo "Starting Smart Waste Detection Server..."
uvicorn app:app --host 0.0.0.0 --port 8000
