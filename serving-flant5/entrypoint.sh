#!/bin/bash
export PORT=$AIP_HTTP_PORT
echo $PORT
uvicorn main:app --proxy-headers --host 0.0.0.0 --port $PORT