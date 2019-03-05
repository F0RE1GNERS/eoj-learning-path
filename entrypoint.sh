#!/bin/sh

gunicorn app:app --workers 16 --worker-connections 1000 --error-logfile /app/gunicorn.log \
    --timeout 600 --log-level warning --bind 0.0.0.0:20019
