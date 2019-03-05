#!/bin/sh

gunicorn app:app --workers 1 --worker-connections 1000 --timeout 600 --log-level warning --bind 0.0.0.0:20019
