#!/bin/bash

source .venv/bin/activate
streamlit run app.py --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
