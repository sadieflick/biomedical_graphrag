#!/bin/bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/main.py