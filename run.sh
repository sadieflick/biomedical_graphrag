#!/bin/bash
# source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
chainlit run src/app.py -w
python3.12 src/app.py