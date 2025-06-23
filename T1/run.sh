#! /bin/bash

set -xe
# Run all the scripts in the src folder
python3 src/corpus_analysis.py
python3 src/text_representation.py
python3 src/clustering.py
python3 src/cluster_analysis.py

