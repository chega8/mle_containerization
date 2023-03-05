#!/bin/bash
./src/scripts/prepare.sh && \
./src/scripts/featurize.sh && \
./src/scripts/train.sh && \
./src/scripts/evaluate.sh