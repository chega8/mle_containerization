#!/bin/bash
docker exec --workdir /home/logreg -ti logreg_cont python src/train.py data/features