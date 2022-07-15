#!/bin/bash
docker exec --workdir /home/logreg -ti logreg_cont python src/featurize.py data/prepared data/features