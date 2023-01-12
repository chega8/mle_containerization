#!/bin/bash
# docker exec --workdir /home/logreg -ti logreg_cont python data/models/logreg.pkl data/features
python src/evaluate.py data/models/model.pkl data/features