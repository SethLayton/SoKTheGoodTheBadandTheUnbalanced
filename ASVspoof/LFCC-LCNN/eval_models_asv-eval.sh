#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd ASVspoof/LFCC-LCNN/LFCC-LCNN

echo "Starting LFCC-LCNN Eval using 25-75 training distribution against ASVspoof Eval Dataset"
python project/baseline_LA/score.py \
        --pretrained_model ../models/25-75.pt \
        --scores_file ../results/25-75_asv21-eval.score \
        --designator $1
        

echo "Starting LFCC-LCNN Eval using 50-50 training distribution against ASVspoof Eval Dataset"
python project/baseline_LA/score.py \
        --pretrained_model ../models/50-50.pt \
        --scores_file ../results/50-50_asv21-eval.score \
        --designator $1

echo "Starting LFCC-LCNN Eval using 75-25 training distribution against ASVspoof Eval Dataset"
python project/baseline_LA/score.py \
        --pretrained_model ../models/75-25.pt \
        --scores_file ../results/275-25_asv21-eval.score \
        --designator $1

echo "Starting LFCC-LCNN Eval using 90-10 training distribution against ASVspoof Eval Dataset"
python project/baseline_LA/score.py \
        --pretrained_model ../models/90-10.pt \
        --scores_file ../results/90-10_asv21-eval.score \
        --designator $1

conda deactivate
date