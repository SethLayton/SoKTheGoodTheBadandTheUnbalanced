#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate cifake

cd CIFAKE/CIFAKE/RawCNN

echo "Starting CIFAKE REAL ONLY Eval using 25-75 training distribution against CIFAKE REAL ONLY Eval Dataset"
python score.py \
        --pretrained_model ../models/25-75.h5 \
        --database_path ../../../../DataSets/CIFAKE/STL10/ \
        --output ../results/25-75_ro-eval.scores \
        --designator $1

echo "Starting CIFAKE REAL ONLY Eval using 50-50 training distribution against CIFAKE REAL ONLY Eval Dataset"
python score.py \
        --pretrained_model ../models/50-50.h5 \
        --database_path ../../../../DataSets/CIFAKE/STL10/ \
        --output ../results/50-50_ro-eval.scores \
        --designator $1

echo "Starting CIFAKE REAL ONLY Eval using 75-25 training distribution against CIFAKE REAL ONLY Eval Dataset"
python score.py \
        --pretrained_model ../models/75-25.h5 \
        --database_path ../../../../DataSets/CIFAKE/STL10/ \
        --output ../results/75-25_ro-eval.scores \
        --designator $1

echo "Starting CIFAKE REAL ONLY Eval using 90-10 training distribution against CIFAKE REAL ONLY Eval Dataset"
python score.py \
        --pretrained_model ../models/90-10.h5 \
        --database_path ../../../../DataSets/CIFAKE/STL10/ \
        --output ../results/90-10_ro-eval.scores \
        --designator $1


conda deactivate

date