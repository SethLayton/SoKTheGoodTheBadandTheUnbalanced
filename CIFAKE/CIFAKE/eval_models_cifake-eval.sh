#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate cifake

cd CIFAKE/CIFAKE/RawCNN

echo "Starting CIFAKE Eval using 25-75 training distribution against CIFAKE Eval Dataset"
python score.py \
        --pretrained_model ../models/25-75.h5 \
        --database_path ../../../../DataSets/CIFAKE/CIFAKE/test/ \
        --output ../results/25-75_cifake-eval.scores \
        --designator $1

echo "Starting CIFAKE Eval using 50-50 training distribution against CIFAKE Eval Dataset"
python score.py \
        --pretrained_model ../models/50-50.h5 \
        --database_path ../../../../DataSets/CIFAKE/CIFAKE/test/ \
        --output ../results/50-50_cifake-eval.scores \
        --designator $1

echo "Starting CIFAKE Eval using 75-25 training distribution against CIFAKE Eval Dataset"
python score.py \
        --pretrained_model ../models/75-25.h5 \
        --database_path ../../../../DataSets/CIFAKE/CIFAKE/test/ \
        --output ../results/75-25_cifake-eval.scores \
        --designator $1

echo "Starting CIFAKE Eval using 90-10 training distribution against CIFAKE Eval Dataset"
python score.py \
        --pretrained_model ../models/90-10.h5 \
        --database_path ../../../../DataSets/CIFAKE/CIFAKE/test/ \
        --output ../results/90-10_cifake-eval.scores \
        --designator $1


conda deactivate

date