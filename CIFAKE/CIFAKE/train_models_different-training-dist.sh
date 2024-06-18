#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate cifake

cd CIFAKE/CIFAKE/RawCNN

echo "Starting RawCNN-CIFAKE 25/75 Train"
python train.py \
    --database_path ../../../../DataSets/CIFAKE/CIFAKE2575 \
    --iteration_number 25-75

echo "Starting RawCNN-CIFAKE 5050 Train"
python train.py \
    --database_path ../../../../DataSets/CIFAKE/CIFAKE5050 \
    --iteration_number 50-50

echo "Starting RawCNN-CIFAKE 7525 Train"
python train.py \
    --database_path ../../../../DataSets/CIFAKE/CIFAKE7525 \
    --iteration_number 75-25
    
echo "Starting RawCNN-CIFAKE 90-10 Train"
python train.py \
    --database_path ../../../../DataSets/CIFAKE/CIFAKE9010 \
    --iteration_number 90-10


conda deactivate

date