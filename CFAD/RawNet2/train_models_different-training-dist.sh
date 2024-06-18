#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd CFAD/RawNet2/RawNet2

echo "Starting RawNet2-CFAD 25/75 Train"
python main.py \
    --protocol_path ../../CFADTrain/pkl \
    --database_path ../../../../DataSets/CFAD/trn \
    --iteration_number 25-75

echo "Starting RawNet2-CFAD 5050 Train"
python main.py \
    --protocol_path ../../CFADTrain/pkl \
    --database_path ../../../../DataSets/CFAD/trn \
    --iteration_number 50-50

echo "Starting RawNet2-CFAD 7525 Train"
python main.py \
    --protocol_path ../../CFADTrain/pkl \
    --database_path ../../../../DataSets/CFAD/trn \
    --iteration_number 75-25
    
echo "Starting RawNet2-CFAD 90-10 Train"
python main.py \
    --protocol_path ../../CFADTrain/pkl \
    --database_path ../../../../DataSets/CFAD/trn \
    --iteration_number 90-10


conda deactivate

date