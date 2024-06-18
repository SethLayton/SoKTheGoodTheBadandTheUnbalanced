#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd CFAD/LFCC-LCNN/lfcc-lcnn

echo "Starting LFCCLCNN-LibriSpeech25/75 Train"
python train.py \
    --distribution 25-75 \
    --trainlabelfile ../../CFADTrain/protocols \
    --database_path ../../../../DataSets/CFAD/trn

echo "Starting LFCCLCNN-LibriSpeech50/50 Train"
python train.py \
    --distribution 50-50 \
    --trainlabelfile ../../CFADTrain/protocols \
    --database_path ../../../../DataSets/CFAD/trn


echo "Starting LFCCLCNN-LibriSpeech75/25 Train"
python train.py \
    --distribution 75-25 \
    --trainlabelfile ../../CFADTrain/protocols \
    --database_path ../../../../DataSets/CFAD/trn


echo "Starting LFCCLCNN-LibriSpeech90/10 Train"
python train.py \
    --distribution 90-10 \
    --trainlabelfile ../../CFADTrain/protocols \
    --database_path ../../../../DataSets/CFAD/trn


conda deactivate

date