#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd ASVspoof/RawNet2/RawNet2

echo "Starting RawNet2-LibriSpeech25/75 Train"

python main.py \
        --iteration_number 25-75 \
        --database_path ../../../../DataSets/ASVspoof/ASVspoofAndLibriSpeech \
        --protocol_path ../../ASVspoofTrain/pkl

echo "Starting RawNet2-LibriSpeech50/50 Train"

python main.py \
        --iteration_number 50-50 \
        --database_path ../../../../DataSets/ASVspoof/ASVspoofAndLibriSpeech \
        --protocol_path ../../ASVspoofTrain/pkl

echo "Starting RawNet2-LibriSpeech75/25 Train"

python main.py \
        --iteration_number 75-25 \
        --database_path ../../../../DataSets/ASVspoof/ASVspoofAndLibriSpeech \
        --protocol_path ../../ASVspoofTrain/pkl

echo "Starting RawNet2-LibriSpeech90/10 Train"

python main.py \
        --iteration_number 90-10 \
        --database_path ../../../../DataSets/ASVspoof/ASVspoofAndLibriSpeech \
        --protocol_path ../../ASVspoofTrain/pkl

conda deactivate

date