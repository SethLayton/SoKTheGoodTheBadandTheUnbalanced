#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate ssl1

cd ASVspoof/wav2vec/SSL_Anti-spoofing

echo "Starting wav2vec-LibriSpeech25/75 Train"

python main_SSL_DF.py \
        --iteration_number 25-75 \
        --database_path ../../../../DataSets/ASVspoof/ASVspoofAndLibriSpeech \
        --protocol_path ../../ASVspoofTrain/protocols

echo "Starting wav2vec-LibriSpeech50/50 Train"

python main_SSL_DF.py \
        --iteration_number 50-50 \
        --database_path ../../../../DataSets/ASVspoof/ASVspoofAndLibriSpeech \
        --protocol_path ../../ASVspoofTrain/protocols

echo "Starting wav2vec-LibriSpeech75/25 Train"

python main_SSL_DF.py \
        --iteration_number 75-25 \
        --database_path ../../../../DataSets/ASVspoof/ASVspoofAndLibriSpeech \
        --protocol_path ../../ASVspoofTrain/protocols

echo "Starting wav2vec-LibriSpeech90/10 Train"

python main_SSL_DF.py \
        --iteration_number 90-10 \
        --database_path ../../../../DataSets/ASVspoof/ASVspoofAndLibriSpeech \
        --protocol_path ../../ASVspoofTrain/protocols

conda deactivate

date