#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd ASVspoof/LFCC-LCNN/LFCC-LCNN

echo "Starting LFCCLCNN-LibriSpeech25/75 Train"
python project/baseline_LA/main.py \
        --save-model-dir models/ \
        --verbose 0 --num-workers 6 \
        --epochs 100 --no-best-epochs 50 \
        --batch-size 64 \
        --sampler block_shuffle_by_length \
        --lr-decay-factor 0.5 \
        --lr-scheduler-type 1 --lr 0.0003 \
        --seed 1000 \
        --iteration_numbers 25-75

echo "Starting LFCCLCNN-LibriSpeech50/50 Train"
python project/baseline_LA/main.py \
        --save-model-dir models/ \
        --verbose 0 --num-workers 6 \
        --epochs 100 --no-best-epochs 50 \
        --batch-size 64 \
        --sampler block_shuffle_by_length \
        --lr-decay-factor 0.5 \
        --lr-scheduler-type 1 --lr 0.0003 \
        --seed 1000 \
        --iteration_numbers 50-50

echo "Starting LFCCLCNN-LibriSpeech75/25 Train"
python project/baseline_LA/main.py \
        --save-model-dir models/ \
        --verbose 0 --num-workers 6 \
        --epochs 100 --no-best-epochs 50 \
        --batch-size 64 \
        --sampler block_shuffle_by_length \
        --lr-decay-factor 0.5 \
        --lr-scheduler-type 1 --lr 0.0003 \
        --seed 1000 \
        --iteration_numbers 75-25

echo "Starting LFCCLCNN-LibriSpeech90/10 Train"
python project/baseline_LA/main.py \
        --save-model-dir models/ \
        --verbose 0 --num-workers 6 \
        --epochs 100 --no-best-epochs 50 \
        --batch-size 64 \
        --sampler block_shuffle_by_length \
        --lr-decay-factor 0.5 \
        --lr-scheduler-type 1 --lr 0.0003 \
        --seed 1000 \
        --iteration_numbers 90-10

conda deactivate

date