#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate ssl1

cd ASVspoof/wav2vec/SSL_Anti-spoofing

echo "Starting wav2vec Eval using 25-75 training distribution against ASVspoof Eval Dataset"
python score_SSL_DF.py \
        --database_path ../../../../DataSets/ASVspoof/Eval/ASVspoof2021_DF_eval/flac/ \
        --pretrained_model ../25-75.pth \
        --eval_protocol_path ../../ASVspoofEval/protocols/asv21_eval.txt \
        --eval_output ../results/25-75_asv21-eval.score \
        --designator $1

echo "Starting wav2vec Eval using 50-50 training distribution against ASVspoof Eval Dataset"
python score_SSL_DF.py \
        --database_path ../../../../DataSets/ASVspoof/Eval/ASVspoof2021_DF_eval/flac/ \
        --pretrained_model ../50-50.pth \
        --eval_protocol_path ../../ASVspoofEval/protocols/asv21_eval.txt \
        --eval_output ../results/50-50_asv21-eval.score \
        --designator $1


echo "Starting wav2vec Eval using 75-25 training distribution against ASVspoof Eval Dataset"
python score_SSL_DF.py \
        --database_path ../../../../DataSets/ASVspoof/Eval/ASVspoof2021_DF_eval/flac/ \
        --pretrained_model ../75-25.pth \
        --eval_protocol_path ../../ASVspoofEval/protocols/asv21_eval.txt \
        --eval_output ../results/75-25_asv21-eval.score \
        --designator $1


echo "Starting wav2vec Eval using 90-10 training distribution against ASVspoof Eval Dataset"
python score_SSL_DF.py \
        --database_path ../../../../DataSets/ASVspoof/Eval/ASVspoof2021_DF_eval/flac/ \
        --pretrained_model ../90-10.pth \
        --eval_protocol_path ../../ASVspoofEval/protocols/asv21_eval.txt \
        --eval_output ../results/90-10_asv21-eval.score \
        --designator $1

conda deactivate
date