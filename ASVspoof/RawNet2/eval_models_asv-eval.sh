#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd ASVspoof/RawNet2/RawNet2

echo "Starting RawNet2 Eval using 25-75 training distribution against ASVspoof Eval Dataset"
python score.py \
        --database_path ../../../../DataSets/ASVspoof/Eval/ASVspoof2021_DF_eval/flac/ \
        --pretrained_model ../models/25-75.pth \
        --eval_protocol_path ../../ASVspoofEval/pkl/asv21_eval.pkl \
        --output ../results/25-75_asv21-eval.scores \
        --designator $1


echo "Starting RawNet2 Eval using 50-50 training distribution against ASVspoof Eval Dataset"
python score.py \
        --database_path ../../../../DataSets/ASVspoof/Eval/ASVspoof2021_DF_eval/flac/ \
        --pretrained_model ../models/50-50.pth \
        --eval_protocol_path ../../ASVspoofEval/pkl/asv21_eval.pkl \
        --output ../results/50-50_asv21-eval.scores \
        --designator $1

echo "Starting RawNet2 Eval using 75-25 training distribution against ASVspoof Eval Dataset"
python score.py \
        --database_path ../../../../DataSets/ASVspoof/Eval/ASVspoof2021_DF_eval/flac/ \
        --pretrained_model ../models/75-25.pth \
        --eval_protocol_path ../../ASVspoofEval/pkl/asv21_eval.pkl \
        --output ../results/75-25_asv21-eval.scores \
        --designator $1


echo "Starting RawNet2 Eval using 90-10 training distribution against ASVspoof Eval Dataset"
python score.py \
        --database_path ../../../../DataSets/ASVspoof/Eval/ASVspoof2021_DF_eval/flac/ \
        --pretrained_model ../models/90-10.pth \
        --eval_protocol_path ../../ASVspoofEval/pkl/asv21_eval.pkl \
        --output ../results/90-10_asv21-eval.scores \
        --designator $1


conda deactivate

date