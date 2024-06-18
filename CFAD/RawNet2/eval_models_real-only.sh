#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd CFAD/RawNet2/RawNet2

echo "Starting RawNet2 Eval using 25-75 training distribution against WenetSpeech Real Only Eval Dataset"
python score.py \
        --pretrained_model ../models/25-75.pth \
        --database_path ../../../../DataSets/WenetSpeech/wav_distributed_tst/ \
        --eval_protocol_path ../../RealOnlyEval/pkl/ro_eval.pkl \
        --output ../results/25-75_ro-eval.scores \
        --designator $1

echo "Starting RawNet2 Eval using 50-50 training distribution against WenetSpeech Real Only Eval Dataset"
python score.py \
        --pretrained_model ../models/50-50.pth \
        --database_path ../../../../DataSets/WenetSpeech/wav_distributed_tst/ \
        --eval_protocol_path ../../RealOnlyEval/pkl/ro_eval.pkl \
        --output ../results/50-50_ro-eval.scores \
        --designator $1

echo "Starting RawNet2 Eval using 75-25 training distribution against WenetSpeech Real Only Eval Dataset"
python score.py \
        --pretrained_model ../models/75-25.pth \
        --database_path ../../../../DataSets/WenetSpeech/wav_distributed_tst/ \
        --eval_protocol_path ../../RealOnlyEval/pkl/ro_eval.pkl \
        --output ../results/75-25_ro-eval.scores \
        --designator $1

echo "Starting RawNet2 Eval using 90-10 training distribution against WenetSpeech Real Only Eval Dataset"
python score.py \
        --pretrained_model ../models/90-10.pth \
        --database_path ../../../../DataSets/WenetSpeech/wav_distributed_tst/ \
        --eval_protocol_path ../../RealOnlyEval/pkl/ro_eval.pkl \
        --output ../results/90-10_ro-eval.scores \
        --designator $1

conda deactivate
date