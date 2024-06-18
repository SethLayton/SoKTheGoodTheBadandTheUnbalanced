#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd CFAD/LFCC-LCNN/lfcc-lcnn


echo "Starting LFCC-LCNN Eval using 25-75 training distribution against WenetSpeech Real Only Dataset"
python test.py \
    --distribution 25-75 \
    --testlabelfile ../../RealOnlyEval/protocols/ro_eval.txt \
    --database_path ../../../../DataSets/WenetSpeech/wav_distributed_tst/ \
    --designator $1 \
    --real_only True

echo "Starting LFCC-LCNN Eval using 50-50 training distribution against WenetSpeech Real Only Dataset"
python test.py \
    --distribution 50-50 \
    --testlabelfile ../../RealOnlyEval/protocols/ro_eval.txt \
    --database_path ../../../../DataSets/WenetSpeech/wav_distributed_tst/ \
    --designator $1 \
    --real_only True

echo "Starting LFCC-LCNN Eval using 75-25 training distribution against WenetSpeech Real Only Dataset"
python test.py \
    --distribution 75-25 \
    --testlabelfile ../../RealOnlyEval/protocols/ro_eval.txt \
    --database_path ../../../../DataSets/WenetSpeech/wav_distributed_tst/ \
    --designator $1 \
    --real_only True

echo "Starting LFCC-LCNN Eval using 90-10 training distribution against WenetSpeech Real Only Dataset"
python test.py \
    --distribution 90-10 \
    --testlabelfile ../../RealOnlyEval/protocols/ro_eval.txt \
    --database_path ../../../../DataSets/WenetSpeech/wav_distributed_tst/ \
    --designator $1 \
    --real_only True

conda deactivate
date