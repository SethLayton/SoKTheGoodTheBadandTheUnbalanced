#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd CFAD/LFCC-LCNN/lfcc-lcnn


echo "Starting LFCC-LCNN Eval using 25-75 training distribution against CFAD Eval Dataset"
python test.py \
    --distribution 25-75 \
    --testlabelfile ../../CFADEval/protocols/cfad_eval.txt \
    --database_path ../../../../DataSets/CFAD/tst/ \
    --designator $1

echo "Starting LFCC-LCNN Eval using 50-50 training distribution against CFAD Eval Dataset"
python test.py \
    --distribution 50-50 \
    --testlabelfile ../../CFADEval/protocols/cfad_eval.txt \
    --database_path ../../../../DataSets/CFAD/tst/ \
    --designator $1

echo "Starting LFCC-LCNN Eval using 75-25 training distribution against CFAD Eval Dataset"
python test.py \
    --distribution 75-25 \
    --testlabelfile ../../CFADEval/protocols/cfad_eval.txt \
    --database_path ../../../../DataSets/CFAD/tst/ \
    --designator $1

echo "Starting LFCC-LCNN Eval using 90-10 training distribution against CFAD Eval Dataset"
python test.py \
    --distribution 90-10 \
    --testlabelfile ../../CFADEval/protocols/cfad_eval.txt \
    --database_path ../../../../DataSets/CFAD/tst/ \
    --designator $1

conda deactivate
date