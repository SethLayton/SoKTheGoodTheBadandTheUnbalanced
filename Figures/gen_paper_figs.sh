#!/bin/sh

eval "$(conda shell.bash hook)"
conda activate pytorch-asvspoof2021

cd Figures/
mkdir figs

python gen_figs.py --figure paper


python gen_tables.py --table paper
