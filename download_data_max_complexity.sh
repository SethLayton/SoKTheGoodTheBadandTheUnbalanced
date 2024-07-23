#!/bin/bash


############## Download train/test datasets ##############

cd ../DataSets	

mkdir ASVspoof

wget "https://zenodo.org/records/12007844/files/asv_libri.tar.gz.00"
wget "https://zenodo.org/records/12007844/files/asv_libri.tar.gz.01"
cat asv_libri.tar.gz.* | tar xzvf - -C ASVspoof/

wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.00"
wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.01"
wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.02"
wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.03"
wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.04"
wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.05"
wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.06"
cat asv_realonly.tar.gz.* | tar xzvf - -C ASVspoof/

wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz"
wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz"
wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz"
wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz"
tar -xvzf ASVspoof2021_DF_eval_part00.tar.gz -C ASVspoof/
tar -xvzf ASVspoof2021_DF_eval_part01.tar.gz -C ASVspoof/
tar -xvzf ASVspoof2021_DF_eval_part02.tar.gz -C ASVspoof/
tar -xvzf ASVspoof2021_DF_eval_part03.tar.gz -C ASVspoof/


mkdir CFAD	

wget "https://zenodo.org/records/12089727/files/cfad_eval.tar.gz.00"
wget "https://zenodo.org/records/12089727/files/cfad_eval.tar.gz.01"
wget "https://zenodo.org/records/12089727/files/cfad_eval.tar.gz.02"
cat cfad_eval.tar.gz.* | tar xzvf - -C CFAD/

wget "https://zenodo.org/records/12089727/files/cfad_train.tar.gz.00"
wget "https://zenodo.org/records/12089727/files/cfad_train.tar.gz.01"
wget "https://zenodo.org/records/12089727/files/cfad_train.tar.gz.02"
wget "https://zenodo.org/records/12089727/files/cfad_train.tar.gz.03"
cat cfad_train.tar.gz.* | tar xzvf - -C CFAD/

wget "https://zenodo.org/records/12089727/files/wenet_real_only.tar.gz.00"
wget "https://zenodo.org/records/12089727/files/wenet_real_only.tar.gz.01"
cat wenet_real_only.tar.gz.* | tar xzvf -


mkdir CIFAKE

wget "https://zenodo.org/records/12090252/files/cifake2575.tar.gz"
tar -xvzf cifake2575.tar.gz -C CIFAKE/

wget "https://zenodo.org/records/12090252/files/cifake5050.tar.gz"
tar -xvzf cifake5050.tar.gz -C CIFAKE/

wget "https://zenodo.org/records/12090252/files/7525.tar.gz"
tar -xvzf cifake7525.tar.gz -C CIFAKE/

wget "https://zenodo.org/records/12090252/files/9010.tar.gz"
tar -xvzf cifake9010.tar.gz -C CIFAKE/

wget "https://zenodo.org/records/12090252/files/cifakeRO.tar.gz"
tar -xvzf cifakeRO.tar.gz -C CIFAKE/

wget "https://zenodo.org/records/12090252/files/cifaketest.tar.gz"
tar -xvzf cifaketest.tar.gz -C CIFAKE/


cd ../DataSets
wget "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"